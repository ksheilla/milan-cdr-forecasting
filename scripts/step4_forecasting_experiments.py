import platform
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from config import FIGURES_DIR, RESULTS_DIR, require_parquet

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {platform.processor() or platform.machine()}, device = {DEVICE}")

print("Loading dataset...")
df = pd.read_parquet(require_parquet())

TARGET_SQUARES = [5161, 5059, 5259]
TUNING_SQUARE = 5161  # tune once on the top-traffic area to keep runtime tractable;
                       # the resulting hyperparameters are then reused for all 3 areas
                       # (stated explicitly here and in the report as a scope decision).
SEQ_LEN = 144  # 24 hours at 10-minute resolution

# Split points: tuning-train -> validation -> final test (assignment-mandated week)
VAL_START = '2013-12-09 00:00:00'
VAL_END = '2013-12-15 23:50:00'
TEST_START = '2013-12-16 00:00:00'
TEST_END = '2013-12-22 23:50:00'


# --- Model Definitions ---------------------------------------------------

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class Chomp1d(nn.Module):
    """Removes the extra right-side values introduced by symmetric padding,
    restoring strict causality: output at time t only depends on inputs
    at time <= t, never on future padding."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TCNForecaster(nn.Module):
    # RECEPTIVE FIELD BUG (found and fixed in this project):
    # This builds ONE convolution per entry of num_channels with dilation 2**i.
    # The original tuning grid only varied channel width and kernel size,
    # never depth (num_channels' length) — which is the parameter that
    # actually controls receptive field. With num_channels=(32, 64) and
    # kernel_size=3 (2 layers), the receptive field is:
    #     1 + (3-1)*1 + (3-1)*2 = 7 timesteps = 70 minutes,
    # even though SEQ_LEN = 144 timesteps (24 h) is fed in. 137 of the 144
    # inputs could not influence the output at all — no gradient path existed.
    #
    # Fix: the tcn_grid below now varies DEPTH (7 layers, kernel_size=3),
    # giving a receptive field of 255 timesteps — comfortably covering the
    # full 144-step window. All TCN results in results/ were regenerated
    # with this corrected grid.
    #
    # Note: correcting the receptive field did NOT improve accuracy — MAE was
    # roughly unchanged on square 5161 and worse on squares 5059/5259, while
    # training time increased ~5-6x. The likely cause: this TCN has no
    # residual connections (unlike Bai et al. 2018's full architecture), so a
    # 7-layer plain stack is harder to train than a shallow one, even with a
    # theoretically sufficient receptive field. See the report's Technical
    # Decision / Limitations sections for the full discussion.
    #
    # The default num_channels=(32, 64) below is only a fallback signature
    # default — every actual call in this script passes channels explicitly
    # via tcn_grid / best_tcn_cfg.
    def __init__(self, input_dim=1, num_channels=(32, 64), kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
                Chomp1d(padding),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        return self.fc(out[:, :, -1])


def compute_mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def build_sequences(scaled_series, seq_len):
    """Sliding-window sequence builder. scaled_series should already include
    seq_len steps of lookback context before the period you actually want
    predictions for (see how callers below construct it)."""
    X, y = [], []
    for i in range(seq_len, len(scaled_series)):
        X.append(scaled_series[i - seq_len:i, 0])
        y.append(scaled_series[i, 0])
    return np.array(X), np.array(y)


def add_lightgbm_features(X_lag, timestamps):
    """Augments raw lag features with calendar and rolling-window features,
    as justified in Task 3 (LightGBM has no innate sequence mechanism, so we
    hand-engineer the periodicity/context signal instead)."""
    hour = timestamps.hour.values
    dow = timestamps.dayofweek.values
    mean_last_hour = X_lag[:, -6:].mean(axis=1)     # last 6 x 10min = 1 hour
    std_last_hour = X_lag[:, -6:].std(axis=1)
    mean_full_window = X_lag.mean(axis=1)            # full 24h context
    extra = np.column_stack([hour, dow, mean_last_hour, std_last_hour, mean_full_window])
    return np.hstack([X_lag, extra])


def train_lstm(X_train, y_train, hidden_dim, num_layers, lr, epochs=10):
    model = LSTMForecaster(hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)
    model.train()
    t0 = time.time()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    train_time = time.time() - t0
    return model, train_time


def train_tcn(X_train, y_train, channels, kernel_size, lr, epochs=10):
    model = TCNForecaster(num_channels=channels, kernel_size=kernel_size).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)
    model.train()
    t0 = time.time()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    train_time = time.time() - t0
    return model, train_time


def predict_torch(model, X, scaler):
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_t).cpu().numpy()
    return scaler.inverse_transform(preds_scaled).flatten()


# --- Step A: Hyperparameter tuning (documented), on TUNING_SQUARE only ----

print("\n" + "=" * 60)
print(f"HYPERPARAMETER TUNING on Square {TUNING_SQUARE} (validation: {VAL_START} to {VAL_END})")
print("=" * 60)

sub_df_tune = df[df['square_id'] == TUNING_SQUARE].sort_values('timestamp').set_index('timestamp')
sub_df_tune = sub_df_tune.resample('10min').mean().interpolate(method='linear')

tune_train_series = sub_df_tune[:'2013-12-08 23:50:00']['internet_traffic'].values
val_series = sub_df_tune[VAL_START:VAL_END]['internet_traffic'].values

tune_scaler = MinMaxScaler()
tune_train_scaled = tune_scaler.fit_transform(tune_train_series.reshape(-1, 1))
val_scaled = tune_scaler.transform(val_series.reshape(-1, 1))

full_tune_scaled = np.vstack([tune_train_scaled, val_scaled])
X_tune_seq, y_tune_seq = build_sequences(full_tune_scaled, SEQ_LEN)
split_idx_tune = len(tune_train_series) - SEQ_LEN
X_tune_train, y_tune_train = X_tune_seq[:split_idx_tune], y_tune_seq[:split_idx_tune]
X_val, y_val = X_tune_seq[split_idx_tune:], y_tune_seq[split_idx_tune:]
y_val_orig = tune_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

tuning_log = []

# LSTM grid
lstm_grid = [
    {'hidden_dim': 32, 'num_layers': 1, 'lr': 0.001},
    {'hidden_dim': 64, 'num_layers': 2, 'lr': 0.001},
    {'hidden_dim': 64, 'num_layers': 2, 'lr': 0.0005},
]
best_lstm_cfg, best_lstm_rmse = None, np.inf
for cfg in lstm_grid:
    model, _ = train_lstm(X_tune_train, y_tune_train, **cfg, epochs=8)
    preds = predict_torch(model, X_val, tune_scaler)
    rmse = np.sqrt(mean_squared_error(y_val_orig, preds))
    mae = mean_absolute_error(y_val_orig, preds)
    tuning_log.append({'Model': 'LSTM', **cfg, 'Val_RMSE': rmse, 'Val_MAE': mae})
    print(f"  LSTM {cfg} -> val RMSE={rmse:.2f}, MAE={mae:.2f}")
    if rmse < best_lstm_rmse:
        best_lstm_rmse, best_lstm_cfg = rmse, cfg
print(f"  Selected LSTM config: {best_lstm_cfg} (val RMSE={best_lstm_rmse:.2f})")

# TCN grid
tcn_grid = [
    {'channels': (32,) * 7, 'kernel_size': 3, 'lr': 0.001},   # RF = 255 steps
    {'channels': (64,) * 7, 'kernel_size': 3, 'lr': 0.001},   # RF = 255 steps
    {'channels': (32,) * 6, 'kernel_size': 5, 'lr': 0.001},   # RF = 253 steps
]
best_tcn_cfg, best_tcn_rmse = None, np.inf
for cfg in tcn_grid:
    model, _ = train_tcn(X_tune_train, y_tune_train, **cfg, epochs=8)
    preds = predict_torch(model, X_val, tune_scaler)
    rmse = np.sqrt(mean_squared_error(y_val_orig, preds))
    mae = mean_absolute_error(y_val_orig, preds)
    tuning_log.append({'Model': 'TCN', **cfg, 'Val_RMSE': rmse, 'Val_MAE': mae})
    print(f"  TCN {cfg} -> val RMSE={rmse:.2f}, MAE={mae:.2f}")
    if rmse < best_tcn_rmse:
        best_tcn_rmse, best_tcn_cfg = rmse, cfg
print(f"  Selected TCN config: {best_tcn_cfg} (val RMSE={best_tcn_rmse:.2f})")

# LightGBM grid (with calendar/rolling features)
val_timestamps = sub_df_tune[VAL_START:VAL_END].index[:len(y_val_orig)]
X_tune_train_lgb = add_lightgbm_features(X_tune_train, sub_df_tune.index[SEQ_LEN:SEQ_LEN + len(X_tune_train)])
X_val_lgb = add_lightgbm_features(X_val, val_timestamps)

lgb_grid = [
    {'n_estimators': 100, 'learning_rate': 0.05, 'num_leaves': 31},
    {'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 31},
    {'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 63},
]
best_lgb_cfg, best_lgb_rmse = None, np.inf
for cfg in lgb_grid:
    model = lgb.LGBMRegressor(random_state=42, verbose=-1, **cfg)
    model.fit(X_tune_train_lgb, y_tune_train)
    preds_scaled = model.predict(X_val_lgb)
    preds = tune_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(y_val_orig, preds))
    mae = mean_absolute_error(y_val_orig, preds)
    tuning_log.append({'Model': 'LightGBM', **cfg, 'Val_RMSE': rmse, 'Val_MAE': mae})
    print(f"  LightGBM {cfg} -> val RMSE={rmse:.2f}, MAE={mae:.2f}")
    if rmse < best_lgb_rmse:
        best_lgb_rmse, best_lgb_cfg = rmse, cfg
print(f"  Selected LightGBM config: {best_lgb_cfg} (val RMSE={best_lgb_rmse:.2f})")

tuning_log_df = pd.DataFrame(tuning_log)
tuning_log_df.to_csv(RESULTS_DIR / "hyperparameter_tuning_log.csv", index=False)
print(f"\nFull tuning log saved to {RESULTS_DIR / 'hyperparameter_tuning_log.csv'}")

# --- Step B: Final training (train+val combined) and evaluation on test week ---

results = []
predictions_log = []  # per-timestep predictions, needed for Step 5 failure analysis

for sq_id in TARGET_SQUARES:
    print("\n" + "=" * 50)
    print(f"FINAL TRAINING & EVALUATION: SQUARE ID {sq_id}")
    print("=" * 50)

    sub_df = df[df['square_id'] == sq_id].sort_values('timestamp').set_index('timestamp')
    sub_df = sub_df.resample('10min').mean().interpolate(method='linear')

    train_series = sub_df[:'2013-12-15 23:50:00']['internet_traffic'].values  # tuning-train + val combined
    test_series = sub_df[TEST_START:TEST_END]['internet_traffic'].values

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
    test_scaled = scaler.transform(test_series.reshape(-1, 1))
    full_scaled = np.vstack([train_scaled, test_scaled])

    X_seq, y_seq = build_sequences(full_scaled, SEQ_LEN)
    split_idx = len(train_series) - SEQ_LEN
    X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
    X_test, y_test = X_seq[split_idx:], y_seq[split_idx:]
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    test_timestamps = sub_df[TEST_START:TEST_END].index[:len(y_test_orig)]

    # LightGBM (with best config + calendar/rolling features)
    train_timestamps_for_lgb = sub_df.index[SEQ_LEN:SEQ_LEN + len(X_train)]
    X_train_lgb = add_lightgbm_features(X_train, train_timestamps_for_lgb)
    X_test_lgb = add_lightgbm_features(X_test, test_timestamps)

    t0 = time.time()
    lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1, **best_lgb_cfg)
    lgb_model.fit(X_train_lgb, y_train)
    lgb_train_time = time.time() - t0
    t0 = time.time()
    lgb_preds_scaled = lgb_model.predict(X_test_lgb)
    lgb_exec_time = time.time() - t0
    lgb_preds = scaler.inverse_transform(lgb_preds_scaled.reshape(-1, 1)).flatten()

    # LSTM (best config)
    lstm_model, lstm_train_time = train_lstm(X_train, y_train, **best_lstm_cfg, epochs=10)
    t0 = time.time()
    lstm_preds = predict_torch(lstm_model, X_test, scaler)
    lstm_exec_time = time.time() - t0

    # TCN (best config)
    tcn_model, tcn_train_time = train_tcn(X_train, y_train, **best_tcn_cfg, epochs=10)
    t0 = time.time()
    tcn_preds = predict_torch(tcn_model, X_test, scaler)
    tcn_exec_time = time.time() - t0

    models_preds = {
        'LightGBM': (lgb_preds, lgb_train_time, lgb_exec_time),
        'LSTM': (lstm_preds, lstm_train_time, lstm_exec_time),
        'TCN': (tcn_preds, tcn_train_time, tcn_exec_time),
    }

    for name, (preds, tr_t, ex_t) in models_preds.items():
        mae = mean_absolute_error(y_test_orig, preds)
        rmse = np.sqrt(mean_squared_error(y_test_orig, preds))
        mape = compute_mape(y_test_orig, preds)

        results.append({
            'Square_ID': sq_id, 'Model': name, 'MAE': mae,
            'RMSE': rmse, 'MAPE (%)': mape,
            'Train Time (s)': tr_t, 'Inference Time (s)': ex_t
        })

        for ts, actual_val, pred_val in zip(test_timestamps, y_test_orig, preds):
            predictions_log.append({
                'Square_ID': sq_id, 'Model': name, 'timestamp': ts,
                'actual': actual_val, 'predicted': pred_val,
                'abs_error': abs(actual_val - pred_val)
            })

        plt.figure(figsize=(12, 5))
        plt.plot(test_timestamps, y_test_orig, label='Actual Traffic', alpha=0.7, color='black', linewidth=1.2)
        plt.plot(test_timestamps, preds, label=f'{name} Predicted', alpha=0.8, color='crimson', linestyle='--')
        plt.title(f'{name} One-Step Forecast vs Actual (Dec 16-22) - Square {sq_id}', fontsize=12, fontweight='bold')
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Internet Traffic', fontsize=10)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"forecast_{name}_sq{sq_id}.png", dpi=300)
        plt.close()

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_DIR / "model_evaluation_metrics.csv", index=False)

predictions_df = pd.DataFrame(predictions_log)
predictions_df.to_csv(RESULTS_DIR / "predictions_detailed.csv", index=False)
print(f"Per-timestep predictions saved to {RESULTS_DIR / 'predictions_detailed.csv'}")

print("\n" + "=" * 50)
print("EXPERIMENT EVALUATION SUMMARY")
print("=" * 50)
print(results_df.to_string(index=False))
print("=" * 50)
print(f"\nHyperparameters used (selected via validation on square {TUNING_SQUARE}):")
print(f"  LSTM: {best_lstm_cfg}")
print(f"  TCN: {best_tcn_cfg}")
print(f"  LightGBM: {best_lgb_cfg}")
print(f"\nForecasting pipeline finished! 9 forecast plots saved in {FIGURES_DIR},")
print(f"evaluation metrics and the full tuning log saved in {RESULTS_DIR}.")