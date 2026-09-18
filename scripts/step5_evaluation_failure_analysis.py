import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES_DIR, RESULTS_DIR

print("Loading Step 4 outputs...")
metrics_df = pd.read_csv(RESULTS_DIR / "model_evaluation_metrics.csv")
preds_df = pd.read_csv(RESULTS_DIR / "predictions_detailed.csv", parse_dates=['timestamp'])

MODEL_ORDER = ['LightGBM', 'LSTM', 'TCN']
SQUARES = sorted(preds_df['Square_ID'].unique())

# --- 1. Comparative bar chart: MAE per model, grouped by square ----------

pivot_mae = metrics_df.pivot(index='Square_ID', columns='Model', values='MAE')[MODEL_ORDER]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(pivot_mae.index))
width = 0.25
for i, model in enumerate(MODEL_ORDER):
    ax.bar(x + i * width, pivot_mae[model], width, label=model)
ax.set_xticks(x + width)
ax.set_xticklabels([f"Square {sq}" for sq in pivot_mae.index])
ax.set_ylabel("MAE")
ax.set_title("Model Comparison: MAE by Square", fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "comparative_mae_barchart.png", dpi=300)
plt.close()
print(f"Saved {FIGURES_DIR / 'comparative_mae_barchart.png'}")

# --- 2. Failure analysis: does error concentrate on weekends? ------------

preds_df['day_of_week'] = preds_df['timestamp'].dt.day_name()
preds_df['is_weekend'] = preds_df['timestamp'].dt.dayofweek >= 5

dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_summary = (
    preds_df.groupby(['Model', 'day_of_week'])['abs_error']
    .mean()
    .reset_index()
)
dow_summary['day_of_week'] = pd.Categorical(dow_summary['day_of_week'], categories=dow_order, ordered=True)
dow_summary = dow_summary.sort_values(['Model', 'day_of_week'])

print("\n" + "=" * 60)
print("MEAN ABSOLUTE ERROR BY DAY OF WEEK (averaged across all 3 squares)")
print("=" * 60)
print(dow_summary.pivot(index='day_of_week', columns='Model', values='abs_error')[MODEL_ORDER].to_string())

fig, ax = plt.subplots(figsize=(10, 5))
for model in MODEL_ORDER:
    sub = dow_summary[dow_summary['Model'] == model]
    ax.plot(sub['day_of_week'], sub['abs_error'], marker='o', label=model)
ax.set_ylabel("Mean Absolute Error")
ax.set_title("Failure Analysis: Mean Absolute Error by Day of Week", fontsize=13, fontweight='bold')
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "failure_dayofweek_analysis.png", dpi=300)
plt.close()
print(f"\nSaved {FIGURES_DIR / 'failure_dayofweek_analysis.png'}")

# Quantify the weekend-vs-weekday gap per model, as direct evidence for the report
weekend_gap = (
    preds_df.groupby(['Model', 'is_weekend'])['abs_error']
    .mean()
    .unstack()
    .rename(columns={False: 'Weekday_MAE', True: 'Weekend_MAE'})
)
weekend_gap['Weekend_vs_Weekday_pct_increase'] = (
    (weekend_gap['Weekend_MAE'] - weekend_gap['Weekday_MAE']) / weekend_gap['Weekday_MAE'] * 100
)
weekend_gap = weekend_gap.reindex(MODEL_ORDER)
weekend_gap.to_csv(RESULTS_DIR / "weekend_vs_weekday_error.csv")

print("\n" + "=" * 60)
print("WEEKEND VS. WEEKDAY ERROR (evidence for weekend-blindness hypothesis)")
print("=" * 60)
print(weekend_gap.to_string())
print("=" * 60)

# --- 3. Top individual worst-case errors, for concrete examples ----------

top_errors = preds_df.sort_values('abs_error', ascending=False).head(15)
top_errors.to_csv(RESULTS_DIR / "top_worst_errors.csv", index=False)
print(f"\nTop 15 single worst-case errors saved to {RESULTS_DIR / 'top_worst_errors.csv'}")
print(top_errors[['Square_ID', 'Model', 'timestamp', 'actual', 'predicted', 'abs_error']].to_string(index=False))

# --- 4. Zoomed-in plot around the single worst error, for a concrete example ---

worst = top_errors.iloc[0]
worst_sq, worst_model, worst_ts = worst['Square_ID'], worst['Model'], worst['timestamp']
window = preds_df[
    (preds_df['Square_ID'] == worst_sq) &
    (preds_df['Model'] == worst_model) &
    (preds_df['timestamp'] >= worst_ts - pd.Timedelta(hours=18)) &
    (preds_df['timestamp'] <= worst_ts + pd.Timedelta(hours=18))
].sort_values('timestamp')

plt.figure(figsize=(12, 5))
plt.plot(window['timestamp'], window['actual'], label='Actual', color='black', linewidth=1.3)
plt.plot(window['timestamp'], window['predicted'], label=f'{worst_model} Predicted', color='crimson', linestyle='--')
plt.axvline(worst_ts, color='gray', linestyle=':', label='Worst error point')
plt.title(f"Failure Case: {worst_model} on Square {worst_sq} (largest single error)", fontsize=12, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Internet Traffic")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "failure_case_zoom.png", dpi=300)
plt.close()
print(f"\nSaved {FIGURES_DIR / 'failure_case_zoom.png'} "
      f"(zoomed around {worst_model} / Square {worst_sq} at {worst_ts})")

print("\nStep 5 analysis complete.")