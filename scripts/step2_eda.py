import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from config import FIGURES_DIR, require_parquet

# Set style for academic reporting
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

print("Loading optimized Parquet dataset...")
df = pd.read_parquet(require_parquet())

# --- 1. Identify Top 3 Areas + 4159 & 4556 ---
print("\nCalculating total traffic per grid square...")
area_totals = df.groupby('square_id')['internet_traffic'].sum().sort_values(ascending=False)

top_3 = area_totals.head(3)
print("\n" + "=" * 50)
print("TOP 3 HIGH-TRAFFIC SQUARE IDs:")
for rank, (sq_id, val) in enumerate(top_3.items(), 1):
    print(f"Rank {rank}: Square ID {sq_id} | Total Traffic: {val:,.2f}")
print("=" * 50)

top_1_id = top_3.index[0]
target_squares = list(top_3.index) + [4159, 4556]

# Also print totals for 4159/4556 for reference in the report
print("\nFor reference, total traffic for the two fixed comparison squares:")
for sq_id in [4159, 4556]:
    val = area_totals.get(sq_id, np.nan)
    print(f"Square ID {sq_id} | Total Traffic: {val:,.2f}")

# --- 2. Figure 1: Spatial Traffic Distribution ---
plt.figure(figsize=(10, 5))
sns.histplot(area_totals.values / 1e6, bins=100, kde=True, color='navy')
plt.title("Distribution of Total Internet Traffic Across Milan Grid Areas (2 Months)", fontsize=12, fontweight='bold')
plt.xlabel("Total Internet Traffic (Millions of Units)", fontsize=10)
plt.ylabel("Number of Grid Squares", fontsize=10)
plt.yscale('log')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig1_traffic_distribution.png", dpi=300)
plt.close()
print(f"Saved {FIGURES_DIR / 'fig1_traffic_distribution.png'}")

# --- 3. Two-week comparison across 5 areas (resampled consistently) ---
min_date = df['timestamp'].min()
two_weeks_end = min_date + pd.Timedelta(days=14)
df_2weeks = df[(df['timestamp'] >= min_date) & (df['timestamp'] <= two_weeks_end)]


def resample_area(df_subset):
    """Resample a single area's traffic onto a continuous 10-min grid
    and linearly interpolate any missing intervals, so comparisons
    across areas aren't distorted by differing data availability."""
    s = df_subset.set_index('timestamp')['internet_traffic']
    s = s.resample('10min').mean().interpolate(method='linear')
    return s


resampled_series = {}
for sq_id in target_squares:
    sub = df_2weeks[df_2weeks['square_id'] == sq_id][['timestamp', 'internet_traffic']]
    resampled_series[sq_id] = resample_area(sub)

# Fig 2a: overlay plot — good for comparing overall MAGNITUDE differences
plt.figure(figsize=(14, 8))
for sq_id in target_squares:
    s = resampled_series[sq_id]
    label_text = f"Square {sq_id} (Top 1)" if sq_id == top_1_id else f"Square {sq_id}"
    plt.plot(s.index, s.values, label=label_text, alpha=0.8, linewidth=1.2)
plt.title("Internet Traffic Dynamics: First 2 Weeks (Overlay, Same Scale)", fontsize=14, fontweight='bold')
plt.xlabel("Date", fontsize=11)
plt.ylabel("Internet Traffic Activity", fontsize=11)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2a_first_two_weeks_overlay.png", dpi=300)
plt.close()
print(f"Saved {FIGURES_DIR / 'fig2a_first_two_weeks_overlay.png'}")

# Fig 2b: small multiples — each area on its own scale, for comparing SHAPE/PATTERN
fig, axes = plt.subplots(len(target_squares), 1, figsize=(14, 14), sharex=True)
for ax, sq_id in zip(axes, target_squares):
    s = resampled_series[sq_id]
    label_text = f"Square {sq_id} (Top 1)" if sq_id == top_1_id else f"Square {sq_id}"
    ax.plot(s.index, s.values, linewidth=1.0, color='navy')
    ax.set_title(label_text, fontsize=10, loc='left')
    ax.set_ylabel("Traffic")
axes[-1].set_xlabel("Date")
fig.suptitle("Internet Traffic Dynamics: First 2 Weeks (Individual Scales)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2b_first_two_weeks_small_multiples.png", dpi=300)
plt.close()
print(f"Saved {FIGURES_DIR / 'fig2b_first_two_weeks_small_multiples.png'}")

# --- 4. Deep-dive statistical analysis on top area ---
print(f"\nRunning Deep-Dive Statistical Analysis on Top Area (Square ID {top_1_id})...")
df_top = df[df['square_id'] == top_1_id].sort_values('timestamp').set_index('timestamp')

df_top = df_top.resample('10min').mean().interpolate(method='linear')

adf_result = adfuller(df_top['internet_traffic'].dropna())
print("\n" + "=" * 50)
print(f"ADF STATIONARITY TEST RESULTS (Square ID {top_1_id}):")
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4e}")
print("Critical Values:")
for key, val in adf_result[4].items():
    print(f"   {key}: {val:.4f}")
print("=" * 50)

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
plot_acf(df_top['internet_traffic'], lags=144, ax=axes[0], title=f"Autocorrelation (ACF) - Square ID {top_1_id} (24h Window)")
plot_pacf(df_top['internet_traffic'], lags=144, ax=axes[1], title=f"Partial Autocorrelation (PACF) - Square ID {top_1_id} (24h Window)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig3_acf_pacf.png", dpi=300)
plt.close()
print(f"Saved {FIGURES_DIR / 'fig3_acf_pacf.png'}")

one_week = df_top.iloc[:144 * 7]
decomp = seasonal_decompose(one_week['internet_traffic'], model='additive', period=144)

fig = decomp.plot()
fig.set_size_inches(12, 8)
fig.suptitle(f"Seasonal Decomposition (1 Week) - Square ID {top_1_id}", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig4_seasonal_decomposition.png", dpi=300)
plt.close()
print(f"Saved {FIGURES_DIR / 'fig4_seasonal_decomposition.png'}")

print(f"\nTask 2 EDA complete! Check the generated images in: {FIGURES_DIR}")
