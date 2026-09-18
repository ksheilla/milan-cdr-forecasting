from __future__ import annotations
 
import os
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
PRED_PATH = os.path.join("results", "predictions_detailed.csv")
OUT_CSV = os.path.join("results", "baseline_comparison.csv")
OUT_FIG = os.path.join("figures", "baseline_skill_scores.png")
 
SEASONAL_PERIOD = 144  # 24 h at 10-minute resolution
MODEL_ORDER = ["LightGBM", "LSTM", "TCN"]
 
 
# --- metrics --------------------------------------------------------------
 
def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))
 
 
def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))
 
 
def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    """MAPE with an explicit, reported guard on small denominators.
 
    The original compute_mape() masked only exact zeros. No exact zeros occur
    in these three squares, so that guard never fires and the reported MAPE is
    sound *here* — but it would silently explode on a low-traffic square. We
    mask on a relative floor instead and report how many points were dropped.
    """
    floor = 0.01 * np.mean(actual)
    keep = actual > floor
    dropped = int((~keep).sum())
    if dropped:
        print(f"      [mape] excluded {dropped} points with actual <= {floor:.3f}")
    return float(np.mean(np.abs((actual[keep] - pred[keep]) / actual[keep])) * 100)
 
 
# --- benchmarks -----------------------------------------------------------
 
def build_benchmarks(y: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {name: (actual_aligned, pred_aligned)} for each naive forecaster.
 
    Each benchmark loses a different amount of warm-up at the start of the
    test week, so we return the aligned pair rather than assuming a common
    length. Comparisons against the models are made on the *intersection*
    below, so nobody is scored on a subset the others did not see.
    """
    return {
        "Persistence": (y[1:], y[:-1]),
        "SeasonalNaive_24h": (y[SEASONAL_PERIOD:], y[:-SEASONAL_PERIOD]),
        "DriftDamped": (y[2:], y[1:-1] + 0.5 * (y[1:-1] - y[:-2])),
    }
 
 
def main() -> None:
    if not os.path.exists(PRED_PATH):
        raise SystemExit(
            f"{PRED_PATH} not found. Run scripts/step4_forecasting_experiments.py first."
        )
 
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
 
    preds = pd.read_csv(PRED_PATH, parse_dates=["timestamp"])
    rows: list[dict] = []
 
    for square_id, per_square in preds.groupby("Square_ID"):
        # The actual series is identical across models; take it from any one.
        actual_series = (
            per_square[per_square["Model"] == MODEL_ORDER[0]]
            .sort_values("timestamp")
            .set_index("timestamp")["actual"]
        )
        y = actual_series.to_numpy()
 
        print(f"\n{'=' * 64}\nSquare {square_id}  (n = {len(y)} ten-minute intervals)\n{'=' * 64}")
 
        benchmarks = build_benchmarks(y)
 
        # Persistence defines the skill-score denominator. Every model is
        # re-scored on the SAME aligned subset (drop the first observation) so
        # the comparison is like-for-like.
        bench_actual, bench_pred = benchmarks["Persistence"]
        persistence_mae = mae(bench_actual, bench_pred)
        persistence_rmse = rmse(bench_actual, bench_pred)
 
        for name, (a, p) in benchmarks.items():
            rows.append(
                {
                    "Square_ID": square_id,
                    "Forecaster": name,
                    "Type": "benchmark",
                    "MAE": mae(a, p),
                    "RMSE": rmse(a, p),
                    "MAPE (%)": mape(a, p),
                    "MAE_skill_vs_persistence (%)": (1 - mae(a, p) / persistence_mae) * 100,
                    "n": len(a),
                }
            )
            print(f"  [benchmark] {name:18s} MAE={mae(a, p):8.2f}  RMSE={rmse(a, p):8.2f}")
 
        print()
        for model_name in MODEL_ORDER:
            g = per_square[per_square["Model"] == model_name].sort_values("timestamp")
            a = g["actual"].to_numpy()[1:]   # align with persistence
            p = g["predicted"].to_numpy()[1:]
            model_mae = mae(a, p)
            skill = (1 - model_mae / persistence_mae) * 100
            verdict = "beats persistence" if skill > 0 else "WORSE THAN PERSISTENCE"
            rows.append(
                {
                    "Square_ID": square_id,
                    "Forecaster": model_name,
                    "Type": "model",
                    "MAE": model_mae,
                    "RMSE": rmse(a, p),
                    "MAPE (%)": mape(a, p),
                    "MAE_skill_vs_persistence (%)": skill,
                    "n": len(a),
                }
            )
            print(
                f"  [model]     {model_name:18s} MAE={model_mae:8.2f}  RMSE={rmse(a, p):8.2f}"
                f"  skill={skill:+6.1f}%  -> {verdict}"
            )
 
        print(f"\n  Reference: persistence MAE={persistence_mae:.2f}, RMSE={persistence_rmse:.2f}")
 
    results = pd.DataFrame(rows)
    results.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")
 
    # --- skill-score figure ------------------------------------------------
    models_only = results[results["Type"] == "model"]
    pivot = models_only.pivot(
        index="Square_ID", columns="Forecaster", values="MAE_skill_vs_persistence (%)"
    )[MODEL_ORDER]
 
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pivot.index))
    width = 0.25
    for i, model_name in enumerate(MODEL_ORDER):
        values = pivot[model_name].to_numpy()
        bars = ax.bar(x + i * width, values, width, label=model_name)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (2 if value >= 0 else -6),
                f"{value:+.1f}%",
                ha="center",
                fontsize=8,
            )
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Square {sq}" for sq in pivot.index])
    ax.set_ylabel("MAE skill score vs persistence (%)")
    ax.set_title(
        "Forecast skill relative to the persistence baseline\n"
        "(above 0 = model adds value; below 0 = worse than doing nothing)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    plt.close()
    print(f"Saved {OUT_FIG}")
 
 
if __name__ == "__main__":
    main()
 