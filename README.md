# Milan CDR Traffic Forecasting

Comparative analysis of sequential models (LSTM, TCN, LightGBM) for
one-step-ahead mobile Internet traffic forecasting, using Call Detail Record
(CDR) data from Telecom Italia's Milan grid.

**Course/Assignment:** Machine Learning Techniques — Formative Assignment 1,
African Leadership University

## Overview

This project investigates one-step-ahead forecasting of mobile Internet
traffic across Milan's 10,000-cell grid, using ~2 months of Call Detail
Record (CDR) data at 10-minute resolution. Rather than simply identifying
the single best-performing model, the study compares three structurally
different sequential modelling approaches  recurrent (LSTM), convolutional
(TCN), and tree-based (LightGBM) and examines whether their relative
performance holds consistently across geographical areas with different
traffic characteristics.

**Research question:** How do different sequential models compare for
one-step-ahead mobile network traffic forecasting, and how does their
performance vary across geographical areas with different traffic
characteristics?

## Key Findings

- **Only LightGBM beat a naive persistence baseline.** Against `x̂(t+1) = x(t)`,
  LightGBM improved MAE by 4.1–12.5% across the three squares, while LSTM
  (−0.8% to −37.9%) and TCN (−12.8% to −94.6%) were both *worse than doing
  nothing*. Absolute error rankings alone are therefore misleading, and the
  skill scores are reported alongside them throughout.
- The ranking (LightGBM > LSTM > TCN) held consistently across all three
  squares, even though hyperparameters were tuned only on the
  highest-traffic square.
- Forecast difficulty varied substantially by area: the highest-traffic
  square (5161) was markedly harder to predict for every model than the
  other two, both in absolute error and in scale-normalized MAPE.
- Failure analysis found that the largest prediction errors, across all
  models, concentrated entirely on Square 5161's afternoon peak hours and
  were consistently under-predictions — consistent with the systematic
  bias of MSE-trained models toward the central tendency on volatile,
  spiky signals.
- A weekend-specific error hypothesis was tested and **not supported** by
  the data (see `results/weekend_vs_weekday_error.csv`).

## Repository Structure

\```
cdr-traffic-prediction/
├── scripts/
│   ├── step1_memory_optimization.py         # Raw data -> optimized Parquet
│   ├── step2_eda.py                         # Exploratory data analysis
│   ├── step3_related_work_model_selection.md # Related work review and model selection justification
│   ├── step4_forecasting_experiments.py     # Hyperparameter tuning + model training/eval
│   ├── step5_evaluation_failure_analysis.py # Comparative viz + failure analysis
│   ├── step6_baseline_comparison.py         # Naive baselines + forecast skill scores
├── figures/                     # Generated plots (created by running the scripts)
├── results/                     # Generated CSVs: metrics, tuning log, worst errors      # Full written report
├── task3_related_work_model_selection.md
├── .gitignore
└── README.md
\```

Raw data (`dataverse Files/`) and the generated `milan_internet_optimized.parquet`
are **not included in this repository** since they exceed GitHub's file size
limits and are excluded via `.gitignore`. See the Dataset section below to
regenerate them locally.

## Setup

Requires Python 3.10+.

\```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install pandas numpy matplotlib seaborn statsmodels scikit-learn lightgbm torch psutil pyarrow
\```

## Dataset

This project uses the **Telecommunications - SMS, Call, Internet - MI**
dataset from Harvard Dataverse:
https://doi.org/10.7910/DVN/EGZHFV

Part of the wider *"A multi-source dataset of urban life in the city of
Milan and the Province of Trentino"* collection (Barlacchi et al., 2015),
originally released for the Telecom Italia Big Data Challenge.

To obtain it:

1. Go to the dataset page above and select the 62
   `sms-call-internet-mi-*.txt` files you need.
2. You'll need to answer a short guestbook form (name, institution, purpose
   of use) before downloading, this is required by the dataset owner, not
   this project.
3. Place all 62 downloaded `.txt` files into a folder named `dataverse Files/`
   in the project root.

## Reproducing the Pipeline

Run scripts **from the project root** (not from inside `scripts/`), in this order:

\```bash
python scripts/step1_memory_optimization.py
python scripts/step2_eda.py
python scripts/step4_forecasting_experiments.py
python scripts/step5_evaluation_failure_analysis.py
python scripts/step6_baseline_comparison.py
\```

| Step | Script | What it does | Key outputs |
|---|---|---|---|
| 1 | `step1_memory_optimization.py` | Loads the 62 raw files, downcasts dtypes, aggregates across country codes, saves the optimized Parquet file. Includes a naive-vs-optimized memory comparison and continuous peak-memory tracking. | Console memory/timing summary |
| 2 | `step2_eda.py` | Identifies top-traffic grid squares, plots the traffic distribution and two-week comparisons (overlay + small multiples), runs ADF stationarity test, ACF/PACF, and seasonal decomposition. | `figures/fig1-fig4*.png` |
| 3 | *(no script — literature review + model justification)* | See `step3_related_work_model_selection.md` | — |
| 4 | `step4_forecasting_experiments.py` | Documented hyperparameter tuning (validation split, 3 configs per model) on the top square, then final training/evaluation of LSTM, TCN, and LightGBM across all 3 target squares on the Dec 16-22 test week. | `figures/forecast_*.png`, `results/model_evaluation_metrics.csv`, `results/hyperparameter_tuning_log.csv`, `results/predictions_detailed.csv` |
| 5 | `step5_evaluation_failure_analysis.py` | Comparative MAE bar chart, day-of-week failure analysis, weekend-vs-weekday hypothesis test, worst-error identification with a zoomed failure-case plot. | `figures/comparative_mae_barchart.png`, `figures/failure_*.png`, `results/weekend_vs_weekday_error.csv`, `results/top_worst_errors.csv` |
| 6 | `step6_baseline_comparison.py` | Scores the three models against naive reference forecasters (persistence, 24-hour seasonal naive, drift-damped persistence) and computes MAE skill scores, establishing whether each model adds value over a trivial predictor. Reads Step 4's saved predictions, so no retraining is required. | `results/baseline_comparison.csv`, `figures/baseline_skill_scores.png` |

**Note on runtime:** Step 4 includes hyperparameter tuning (9 additional
model trainings) before the final evaluation, and everything runs on CPU
only — expect this step to take several minutes.

## Methodology Summary

- **Sequence length:** 144 steps (24 hours at 10-minute resolution).
- **Preprocessing:** Per-square `MinMaxScaler`, fit only on training data.
- **Train/validation/test split:** Tuning-train → start of dataset to Dec 8;
  validation → Dec 9-15; final training set → start of dataset to Dec 15;
  test set → Dec 16-22 (never used during tuning).
- **Hyperparameter tuning:** Small grid search (3 configs per model),
  tuned once on the highest-traffic square and reused across all three
  target squares, to keep runtime tractable on CPU-only hardware.
- **LightGBM features:** Raw 144 lags plus hour-of-day, day-of-week,
  1-hour rolling mean/std, and full-window mean — since LightGBM has no
  native sequence-modelling mechanism.
- **Hardware:** CPU only (no GPU used for any experiment).

## Limitations

- The TCN implementation omits the residual/skip connections present in
  the full architecture proposed by Bai et al. (2018).
- All models, trained with MSE loss, systematically under-predict rare,
  high-magnitude traffic spikes.
- Hyperparameter tuning used a small grid, tuned on a single square rather
  than independently per square, due to compute constraints.

See the report's Conclusion and Future Work section for details and
proposed next steps.

## Dataset Citation

Barlacchi, G., De Nadai, M., Larcher, R., et al. (2015). A multi-source
dataset of urban life in the city of Milan and the Province of Trentino.
Scientific Data, 2, 150055. https://doi.org/10.1038/sdata.2015.55
