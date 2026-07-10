# Predictive Forecasting of Care Load & Placement Demand -- HHS UAC Program

Time-series forecasting suite + interactive dashboard for the HHS Unaccompanied Alien Children
(UAC) program: forecasts **care load** (`Children in HHS Care`), **discharge/placement demand**
(`Children discharged from HHS Care`), and a derived **net-flow pressure** early-warning signal.

See [`reports/research_paper.md`](reports/research_paper.md) for full methodology and results, and
[`reports/executive_summary.md`](reports/executive_summary.md) for a non-technical summary.

## Project Structure

```
data/                            raw + cleaned CSVs
src/
  config.py                      central config (paths, columns, seeds, thresholds)
  data_loader.py                 clean raw CSV, reindex to Sun-Thu reporting cadence
  eda.py                         plots + decomposition + stationarity tests -> reports/figures/
  feature_engineering.py         lag/rolling/calendar features, no-leakage direct multi-horizon targets
  train.py                       walk-forward validation across all models, persists best model/target
  evaluate.py                    metrics, comparison charts, KPI report (all from walk-forward folds)
  forecasting.py                 live future forecasting + scenario-shock utility (used by the dashboard)
app/
  streamlit_app.py               interactive dashboard
models/                          persisted best models + metadata JSON per target
reports/
  figures/                       EDA + evaluation figures (with insights.txt captions)
  research_paper.md
  executive_summary.md
  model_comparison_metrics.csv
  kpi_report.json
  walkforward_predictions.csv
```

## Setup

```bash
# from the project root
python -m venv venv
source venv/Scripts/activate        # Windows Git Bash / PowerShell: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place `HHS_Unaccompanied_Alien_Children_Program.csv` in `data/` (already present in this repo).

All randomized steps use `random_state = 42` (see `src/config.py`) for reproducibility.

## Run the Full Pipeline (in order)

```bash
python -m src.data_loader          # clean + validate data, print continuity/cadence report
python -m src.eda                  # EDA plots + decomposition -> reports/figures/
python -m src.feature_engineering  # sanity-check the feature matrix (also used by train.py)
python -m src.train                # walk-forward validation, saves models/ + walkforward_predictions.csv
python -m src.evaluate             # metrics tables, comparison charts, KPI report
```

Each step prints a validation report (shapes, gap counts, or score summary) so you can confirm it
ran correctly before moving to the next.

## Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Then open the printed local URL (default `http://localhost:8501`). The dashboard includes:

- **Care-Load Forecast** and **Discharge-Demand Forecast** panels with a fan chart (history +
  forecast + 95% CI) and a forecast table for horizons 1/7/14.
- **Model Selection & Comparison**: walk-forward MAE/RMSE/MAPE by model and horizon.
- **Confidence Intervals** tab: interval widths and their source per model type.
- **Scenario Comparison**: overlay a baseline forecast against a user-defined "intake shock" (%)
  scenario.
- **Methodology & Data Notes**: the Sun-Thu reporting-cadence explanation and validation
  discipline, in-app.

Sidebar controls: forecast horizon selector, model toggle (or "Best (auto)"), scenario shock
slider, and an editable capacity threshold for the breach-probability estimate.

## Key Data Caveat (Read This Before Changing Anything)

The source CSV reports on an approximate **Sunday-Thursday** cadence -- Friday appears twice and
Saturday never appear across 720 real rows. **~33% of calendar days in the covered date range are
structurally unobserved, not missing.** `src/data_loader.py` reindexes onto a
`CustomBusinessDay(weekmask="Sun Mon Tue Wed Thu")` frequency that matches this cadence exactly and
only interpolates short *internal* gaps (<= 3 reporting days). Do not "fix" apparent gaps with a
blanket daily reindex + interpolate -- that would fabricate roughly a third of the dataset.

## Validation Discipline

Every accuracy number reported in `reports/research_paper.md`, `reports/executive_summary.md`, the
KPI report, and the dashboard's "Model Comparison" / top-line metrics comes from **walk-forward
(rolling-origin), out-of-sample validation** (`src/train.py`, `reports/walkforward_predictions.csv`).
No in-sample fit statistic is ever presented as forecast skill. The dashboard's live forecast charts
project into genuinely future (unvalidated-by-construction) dates using the same fitting logic --
this is standard operational deployment practice, but is called out explicitly in the dashboard's
Methodology tab so it is never confused with a backtested accuracy claim.
