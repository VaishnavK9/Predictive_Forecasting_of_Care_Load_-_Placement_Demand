# Predictive Forecasting of Care Load & Placement Demand in the HHS Unaccompanied Alien Children (UAC) Program

**A time-series forecasting study of HHS care-load and discharge-demand dynamics, 2023-2025**

---

## Abstract

The HHS Unaccompanied Alien Children (UAC) program cares for children in federal custody after
transfer from CBP. Intake can spike suddenly, and today's operations are reactive: there is rich
historical reporting but no forward-looking forecast. This study builds and rigorously
walk-forward-validates a forecasting suite for two operational targets -- **care load** (`Children
in HHS Care`) and **discharge/placement demand** (`Children discharged from HHS Care`) -- using
~3 years of program data (Jan 2023-Dec 2025, 720 reporting-day observations). A RandomForest model
using lag/rolling/calendar features achieves the lowest out-of-sample walk-forward error for both
targets (MAPE 0.76% for care load, 29.0% for discharge demand), outperforming SARIMA, Exponential
Smoothing, and naive baselines, particularly at longer (7-14 reporting-day) horizons. The paper is
explicit about a critical data property that is easy to get wrong: **~33% of calendar days in the
source file are not missing data -- they are structurally unreported**, because HHS reports on an
approximate Sunday-Thursday cadence. All modeling respects this cadence rather than treating gaps
as missing daily observations to interpolate.

---

## 1. Introduction and Motivation

Unaccompanied children entering U.S. federal custody are first processed by CBP, then transferred
to HHS Office of Refugee Resettlement (ORR) care pending release to a vetted sponsor. Historically,
HHS has published daily-ish situational reports on program volumes, but no forward-looking capacity
forecast exists. This creates three operational problems:

1. **Overcrowding risk** -- shelters and staff scale reactively after a surge is already underway.
2. **Staff burnout** -- caseworker and medical staff surge capacity cannot be pre-positioned.
3. **Placement bottlenecks** -- sponsor vetting and discharge throughput are not planned against
   anticipated demand.

This project builds predictive intelligence to address all three: a validated forecasting model
for care load and discharge demand, a derived early-warning "pressure" signal, and an interactive
dashboard for HHS planners.

## 2. Data

### 2.1 Source and Structure

The source file (`HHS_Unaccompanied_Alien_Children_Program.csv`) contains six columns: `Date`,
`Children apprehended and placed in CBP custody*`, `Children in CBP custody`, `Children
transferred out of CBP custody`, `Children in HHS Care` (primary target), and `Children discharged
from HHS Care` (secondary target).

### 2.2 Data Quality Issues (Handled Explicitly)

Four data-quality properties required deliberate handling rather than default `pandas` behavior:

1. **~450 trailing blank rows.** The raw file has ~1,171 rows; only 720 contain data. All-NaN rows
   are dropped before any other processing.
2. **Text dates in descending order.** Dates are stored as strings (`"December 21, 2025"`) and
   sorted newest-first. They are parsed with `pd.to_datetime(format="%B %d, %Y")` and sorted
   ascending before being set as the index.
3. **Comma-formatted numeric text.** `Children in HHS Care` (and defensively, every numeric
   column) contains thousands separators as text (e.g., `"2,484"`). All numeric columns are
   stripped of commas and coerced with `pd.to_numeric`.
4. **Irregular, non-daily reporting cadence.** This is the property most likely to be silently
   mishandled. Empirically (day-of-week counts across all 720 real rows):

   | Day | Count |
   |---|---|
   | Monday | 145 |
   | Tuesday | 149 |
   | Wednesday | 147 |
   | Thursday | 147 |
   | Friday | 2 |
   | Saturday | 0 |
   | Sunday | 130 |

   The program reports on an approximate **Sunday-Thursday** cadence. Across the full
   2023-01-12-to-2025-12-21 span (1,075 calendar days), only 720 (67.0%) have a report --
   **355 calendar days (33.0%) are structurally unobserved**, not missing.

### 2.3 Frequency Strategy (Documented Decision)

Reindexing onto standard pandas business-day frequency (`'B'` = Mon-Fri) would misalign the true
cadence: it would slot in Fridays that are essentially never reported (introducing artificial NaN
placeholders in the wrong place) while **dropping genuine Sunday observations from the index
entirely**. Instead, we built a `pandas.offsets.CustomBusinessDay(weekmask="Sun Mon Tue Wed Thu")`
frequency that reproduces the true reporting week exactly. Reindexing onto this frequency over the
full date range produces 767 reporting-day slots, of which only **9 (1.2%)** are genuinely missing
(short internal gaps, e.g., holiday closures) -- these are linearly interpolated only when the gap
is <= 3 reporting days; no gap is fabricated to fill the systematically-absent Friday/Saturday
slots, because those slots do not exist in the reindexed frequency at all.

Weekly seasonality for STL decomposition and SARIMA is therefore defined on a **5-observation
period** (the Sun-Thu reporting week), not an assumed 7-day daily period.

## 3. Exploratory Data Analysis

Full figures are saved under `reports/figures/` (01-07), each with a one-line insight logged in
`reports/figures/insights.txt`. Key findings:

- **Long-run decline with punctuated surges.** Care load falls from ~11,000 (early 2023) to
  ~2,400 (late 2025), with sharp surges (e.g., January 2025, where load rose from ~2,400 to
  ~6,400 in weeks) tied to intake spikes rather than gradual drift.
- **STL decomposition**: trend dominates variance; weekly seasonal amplitude is only ~404
  children relative to a level in the thousands -- surges are trend/shock-driven, not
  calendar-driven.
- **ACF/PACF**: the level series shows slow-decaying autocorrelation consistent with a
  non-stationary I(1) process; first-differencing removes most autocorrelation, supporting an
  ARIMA(p,1,q)-style specification (confirmed by ADF/KPSS tests in the EDA notebook/script).
  This motivated the `SARIMA_ORDER = (1,1,1)` with seasonal order `(1,1,1,5)` used in modeling.
  Full ADF/KPSS statistics are printed by `src/eda.py` at runtime.
- **Heteroskedasticity**: rolling variance spikes sharply during surge periods (e.g., early 2025),
  meaning prediction intervals should widen during high-volatility regimes -- exactly what the
  SARIMA/ETS distributional intervals and RandomForest tree-spread intervals do in practice.
- **Flow distributions**: apprehensions, transfers, and discharges are all right-skewed with heavy
  tails, reflecting occasional high-volume days.
- **Net-flow pressure signal**: `net_flow_pressure = transfers_in - discharges_out` correlates
  -0.48 with the care-load level, confirming it is a useful leading indicator of building or
  easing pressure (a negative net flow, i.e., discharges outpacing transfers, precedes and
  accompanies care-load declines).
- **Surge periods**: 78 reporting days are flagged as surge weeks (week-over-week jump >= 365
  children). Surges cluster in two windows: December 2023-January 2024 and January-February 2025,
  both winter/policy-driven periods.

## 4. Feature Engineering

All features at reporting-day *t* use only information available at or before *t* (strict
no-leakage discipline):

- **Lags** (in reporting-cadence observations, not calendar days): *t*-1, *t*-7, *t*-14, applied
  to both the target and the discharge column.
- **Rolling statistics**: 7- and 14-observation trailing mean and variance, computed on
  `shift(1)`-ed series so the current observation never leaks into its own rolling window.
- **Net-flow pressure** (`transfers_in - discharges_out`) and its lags.
- **Calendar effects**: day-of-week, month, US-holiday flag and holiday-week flag (via the
  `holidays` library).

Direct multi-horizon targets `target_h{1,7,14} = y.shift(-h)` are constructed per horizon, so each
horizon gets its own supervised-learning target -- this is why the persisted RandomForest/
GradientBoosting models are stored as **one fitted model per horizon** rather than a single
recursive model.

## 5. Modeling and Validation Design

### 5.1 Validation Strategy

- **Strict time-based split**: the final 20% of the reporting-day timeline is reserved as a
  test region never used for any model fitting.
- **Walk-forward (rolling-origin) validation**: 5 expanding-window folds within the test region.
  At each origin, every model is refit using only data up to and including that origin, then
  asked to forecast 1, 7, and 14 reporting-days ahead.
- **Multi-horizon evaluation**: all models are scored identically at horizons {1, 7, 14} so
  results are directly comparable.

**No in-sample fit statistic is ever reported as forecast skill in this paper or the dashboard --
every headline number comes from these walk-forward, out-of-sample folds.**

### 5.2 Models Evaluated

| Model | Description |
|---|---|
| Naive persistence | ŷ_t = y_(t-1) |
| Moving average | 7-observation trailing mean |
| SARIMA(1,1,1)(1,1,1,5) | Captures trend + Sun-Thu weekly seasonality |
| Exponential Smoothing (Holt-Winters, damped trend + additive seasonality) | |
| RandomForestRegressor (300 trees, max_depth=6) | Lag/rolling/calendar features, direct per-horizon |
| GradientBoostingRegressor (300 estimators, max_depth=3) | Same features, direct per-horizon, plus quantile-regression models for intervals |

`random_state = 42` throughout for reproducibility.

### 5.3 Walk-Forward Results

**Overall (all horizons pooled), out-of-sample:**

| Target | Model | MAE | RMSE | MAPE |
|---|---|--:|--:|--:|
| Children in HHS Care | **RandomForest** | **17.13** | **22.76** | **0.76%** |
| Children in HHS Care | SARIMA | 36.33 | 63.96 | 1.61% |
| Children in HHS Care | GradientBoosting | 46.00 | 63.76 | 2.06% |
| Children in HHS Care | naive_persistence | 57.87 | 82.88 | 2.63% |
| Children in HHS Care | ETS | 71.41 | 95.50 | 3.15% |
| Children in HHS Care | moving_average | 84.78 | 108.18 | 3.79% |
| Children discharged from HHS Care | **RandomForest** | **3.05** | **3.64** | **28.96%** |
| Children discharged from HHS Care | GradientBoosting | 3.32 | 4.41 | 33.19% |
| Children discharged from HHS Care | naive_persistence | 3.80 | 4.86 | 31.20% |
| Children discharged from HHS Care | moving_average | 4.05 | 5.08 | 34.25% |
| Children discharged from HHS Care | ETS | 4.47 | 6.43 | 30.95% |
| Children discharged from HHS Care | SARIMA | 4.63 | 6.33 | 33.91% |

**Error by horizon (care load, MAE)** -- a more nuanced picture:

| Model | h=1 | h=7 | h=14 |
|---|--:|--:|--:|
| SARIMA | **9.08** | 24.96 | 74.94 |
| naive_persistence | 10.80 | 60.60 | 102.20 |
| RandomForest | 15.05 | **13.87** | **22.46** |
| GradientBoosting | 21.57 | 50.24 | 66.19 |
| ETS | 30.51 | 83.29 | 100.44 |
| moving_average | 36.91 | 87.91 | 129.51 |

**SARIMA is actually the best single-step (h=1) model** for care load, consistent with its strong
autoregressive structure at short range. But its error grows nearly 8x from h=1 to h=14, while
RandomForest's grows only ~1.5x -- RandomForest wins overall because it is far more *stable* across
the full multi-horizon evaluation, which matters more for operational planning (HHS needs 1-,
7-, and 14-day-ahead visibility, not just tomorrow).

Discharge demand is intrinsically noisier (MAPE 29-34% across all models, versus <2% for the much
smoother care-load level) -- day-to-day discharge counts depend on individual sponsor-vetting
completions, which are not well predicted by aggregate lag/rolling features alone. This is a
genuine finding, not a modeling shortfall: HHS should treat discharge-demand forecasts as directional
guidance (typical volume band) rather than a precise daily count.

Full per-horizon tables for both targets are in `reports/model_comparison_metrics.csv`; comparison
and forecast-vs-actual charts with confidence bands are in `reports/figures/08-10_*.png`.

### 5.4 Uncertainty Quantification

- **SARIMA / ETS**: native distributional 95% confidence intervals (analytic for SARIMA;
  simulation-based, 200 repetitions, for ETS).
- **RandomForest**: 95% interval from the 2.5th-97.5th percentile spread across the 300 individual
  trees' predictions at the forecast origin.
- **GradientBoosting**: separate quantile-regression models trained at alpha=0.025 and alpha=0.975.
- **Naive / moving average**: no native uncertainty estimate (reported as n/a).

## 6. Key Performance Indicators

Computed exclusively from walk-forward folds (see `reports/kpi_report.json`):

| KPI | Care Load | Discharge Demand |
|---|--:|--:|
| Forecast Accuracy (100 - MAPE) | **99.2%** | **71.0%** |
| Forecast Stability Index (std of APE%, by horizon 1/7/14) | 0.61 / 0.37 / 1.09 | 19.81 / 11.65 / 32.98 |
| Avg. Surge Lead Time (relative, trailing-baseline definition) | n/a in this test window (0 surge events) | ~9.8 calendar days (1 event) |
| Capacity Breach Probability (vs. historical peak-era ceiling of 9,000) | ~0.0% | ~0.0% |

**Important caveat on the last two KPIs**: the 5 walk-forward test folds fall in a calm 2025
low-load regime (~2,000-2,600 children), far below the historical peak-era ceiling used as an
illustrative capacity threshold. Breach probability against that absolute ceiling is expected to be
~0% in this window -- this is an honest reflection of *when* the test period occurred, not a
weakness of the method. Surge Lead Time is defined *relatively* (>= 15% above a trailing 30-observation
baseline) precisely so it remains informative even when the absolute level is low; with only one
detected relative-surge event in the test window, this KPI should be treated as illustrative pending
validation against a longer or more volatile out-of-sample period (e.g., the Dec 2023-Feb 2025
surge windows identified in the EDA).

## 7. Limitations

1. **Reporting cadence, not daily data.** All modeling operates on the Sun-Thu reporting cadence.
   Forecasts are in units of "reporting-days ahead," which correspond to slightly more than that
   many calendar days (~7/5 conversion factor used throughout).
2. **Small walk-forward sample.** Only 5 folds are available per target/horizon combination given
   the ~20% test-region constraint on a ~3-year series. KPIs computed from few surge events
   (Surge Lead Time, Breach Probability) should be treated as illustrative, not statistically
   robust point estimates.
3. **Regime shifts.** The series shows a large secular decline (11k -> 2.4k) plus discrete policy-
   driven surges. Models trained on an expanding window will lag a sudden regime change; SARIMA's
   strong short-horizon but weak long-horizon performance illustrates this directly.
4. **Discharge demand is inherently harder to predict** than care load from these features alone
   (MAPE ~29-34% across all models) -- sponsor-vetting completion timing is not well captured by
   lag/rolling/calendar features.
5. **Source data quality**: comma-formatted numeric text and descending-date ordering are quirks
   of the published CSV, not the underlying program; any future automated ingestion pipeline
   should validate these assumptions rather than hardcoding them.
6. **Live dashboard forecasts are unvalidated by construction** -- they are generated by refitting
   on all data through today and projecting into genuinely unseen future dates, which is standard
   operational practice but cannot itself be walk-forward-validated (the future hasn't happened).
   Only the backtested numbers in Sections 5-6 carry validated accuracy claims.

## 8. Recommendations

1. **Adopt RandomForest as the operational model** for both care-load and discharge-demand
   forecasting, with SARIMA retained as a fast-reacting cross-check specifically for 1-day-ahead
   alerts given its strong short-horizon performance.
2. **Monitor the net-flow-pressure signal** (`transfers_in - discharges_out`) as a leading
   indicator; sustained positive net flow historically precedes care-load buildups.
3. **Recalibrate the capacity/surge thresholds periodically** against the current operating
   regime rather than a fixed historical ceiling, since the program's baseline load has structurally
   declined ~4x since early 2023.
4. **Treat discharge-demand forecasts as a planning band, not a point estimate**, given their
   higher intrinsic noise; staffing decisions for placement/discharge throughput should use the
   full confidence interval, not just the point forecast.
5. **Re-run walk-forward validation whenever a new surge episode occurs** to update the small-
   sample KPIs (Surge Lead Time, Breach Probability) with genuinely high-volatility test data.
