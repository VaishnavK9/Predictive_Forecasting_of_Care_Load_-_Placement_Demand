"""HHS UAC Care-Load & Placement-Demand Forecasting Dashboard.

Run with: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os

# Must be set before numpy/scipy load their BLAS backend. Threaded BLAS combined
# with joblib's fork()-based multiprocessing is a known native-crash (segfault)
# trigger on constrained single-core containers such as Streamlit Community Cloud.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # headless/non-interactive backend -- required on Streamlit Cloud's server (no display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import threadpoolctl
from scipy.stats import norm

# Belt-and-suspenders alongside the OMP/OPENBLAS/MKL env vars above: those only
# take effect if set *before* numpy/scipy first load their BLAS library, which
# we cannot fully guarantee (Streamlit's own bootstrap may import numpy/pandas
# before this script runs). threadpoolctl patches already-loaded BLAS/OpenMP
# libraries directly, so it works regardless of import order.
threadpoolctl.threadpool_limits(limits=1)

from src import config
from src.data_loader import load_clean_dataset
from src.forecasting import MODEL_NAMES, get_future_forecast

st.set_page_config(page_title="HHS UAC Care-Load Forecasting", layout="wide", page_icon="\U0001F3E5")

TARGETS = {
    "Children in HHS Care (care load)": config.TARGET_CARE_LOAD,
    "Children discharged from HHS Care (discharge demand)": config.TARGET_DISCHARGE,
}


# ---------------------------------------------------------------------------
# Cached data / model access
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning dataset...")
def load_data() -> pd.DataFrame:
    _, bday_df = load_clean_dataset()
    return bday_df


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    path = config.REPORTS_DIR / "model_comparison_metrics.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_kpis() -> dict:
    path = config.REPORTS_DIR / "kpi_report.json"
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_walkforward_predictions() -> pd.DataFrame:
    path = config.REPORTS_DIR / "walkforward_predictions.csv"
    return pd.read_csv(path, parse_dates=["origin_date"])


@st.cache_data(show_spinner="Fitting model and generating forecast...")
def cached_forecast(target_col: str, model_name: str, horizons: tuple, shock_pct: float) -> pd.DataFrame:
    bday_df = load_data()
    return get_future_forecast(bday_df, target_col, model_name, list(horizons), shock_pct)


def best_model_for(target_col: str) -> str:
    metrics = load_metrics()
    sub = metrics[(metrics.target == target_col) & (metrics.horizon == "all")]
    return sub.sort_values("MAE").iloc[0]["model"]


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def fan_chart(history: pd.Series, forecast_df: pd.DataFrame, title: str, history_window: int = 90):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    hist = history.dropna().iloc[-history_window:]
    ax.plot(hist.index, hist.values, color="black", label="Historical (actual)")

    fc_dates = [hist.index[-1]] + forecast_df["date"].tolist()
    fc_vals = [hist.values[-1]] + forecast_df["forecast"].tolist()
    ax.plot(fc_dates, fc_vals, "o--", color="crimson", label="Forecast")

    if forecast_df["lower"].notna().all():
        lower = [hist.values[-1]] + forecast_df["lower"].tolist()
        upper = [hist.values[-1]] + forecast_df["upper"].tolist()
        ax.fill_between(fc_dates, lower, upper, color="crimson", alpha=0.15, label="95% Confidence Interval")

    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_ylabel("Children")
    fig.tight_layout()
    return fig


def overlay_fan_chart(history: pd.Series, baseline_df: pd.DataFrame, scenario_df: pd.DataFrame, title: str, history_window: int = 90):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    hist = history.dropna().iloc[-history_window:]
    ax.plot(hist.index, hist.values, color="black", label="Historical (actual)")

    base_dates = [hist.index[-1]] + baseline_df["date"].tolist()
    base_vals = [hist.values[-1]] + baseline_df["forecast"].tolist()
    ax.plot(base_dates, base_vals, "o--", color="steelblue", label="Baseline forecast")

    scen_dates = [hist.index[-1]] + scenario_df["date"].tolist()
    scen_vals = [hist.values[-1]] + scenario_df["forecast"].tolist()
    ax.plot(scen_dates, scen_vals, "o--", color="darkorange", label="Scenario forecast")

    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_ylabel("Children")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("Forecast Controls")

bday_df = load_data()
last_date = bday_df.index.max()
st.sidebar.caption(f"Data through **{last_date.date()}** (reporting cadence: Sun-Thu)")

horizon = st.sidebar.select_slider(
    "Primary forecast horizon (reporting-days ahead)",
    options=config.FORECAST_HORIZONS,
    value=7,
    help="Reporting-day horizons the models were trained and walk-forward-validated on. "
         "~5 reporting days per 7 calendar days under the Sun-Thu cadence.",
)

model_choice_label = st.sidebar.selectbox(
    "Model",
    options=["Best (auto)"] + MODEL_NAMES,
    index=0,
    help="'Best (auto)' uses the walk-forward-validated top model per target (currently RandomForest for both).",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Scenario Analysis")
shock_pct = st.sidebar.slider(
    "Simulated intake shock over the last reporting week (%)",
    min_value=-50, max_value=100, value=0, step=5,
    help="Smoothly scales the most recent week of care-load and flow data by this percentage "
         "before forecasting, to explore a 'what if intake surged/declined' scenario. "
         "Illustrative sensitivity analysis, not a causal simulation.",
)

capacity_threshold = st.sidebar.number_input(
    "Capacity threshold (for breach probability)",
    min_value=500, max_value=20000, value=config.CAPACITY_THRESHOLD, step=100,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Note: ~33% of calendar days in the source data are structurally unobserved "
    "(federal reporting is essentially Sun-Thu only). These are never fabricated -- "
    "see the Methodology tab for details."
)


# ---------------------------------------------------------------------------
# Header + top-line KPIs (from validated walk-forward backtests)
# ---------------------------------------------------------------------------
st.title("HHS UAC Care-Load & Placement-Demand Forecasting")
st.caption(
    "Predictive intelligence for the Unaccompanied Alien Children program: forecast care load, "
    "anticipate discharge/placement demand, and get early warning of capacity stress."
)

kpis = load_kpis()
care_kpi = kpis.get(config.TARGET_CARE_LOAD, {})
disch_kpi = kpis.get(config.TARGET_DISCHARGE, {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Care-Load Forecast Accuracy", f"{care_kpi.get('forecast_accuracy_pct', 'n/a')}%",
           help="100 - MAPE, computed ONLY from out-of-sample walk-forward folds.")
c2.metric("Discharge-Demand Forecast Accuracy", f"{disch_kpi.get('forecast_accuracy_pct', 'n/a')}%")
lead = care_kpi.get("avg_surge_lead_time_calendar_days")
c3.metric("Avg. Surge Lead Time", f"{lead} days" if lead else "no surges in test window",
           help="Advance warning (calendar days) the forecast would have given before a relative surge, "
                "measured across walk-forward test folds.")
c4.metric("Best Models", "RandomForest (both targets)",
           help="Selected by lowest walk-forward MAE across all evaluated models and horizons.")

st.markdown("---")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_care, tab_discharge, tab_compare, tab_ci, tab_scenario, tab_methodology = st.tabs([
    "Care-Load Forecast", "Discharge-Demand Forecast", "Model Comparison",
    "Confidence Intervals", "Scenario Comparison", "Methodology & Data Notes",
])

# --- (a) Future Care-Load Forecast Chart -----------------------------------
with tab_care:
    st.subheader("Future Care-Load Forecast")
    target = config.TARGET_CARE_LOAD
    model_name = best_model_for(target) if model_choice_label == "Best (auto)" else model_choice_label

    fc_all = cached_forecast(target, model_name, tuple(config.FORECAST_HORIZONS), shock_pct)
    fig = fan_chart(bday_df[target], fc_all, f"Care Load Forecast -- {model_name}"
                    + (f" (scenario shock {shock_pct:+d}%)" if shock_pct else ""))
    st.pyplot(fig)
    plt.close(fig)

    row = fc_all[fc_all.horizon == horizon].iloc[0]
    colA, colB, colC = st.columns(3)
    colA.metric(f"Forecast @ h={horizon}", f"{row['forecast']:.0f}")
    if pd.notna(row["lower"]):
        colB.metric("95% CI lower", f"{row['lower']:.0f}")
        colC.metric("95% CI upper", f"{row['upper']:.0f}")
    else:
        colB.metric("95% CI", "n/a for this model")

    if pd.notna(row["lower"]) and row["upper"] > row["lower"]:
        std_est = (row["upper"] - row["lower"]) / (2 * 1.96)
        breach_prob = 1 - norm.cdf((capacity_threshold - row["forecast"]) / std_est) if std_est > 0 else 0.0
        st.info(f"Estimated probability of exceeding the capacity threshold ({capacity_threshold:,}) "
                f"at horizon {horizon}: **{100 * breach_prob:.2f}%**")

    st.dataframe(fc_all.rename(columns={"horizon": "Horizon", "date": "Date", "forecast": "Forecast",
                                         "lower": "CI Lower", "upper": "CI Upper"}), use_container_width=True)

# --- (b) Discharge-Demand Forecast Panel -----------------------------------
with tab_discharge:
    st.subheader("Discharge (Placement) Demand Forecast")
    target = config.TARGET_DISCHARGE
    model_name = best_model_for(target) if model_choice_label == "Best (auto)" else model_choice_label

    fc_all = cached_forecast(target, model_name, tuple(config.FORECAST_HORIZONS), shock_pct)
    fig = fan_chart(bday_df[target], fc_all, f"Discharge Demand Forecast -- {model_name}"
                    + (f" (scenario shock {shock_pct:+d}%)" if shock_pct else ""))
    st.pyplot(fig)
    plt.close(fig)

    row = fc_all[fc_all.horizon == horizon].iloc[0]
    colA, colB, colC = st.columns(3)
    colA.metric(f"Forecast @ h={horizon}", f"{row['forecast']:.0f}")
    if pd.notna(row["lower"]):
        colB.metric("95% CI lower", f"{row['lower']:.0f}")
        colC.metric("95% CI upper", f"{row['upper']:.0f}")

    st.dataframe(fc_all.rename(columns={"horizon": "Horizon", "date": "Date", "forecast": "Forecast",
                                         "lower": "CI Lower", "upper": "CI Upper"}), use_container_width=True)

    st.markdown("##### Net Flow Pressure (Transfers In - Discharges Out)")
    net_flow = bday_df[config.NET_FLOW_COL].dropna().iloc[-90:]
    fig2, ax2 = plt.subplots(figsize=(11, 3.2))
    ax2.plot(net_flow.index, net_flow.values, color="teal")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_title("Positive = intake into HHS outpacing discharges (building pressure)")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# --- (c) Model Selection & Comparison ---------------------------------------
with tab_compare:
    st.subheader("Model Selection & Comparison (Walk-Forward Backtest)")
    target_label = st.selectbox("Target", list(TARGETS.keys()))
    target = TARGETS[target_label]

    metrics = load_metrics()
    sub = metrics[(metrics.target == target) & (metrics.horizon != "all")].copy()
    sub["horizon"] = sub["horizon"].astype(int)
    overall = metrics[(metrics.target == target) & (metrics.horizon == "all")].sort_values("MAE")

    st.markdown("**Overall (all horizons pooled), out-of-sample walk-forward MAE/RMSE/MAPE:**")
    st.dataframe(overall[["model", "MAE", "RMSE", "MAPE", "n"]].round(2).reset_index(drop=True),
                 use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(10, 4.5))
    for m in sub["model"].unique():
        m_sub = sub[sub.model == m].sort_values("horizon")
        ax3.plot(m_sub.horizon, m_sub.MAE, "o-", label=m)
    ax3.set_xlabel("Horizon (reporting-days ahead)")
    ax3.set_ylabel("MAE")
    ax3.set_title(f"Walk-Forward MAE by Horizon -- {target}")
    ax3.legend()
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.caption(
        "All numbers on this tab come exclusively from rolling-origin, out-of-sample walk-forward "
        "validation (5 folds x horizons 1/7/14) -- never from in-sample fit."
    )

# --- (d) Confidence-Interval Visualization ----------------------------------
with tab_ci:
    st.subheader("Forecast Uncertainty / Confidence Intervals")
    target_label = st.selectbox("Target ", list(TARGETS.keys()), key="ci_target")
    target = TARGETS[target_label]
    model_name_ci = st.selectbox("Model ", MODEL_NAMES, index=MODEL_NAMES.index(best_model_for(target)), key="ci_model")

    fc_all = cached_forecast(target, model_name_ci, tuple(config.FORECAST_HORIZONS), 0.0)
    fig4 = fan_chart(bday_df[target], fc_all, f"{model_name_ci} Forecast with 95% CI -- {target}")
    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown("**Interval width by horizon** (wider intervals = more forecast uncertainty further out):")
    fc_display = fc_all.copy()
    fc_display["interval_width"] = fc_display["upper"] - fc_display["lower"]
    st.dataframe(fc_display[["horizon", "date", "forecast", "lower", "upper", "interval_width"]]
                 .round(1), use_container_width=True)

    st.caption(
        "SARIMA/ETS intervals come from their native distributional forecast (95% analytic/simulated CI). "
        "RandomForest intervals use the spread across individual trees (2.5th-97.5th percentile). "
        "GradientBoosting intervals use separately trained 2.5%/97.5% quantile-regression models. "
        "Naive persistence and moving-average have no native uncertainty estimate (n/a)."
    )

# --- Scenario Comparison view -----------------------------------------------
with tab_scenario:
    st.subheader("Scenario Comparison: Baseline vs. Shocked Intake")
    target_label = st.selectbox("Target  ", list(TARGETS.keys()), key="scenario_target")
    target = TARGETS[target_label]
    model_name_sc = best_model_for(target) if model_choice_label == "Best (auto)" else model_choice_label

    baseline_df = cached_forecast(target, model_name_sc, tuple(config.FORECAST_HORIZONS), 0.0)
    scenario_df = cached_forecast(target, model_name_sc, tuple(config.FORECAST_HORIZONS), shock_pct)

    fig5 = overlay_fan_chart(bday_df[target], baseline_df, scenario_df,
                              f"Baseline vs. {shock_pct:+d}% Shock Scenario -- {model_name_sc} -- {target}")
    st.pyplot(fig5)
    plt.close(fig5)

    compare_df = baseline_df[["horizon", "date", "forecast"]].rename(columns={"forecast": "baseline_forecast"})
    compare_df["scenario_forecast"] = scenario_df["forecast"].values
    compare_df["difference"] = compare_df["scenario_forecast"] - compare_df["baseline_forecast"]
    compare_df["pct_difference"] = (compare_df["difference"] / compare_df["baseline_forecast"] * 100).round(1)
    st.dataframe(compare_df.round(1), use_container_width=True)

    if shock_pct == 0:
        st.info("Set a non-zero shock in the sidebar ('Scenario Analysis') to compare a surge/decline "
                "scenario against the baseline forecast.")

# --- Methodology & Data Notes -----------------------------------------------
with tab_methodology:
    st.subheader("Methodology & Data Notes")
    st.markdown(f"""
**Data coverage**: {len(bday_df)} reporting-day rows on a custom Sun-Thu reporting cadence,
from {bday_df.index.min().date()} to {bday_df.index.max().date()}.

**Why not a plain daily series?** The source data is reported by HHS on an approximate
Sun-Thu schedule -- Friday appears only twice and Saturday never appear across 720 real rows.
**~33% of calendar days in the covered range are structurally unobserved, not missing.**
We never interpolate or fabricate the systematically-absent Friday/Saturday slots. The series
is reindexed onto a `CustomBusinessDay(weekmask="Sun Mon Tue Wed Thu")` frequency that matches
the true cadence exactly; only short *internal* gaps (<= {config.MAX_GAP_INTERPOLATE} reporting
days, e.g. a holiday closure) are linearly interpolated.

**Validation**: every accuracy number shown in this dashboard's "Model Comparison" and top-line
KPI cards comes from **walk-forward (rolling-origin) validation** -- {config.N_WALK_FORWARD_FOLDS}
expanding-window folds, each re-forecasting horizons {config.FORECAST_HORIZONS} reporting-days
ahead using only data available at that fold's origin. No in-sample fit statistic is ever reported
as forecast skill.

**Live forecasts** (the charts in the Care-Load / Discharge-Demand / Confidence-Interval /
Scenario tabs) are generated by refitting the selected model on the full history available today
({bday_df.index.max().date()}) and projecting forward -- this is the standard way to deploy a
validated model operationally, but these specific future values are inherently unverified
(the true future hasn't happened yet).

**Scenario shocks** are a simplified sensitivity tool: they scale the most recent reporting week
of care-load/flow data by a chosen percentage before refitting/forecasting. They are not a causal
simulation of border-crossing or policy dynamics.
    """)
    st.markdown("**Best model per target** (from `models/best_models_summary.json`):")
    try:
        with open(config.MODELS_DIR / "best_models_summary.json") as f:
            st.json(json.load(f))
    except FileNotFoundError:
        st.write("Not available.")
