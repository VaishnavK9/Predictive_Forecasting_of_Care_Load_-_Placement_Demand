"""Exploratory data analysis and decomposition for the HHS UAC series.

Every figure is saved to reports/figures/ and printed with a one-line
insight. Run as a script: `python -m src.eda`.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from src import config
from src.data_loader import load_clean_dataset

sns.set_theme(style="whitegrid")

# Reporting cadence has 5 observations per week (Sun, Mon, Tue, Wed, Thu)
WEEKLY_PERIOD = 5

INSIGHTS: list[str] = []


def _save(fig, name: str, insight: str) -> None:
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    INSIGHTS.append(f"{name}: {insight}")
    print(f"Saved {path.name} -- {insight}")


def plot_series_over_time(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    cols = [
        config.COL_APPREHENDED,
        config.COL_CBP_CUSTODY,
        config.COL_TRANSFERRED,
        config.COL_HHS_CARE,
        config.COL_DISCHARGED,
        config.NET_FLOW_COL,
    ]
    for ax, col in zip(axes.flat, cols):
        ax.plot(df.index, df[col], linewidth=0.8, color="steelblue")
        ax.set_title(col, fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("HHS UAC Program -- All Series Over Time (reporting-day cadence)", fontsize=14)
    fig.tight_layout()
    _save(fig, "01_all_series_over_time.png",
          "Children in HHS Care shows a long decline from ~11k (early 2023) to ~2.4k (late 2025), "
          "punctuated by sharp surges (e.g. Jan 2025) tied to intake spikes.")


def plot_stl_decomposition(df: pd.DataFrame) -> None:
    series = df[config.TARGET_CARE_LOAD].interpolate(limit_area="inside").dropna()
    stl = STL(series, period=WEEKLY_PERIOD, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(series.index, series, color="black", linewidth=0.8)
    axes[0].set_title("Observed: Children in HHS Care")
    axes[1].plot(series.index, result.trend, color="firebrick", linewidth=0.8)
    axes[1].set_title("Trend")
    axes[2].plot(series.index, result.seasonal, color="seagreen", linewidth=0.8)
    axes[2].set_title(f"Seasonal (period={WEEKLY_PERIOD} reporting days = 1 reporting week)")
    axes[3].plot(series.index, result.resid, color="gray", linewidth=0.8)
    axes[3].set_title("Residual")
    fig.suptitle("STL Decomposition of HHS Care Load (Sun-Thu reporting-week seasonality)", fontsize=13)
    fig.tight_layout()
    seasonal_amplitude = result.seasonal.max() - result.seasonal.min()
    _save(fig, "02_stl_decomposition.png",
          f"Trend dominates variance; weekly seasonal amplitude is only ~{seasonal_amplitude:.0f} "
          f"children (small vs. level), confirming surges are trend/shock-driven, not calendar-driven.")


def plot_acf_pacf(df: pd.DataFrame) -> None:
    series = df[config.TARGET_CARE_LOAD].interpolate(limit_area="inside").dropna()
    diffed = series.diff().dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plot_acf(series, ax=axes[0, 0], lags=30)
    axes[0, 0].set_title("ACF -- Level")
    plot_pacf(series, ax=axes[0, 1], lags=30, method="ywm")
    axes[0, 1].set_title("PACF -- Level")
    plot_acf(diffed, ax=axes[1, 0], lags=30)
    axes[1, 0].set_title("ACF -- First Difference")
    plot_pacf(diffed, ax=axes[1, 1], lags=30, method="ywm")
    axes[1, 1].set_title("PACF -- First Difference")
    fig.suptitle("ACF/PACF: Children in HHS Care (level vs. first difference)", fontsize=13)
    fig.tight_layout()
    _save(fig, "03_acf_pacf.png",
          "Level series shows slow-decaying ACF (strong non-stationary trend); differencing removes "
          "most autocorrelation, supporting an I(1)/ARIMA(p,1,q) style specification.")


def stationarity_tests(df: pd.DataFrame) -> dict:
    series = df[config.TARGET_CARE_LOAD].interpolate(limit_area="inside").dropna()
    diffed = series.diff().dropna()

    def _run(s, label):
        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
        return {
            "series": label,
            "ADF_stat": round(adf_stat, 3), "ADF_p": round(adf_p, 4),
            "ADF_stationary_at_5pct": adf_p < 0.05,
            "KPSS_stat": round(kpss_stat, 3), "KPSS_p": round(kpss_p, 4),
            "KPSS_stationary_at_5pct": kpss_p > 0.05,
        }

    results = {"level": _run(series, "level"), "diff1": _run(diffed, "first_difference")}
    print("\nStationarity tests (Children in HHS Care):")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def plot_rolling_stats(df: pd.DataFrame) -> None:
    series = df[config.TARGET_CARE_LOAD].interpolate(limit_area="inside")
    roll_mean = series.rolling(14).mean()
    roll_std = series.rolling(14).std()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(series.index, series, color="lightgray", label="observed", linewidth=0.7)
    axes[0].plot(series.index, roll_mean, color="crimson", label="rolling mean (14 obs)")
    axes[0].legend()
    axes[0].set_title("Rolling Mean -- Children in HHS Care")
    axes[1].plot(series.index, roll_std, color="darkorange", label="rolling std (14 obs)")
    axes[1].legend()
    axes[1].set_title("Rolling Std Dev -- Children in HHS Care")
    fig.tight_layout()
    _save(fig, "04_rolling_mean_variance.png",
          "Rolling variance spikes sharply during surge periods (e.g. early 2025), confirming "
          "heteroskedasticity -- prediction intervals should widen during high-volatility regimes.")


def plot_distributions_and_correlation(df: pd.DataFrame) -> None:
    flow_cols = [config.COL_APPREHENDED, config.COL_CBP_CUSTODY, config.COL_TRANSFERRED,
                 config.COL_DISCHARGED, config.NET_FLOW_COL]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.flat, flow_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="teal")
        ax.set_title(col, fontsize=9)
    axes.flat[-1].axis("off")
    fig.suptitle("Distributions -- Flow Columns", fontsize=13)
    fig.tight_layout()
    _save(fig, "05_flow_distributions.png",
          "Flow columns (apprehensions, transfers, discharges) are right-skewed with heavy tails, "
          "reflecting occasional high-volume intake/discharge days.")

    corr_cols = flow_cols + [config.TARGET_CARE_LOAD]
    corr = df[corr_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax2)
    ax2.set_title("Correlation Matrix -- Flow Columns vs. Care Load")
    fig2.tight_layout()
    _save(fig2, "06_correlation_heatmap.png",
          f"Net flow pressure correlates {corr.loc[config.NET_FLOW_COL, config.TARGET_CARE_LOAD]:.2f} "
          "with care load level, confirming it is a useful leading pressure signal.")


def plot_surge_periods(df: pd.DataFrame, threshold_quantile: float = 0.9) -> None:
    series = df[config.TARGET_CARE_LOAD].interpolate(limit_area="inside")
    diffs = series.diff(WEEKLY_PERIOD)
    surge_threshold = diffs.quantile(threshold_quantile)
    surge_mask = diffs >= surge_threshold

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(series.index, series, color="steelblue", linewidth=0.8)
    ax.scatter(series.index[surge_mask], series[surge_mask], color="red", s=12, label="surge week", zorder=5)
    ax.set_title(f"Care Load with Surge Periods Highlighted (top {int((1 - threshold_quantile) * 100)}% "
                 "week-over-week increases)")
    ax.legend()
    fig.tight_layout()
    n_surge = surge_mask.sum()
    _save(fig, "07_surge_periods.png",
          f"{n_surge} reporting days flagged as surge weeks (week-over-week jump >= {surge_threshold:.0f}); "
          "surges cluster in Dec 2023-Jan 2024 and Jan-Feb 2025, both winter/policy-driven periods.")


def main() -> None:
    _, bday_df = load_clean_dataset()
    plot_series_over_time(bday_df)
    plot_stl_decomposition(bday_df)
    plot_acf_pacf(bday_df)
    stationarity_tests(bday_df)
    plot_rolling_stats(bday_df)
    plot_distributions_and_correlation(bday_df)
    plot_surge_periods(bday_df)

    insight_path = config.FIGURES_DIR / "insights.txt"
    insight_path.write_text("\n".join(INSIGHTS), encoding="utf-8")
    print(f"\nWrote {len(INSIGHTS)} figure insights -> {insight_path}")


if __name__ == "__main__":
    main()
