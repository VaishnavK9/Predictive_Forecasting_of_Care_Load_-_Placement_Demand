"""Evaluate walk-forward validated forecasts: metrics, plots, and KPIs.

Every number here comes from `reports/walkforward_predictions.csv`, which
contains ONLY out-of-sample, rolling-origin forecasts (see src/train.py).
No in-sample fit statistic is ever reported as forecast skill.

Run as a script: `python -m src.evaluate`.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

from src import config
from src.data_loader import load_clean_dataset

sns.set_theme(style="whitegrid")

SURGE_THRESHOLD_MULTIPLIER = 1.15  # "surge" = 15% above trailing 30-obs baseline
STRESS_BASELINE_WINDOW = 30


def load_predictions() -> pd.DataFrame:
    path = config.REPORTS_DIR / "walkforward_predictions.csv"
    df = pd.read_csv(path, parse_dates=["origin_date"])
    return df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    def _agg(g):
        err = g.y_true - g.y_pred
        mae = err.abs().mean()
        rmse = np.sqrt((err ** 2).mean())
        mape = (err.abs() / g.y_true.abs().replace(0, np.nan)).mean() * 100
        return pd.Series({"MAE": mae, "RMSE": rmse, "MAPE": mape, "n": len(g)})

    by_horizon = df.groupby(["target", "model", "horizon"]).apply(_agg, include_groups=False).reset_index()
    overall = df.groupby(["target", "model"]).apply(_agg, include_groups=False).reset_index()
    overall["horizon"] = "all"
    return pd.concat([by_horizon, overall], ignore_index=True)


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    for target in metrics["target"].unique():
        sub = metrics[(metrics.target == target) & (metrics.horizon != "all")].copy()
        sub["horizon"] = sub["horizon"].astype(int)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=sub, x="horizon", y="MAE", hue="model", ax=ax)
        ax.set_title(f"Walk-Forward MAE by Model and Horizon -- {target}")
        ax.set_ylabel("MAE (out-of-sample, walk-forward)")
        fig.tight_layout()
        safe = target.replace(" ", "_").replace("*", "")
        fig.savefig(config.FIGURES_DIR / f"08_model_comparison_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved model comparison chart for '{target}'")


def plot_forecast_vs_actual(df: pd.DataFrame, target: str, model: str, horizon: int = 7) -> None:
    sub = df[(df.target == target) & (df.model == model) & (df.horizon == horizon)].sort_values("origin_date")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sub.origin_date, sub.y_true, "o-", color="black", label="actual")
    ax.plot(sub.origin_date, sub.y_pred, "o--", color="crimson", label="forecast")
    if sub["y_lower"].notna().any():
        ax.fill_between(sub.origin_date, sub.y_lower, sub.y_upper, color="crimson", alpha=0.15, label="95% CI")
    ax.set_title(f"{horizon}-step-ahead Forecast vs. Actual -- {model} -- {target}")
    ax.legend()
    fig.tight_layout()
    safe = target.replace(" ", "_").replace("*", "")
    fig.savefig(config.FIGURES_DIR / f"09_forecast_vs_actual_{safe}_{model}_h{horizon}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved forecast-vs-actual chart: {target} / {model} / h={horizon}")


def plot_residual_diagnostics(df: pd.DataFrame, target: str, model: str) -> None:
    sub = df[(df.target == target) & (df.model == model)].copy()
    if sub.empty:
        return
    sub["residual"] = sub.y_true - sub.y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(sub["residual"], kde=True, ax=axes[0], color="slateblue")
    axes[0].axvline(0, color="black", linestyle="--")
    axes[0].set_title("Residual Distribution")

    axes[1].scatter(sub.y_pred, sub.residual, alpha=0.6, color="slateblue")
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set_xlabel("Forecast")
    axes[1].set_ylabel("Residual (actual - forecast)")
    axes[1].set_title("Residual vs. Forecast")

    fig.suptitle(f"Residual Diagnostics -- {model} -- {target}")
    fig.tight_layout()
    safe = target.replace(" ", "_").replace("*", "")
    fig.savefig(config.FIGURES_DIR / f"10_residual_diagnostics_{safe}_{model}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved residual diagnostics: {target} / {model}")


def kpi_forecast_accuracy(metrics: pd.DataFrame, target: str, model: str) -> dict:
    row = metrics[(metrics.target == target) & (metrics.model == model) & (metrics.horizon == "all")]
    if row.empty:
        return {}
    mape = float(row["MAPE"].iloc[0])
    return {"forecast_accuracy_pct": round(100 - mape, 1), "mape_pct": round(mape, 1)}


def kpi_forecast_stability_index(df: pd.DataFrame, target: str, model: str) -> dict:
    """Variance of walk-forward forecast error across folds, per horizon (lower = more stable)."""
    sub = df[(df.target == target) & (df.model == model)].copy()
    sub["abs_pct_err"] = (sub.y_true - sub.y_pred).abs() / sub.y_true.abs().replace(0, np.nan) * 100
    stability = sub.groupby("horizon")["abs_pct_err"].std().round(2).to_dict()
    return {"forecast_stability_index_by_horizon_pct": stability}


def kpi_surge_lead_time_and_breach_probability(
    raw_series: pd.Series, df: pd.DataFrame, target: str, model: str
) -> dict:
    """Surge = actual value exceeds SURGE_THRESHOLD_MULTIPLIER x trailing baseline.

    Surge Lead Time: among folds where a surge occurs within the forecast
    horizon window but was NOT already occurring at the origin, the smallest
    horizon h at which the model's forecast itself also exceeds the same
    threshold (i.e., how many reporting-days of advance warning the forecast
    would have given), converted to approximate calendar days (reporting
    cadence ~= 5 days per 7 calendar days -> factor 7/5).

    Capacity Breach Probability: for each fold/horizon, P(forecast > CAPACITY_THRESHOLD)
    using a normal approximation from the model's own point forecast + 95% CI width.
    """
    baseline = raw_series.rolling(STRESS_BASELINE_WINDOW).mean()
    sub = df[(df.target == target) & (df.model == model)].copy()

    lead_times = []
    for origin_date, grp in sub.groupby("origin_date"):
        if origin_date not in baseline.index:
            continue
        base = baseline.loc[origin_date]
        if pd.isna(base):
            continue
        threshold = base * SURGE_THRESHOLD_MULTIPLIER
        grp = grp.sort_values("horizon")
        origin_surging = False
        if origin_date in raw_series.index:
            origin_surging = raw_series.loc[origin_date] > threshold
        if origin_surging:
            continue
        actual_surge_horizons = grp[grp.y_true > threshold]["horizon"]
        if actual_surge_horizons.empty:
            continue
        forecast_surge_horizons = grp[grp.y_pred > threshold]["horizon"]
        if not forecast_surge_horizons.empty:
            lead_times.append(int(forecast_surge_horizons.min()))

    avg_lead_reporting_days = float(np.mean(lead_times)) if lead_times else None
    avg_lead_calendar_days = round(avg_lead_reporting_days * 7 / 5, 1) if avg_lead_reporting_days else None

    breach_probs = []
    for _, row in sub.iterrows():
        if pd.isna(row.y_lower) or pd.isna(row.y_upper):
            continue
        std_est = (row.y_upper - row.y_lower) / (2 * 1.96)
        if std_est <= 0:
            continue
        prob = 1 - norm.cdf((config.CAPACITY_THRESHOLD - row.y_pred) / std_est)
        breach_probs.append(prob)

    return {
        "surge_definition": f">= {SURGE_THRESHOLD_MULTIPLIER}x trailing {STRESS_BASELINE_WINDOW}-obs baseline",
        "n_surge_events_detected_in_test_folds": len(lead_times),
        "avg_surge_lead_time_reporting_days": avg_lead_reporting_days,
        "avg_surge_lead_time_calendar_days": avg_lead_calendar_days,
        "capacity_threshold": config.CAPACITY_THRESHOLD,
        "mean_capacity_breach_probability_pct": round(100 * np.mean(breach_probs), 4) if breach_probs else None,
        "note": (
            "Test folds fall in a 2025 low-load regime (~2,000-2,600) far below the historical "
            f"peak-era capacity ceiling of {config.CAPACITY_THRESHOLD}; breach probability against that "
            "absolute ceiling is expected to be ~0% in this window. Surge lead time uses a relative, "
            "trailing-baseline definition so it remains informative even when the absolute level is low."
        ),
    }


def build_kpi_report(df: pd.DataFrame, metrics: pd.DataFrame, bday_df: pd.DataFrame) -> dict:
    report = {}
    for target in df["target"].unique():
        target_metrics = metrics[(metrics.target == target) & (metrics.horizon == "all")]
        best_model = target_metrics.sort_values("MAE").iloc[0]["model"]
        raw_series = bday_df[target].ffill().bfill()

        report[target] = {
            "best_model": best_model,
            **kpi_forecast_accuracy(metrics, target, best_model),
            **kpi_forecast_stability_index(df, target, best_model),
            **kpi_surge_lead_time_and_breach_probability(raw_series, df, target, best_model),
        }
    return report


def main() -> None:
    df = load_predictions()
    _, bday_df = load_clean_dataset()

    metrics = compute_metrics(df)
    metrics_path = config.REPORTS_DIR / "model_comparison_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved metrics table -> {metrics_path}")
    print("\nOverall (all-horizon) comparison:")
    print(metrics[metrics.horizon == "all"].sort_values(["target", "MAE"]).to_string(index=False))

    plot_model_comparison(metrics)

    for target in df["target"].unique():
        best_model = (
            metrics[(metrics.target == target) & (metrics.horizon == "all")]
            .sort_values("MAE").iloc[0]["model"]
        )
        for h in config.FORECAST_HORIZONS:
            plot_forecast_vs_actual(df, target, best_model, horizon=h)
        plot_residual_diagnostics(df, target, best_model)

    kpis = build_kpi_report(df, metrics, bday_df)
    kpi_path = config.REPORTS_DIR / "kpi_report.json"
    with open(kpi_path, "w") as f:
        json.dump(kpis, f, indent=2, default=str)
    print(f"\nSaved KPI report -> {kpi_path}")
    print(json.dumps(kpis, indent=2, default=str))


if __name__ == "__main__":
    main()
