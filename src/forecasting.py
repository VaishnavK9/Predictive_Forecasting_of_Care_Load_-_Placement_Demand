"""Live future forecasting utilities shared by the Streamlit dashboard.

Reuses the exact fitting logic/hyperparameters from `src.train` so dashboard
forecasts are produced the same way as the walk-forward-validated models --
this module only extends them to genuinely unseen future reporting-days
(beyond the last observed date), which walk-forward validation never needed
to do, and adds an optional scenario shock for what-if analysis.
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src import config
from src.feature_engineering import build_feature_matrix, get_feature_columns
from src.train import (
    MOVING_AVG_WINDOW,
    N_CI_SIMULATIONS,
    SARIMA_ORDER,
    SARIMA_SEASONAL_ORDER,
    SEASONAL_PERIOD,
    fill_boundary_gaps,
)

MODEL_NAMES = ["naive_persistence", "moving_average", "SARIMA", "ETS", "RandomForest", "GradientBoosting"]
STAT_MODELS = {"naive_persistence", "moving_average", "SARIMA", "ETS"}

# Live dashboard forecasts refit a fresh model per horizon on every interaction
# (unlike src/train.py's one-time offline walk-forward validation), so we use
# lighter hyperparameters here specifically to keep peak memory/CPU low on
# constrained deployment containers. This does not change the validated
# walk-forward accuracy numbers reported elsewhere, only the live-forecast path.
LIVE_RF_N_ESTIMATORS = 100
LIVE_GBM_N_ESTIMATORS = 100
LIVE_GBM_QUANTILE_N_ESTIMATORS = 60


def future_reporting_index(last_date: pd.Timestamp, n_steps: int) -> pd.DatetimeIndex:
    reporting_day = pd.offsets.CustomBusinessDay(weekmask=config.REPORTING_WEEKMASK)
    return pd.date_range(last_date, periods=n_steps + 1, freq=reporting_day)[1:]


def apply_scenario_shock(df: pd.DataFrame, target_col: str, shock_pct: float, ramp_len: int = 7) -> pd.DataFrame:
    """Simulate a surge/decline scenario.

    Smoothly scales the most recent `ramp_len` observations of `target_col`
    (and the upstream flow columns that feed it) by up to `shock_pct`
    percent, ramped linearly from ~0% to `shock_pct`% so there is no
    discontinuity at the forecast origin. This is a simplified sensitivity
    tool for HHS planning ("what if intake were running X% hot/cold right
    now?"), not a causal simulation of border-crossing dynamics.
    shock_pct == 0 returns `df` unchanged.
    """
    if shock_pct == 0:
        return df
    df = df.copy()
    ramp = 1.0 + np.linspace(1.0 / ramp_len, 1.0, ramp_len) * (shock_pct / 100.0)
    idx = df.index[-ramp_len:]
    for col in (target_col, config.COL_TRANSFERRED, config.COL_DISCHARGED):
        if col in df:
            df.loc[idx, col] = df.loc[idx, col].to_numpy() * ramp
    if config.NET_FLOW_COL in df:
        df[config.NET_FLOW_COL] = df[config.COL_TRANSFERRED] - df[config.COL_DISCHARGED]
    return df


def _stat_model_path(series: pd.Series, model_name: str, max_h: int):
    if model_name == "naive_persistence":
        point = float(series.iloc[-1])
        means = [point] * max_h
        lowers = [np.nan] * max_h
        uppers = [np.nan] * max_h
    elif model_name == "moving_average":
        point = float(series.iloc[-MOVING_AVG_WINDOW:].mean())
        means = [point] * max_h
        lowers = [np.nan] * max_h
        uppers = [np.nan] * max_h
    elif model_name == "SARIMA":
        fitted = SARIMAX(
            series, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        fc = fitted.get_forecast(steps=max_h)
        means = fc.predicted_mean.tolist()
        ci = fc.conf_int(alpha=0.05)
        lowers, uppers = ci.iloc[:, 0].tolist(), ci.iloc[:, 1].tolist()
        del fitted, fc
    elif model_name == "ETS":
        fitted = ExponentialSmoothing(
            series, trend="add", damped_trend=True, seasonal="add",
            seasonal_periods=SEASONAL_PERIOD, initialization_method="estimated",
        ).fit()
        means = fitted.forecast(max_h).tolist()
        try:
            sims = fitted.simulate(nsimulations=max_h, repetitions=N_CI_SIMULATIONS, error="add")
            lowers = np.percentile(sims.values, 2.5, axis=1).tolist()
            uppers = np.percentile(sims.values, 97.5, axis=1).tolist()
            del sims
        except Exception:
            lowers = [np.nan] * max_h
            uppers = [np.nan] * max_h
        del fitted
    else:
        raise ValueError(f"Unknown statistical model: {model_name}")
    gc.collect()
    return means, lowers, uppers


def _ml_forecast(df: pd.DataFrame, target_col: str, model_name: str, horizons: list[int]):
    feats = build_feature_matrix(df, target_col=target_col, horizons=horizons)
    feature_cols = get_feature_columns(feats)
    feats[feature_cols] = feats[feature_cols].ffill().bfill()
    x_origin = feats[feature_cols].iloc[-1].values.astype(float)

    means, lowers, uppers = [], [], []
    for h in horizons:
        target_h_col = f"target_h{h}"
        ml_train = feats.iloc[:-1].dropna(subset=feature_cols + [target_h_col])
        X_train = ml_train[feature_cols].values
        y_train = ml_train[target_h_col].values

        if model_name == "RandomForest":
            # n_jobs=1: multiprocess fork() combined with threaded BLAS is a known
            # segfault trigger on constrained/shared cloud containers (e.g. Streamlit
            # Community Cloud's single-core instances).
            model = RandomForestRegressor(n_estimators=LIVE_RF_N_ESTIMATORS, max_depth=6,
                                           random_state=config.RANDOM_STATE, n_jobs=1)
            model.fit(X_train, y_train)
            tree_preds = np.array([t.predict(x_origin.reshape(1, -1))[0] for t in model.estimators_])
            means.append(float(tree_preds.mean()))
            lowers.append(float(np.percentile(tree_preds, 2.5)))
            uppers.append(float(np.percentile(tree_preds, 97.5)))
            del model, tree_preds
        elif model_name == "GradientBoosting":
            model = GradientBoostingRegressor(n_estimators=LIVE_GBM_N_ESTIMATORS, max_depth=3,
                                               learning_rate=0.05, random_state=config.RANDOM_STATE)
            model.fit(X_train, y_train)
            means.append(float(model.predict(x_origin.reshape(1, -1))[0]))
            del model

            lo_model = GradientBoostingRegressor(loss="quantile", alpha=0.025, n_estimators=LIVE_GBM_QUANTILE_N_ESTIMATORS,
                                                  max_depth=3, learning_rate=0.05, random_state=config.RANDOM_STATE)
            lo_model.fit(X_train, y_train)
            lowers.append(float(lo_model.predict(x_origin.reshape(1, -1))[0]))
            del lo_model

            hi_model = GradientBoostingRegressor(loss="quantile", alpha=0.975, n_estimators=LIVE_GBM_QUANTILE_N_ESTIMATORS,
                                                  max_depth=3, learning_rate=0.05, random_state=config.RANDOM_STATE)
            hi_model.fit(X_train, y_train)
            uppers.append(float(hi_model.predict(x_origin.reshape(1, -1))[0]))
            del hi_model
        else:
            raise ValueError(f"Unknown ML model: {model_name}")

        # Release each horizon's fitted model(s) before building the next --
        # matters on memory-constrained deployment containers where several
        # hundred trees/boosting-stages accumulating across horizons can be
        # the difference between fitting comfortably and being OOM-killed.
        gc.collect()

    return means, lowers, uppers


def get_future_forecast(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    horizons: list[int],
    shock_pct: float = 0.0,
) -> pd.DataFrame:
    """Forecast `target_col` at each horizon in `horizons` (reporting-days ahead
    of the last observed date), for the requested model, optionally under a
    scenario shock. Returns a DataFrame with columns
    [horizon, date, forecast, lower, upper] sorted by horizon.
    """
    df = apply_scenario_shock(df, target_col, shock_pct)
    max_h = max(horizons)
    last_date = df.index[-1]
    future_dates = future_reporting_index(last_date, max_h)

    if model_name in STAT_MODELS:
        series = fill_boundary_gaps(df[target_col].copy())
        means, lowers, uppers = _stat_model_path(series, model_name, max_h)
        sel = [h - 1 for h in horizons]
        out = pd.DataFrame({
            "horizon": horizons,
            "date": [future_dates[i] for i in sel],
            "forecast": [means[i] for i in sel],
            "lower": [lowers[i] for i in sel],
            "upper": [uppers[i] for i in sel],
        })
    else:
        means, lowers, uppers = _ml_forecast(df, target_col, model_name, horizons)
        out = pd.DataFrame({
            "horizon": horizons,
            "date": [future_dates[h - 1] for h in horizons],
            "forecast": means,
            "lower": lowers,
            "upper": uppers,
        })

    return out.sort_values("horizon").reset_index(drop=True)
