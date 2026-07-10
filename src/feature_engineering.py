"""Feature engineering for the HHS UAC forecasting models.

All features at row t are built using information available at or before
time t (lags, trailing rolling windows, calendar attributes of t itself).
Targets for horizon h are y.shift(-h) -- strictly future values relative
to the feature row -- so there is no leakage between features and targets.

Lags/rolling windows are expressed in units of *reporting-cadence
observations* (the business-day-like Sun-Thu index produced by
`src.data_loader`), not raw calendar days, since the series is not daily.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as holidays_lib
    _US_HOLIDAYS = holidays_lib.UnitedStates()
except ImportError:  # pragma: no cover
    _US_HOLIDAYS = None

from src import config

LAGS = (1, 7, 14)
ROLLING_WINDOWS = (7, 14)


def add_lag_features(df: pd.DataFrame, col: str, lags=LAGS) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str, windows=ROLLING_WINDOWS) -> pd.DataFrame:
    """Rolling mean/variance computed strictly on *past* values (shift(1) first)."""
    df = df.copy()
    shifted = df[col].shift(1)
    for w in windows:
        df[f"{col}_rollmean{w}"] = shifted.rolling(w).mean()
        df[f"{col}_rollvar{w}"] = shifted.rolling(w).var()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_holiday"] = 0
    if _US_HOLIDAYS is not None:
        df["is_holiday"] = df.index.to_series().apply(lambda d: int(d in _US_HOLIDAYS)).values
    df["is_holiday_week"] = 0
    if _US_HOLIDAYS is not None:
        df["is_holiday_week"] = (
            df.index.to_series()
            .apply(lambda d: int(any((d + pd.Timedelta(days=k)) in _US_HOLIDAYS for k in range(-3, 4))))
            .values
        )
    return df


def add_net_flow_features(df: pd.DataFrame, lags=LAGS) -> pd.DataFrame:
    """Net-pressure signal (transfers in - discharges out) and its lags.

    The contemporaneous value at t is known at the forecast origin t, so it
    is a valid feature for predicting any future horizon h >= 1; lagged
    versions are added for additional stability signal.
    """
    df = df.copy()
    if config.NET_FLOW_COL not in df:
        df[config.NET_FLOW_COL] = df[config.COL_TRANSFERRED] - df[config.COL_DISCHARGED]
    for lag in lags:
        df[f"{config.NET_FLOW_COL}_lag{lag}"] = df[config.NET_FLOW_COL].shift(lag)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = config.TARGET_CARE_LOAD,
    horizons=config.FORECAST_HORIZONS,
) -> pd.DataFrame:
    """Build the full feature matrix plus one target column per horizon.

    Returns a DataFrame with feature columns and `target_h{h}` columns for
    each horizon in `horizons`. Rows with insufficient lag/rolling history
    or missing future target (at the tail) contain NaN and should be
    dropped by the caller as appropriate for train vs. inference.
    """
    feats = df.copy()
    feats = add_lag_features(feats, target_col)
    feats = add_rolling_features(feats, target_col)
    feats = add_lag_features(feats, config.COL_DISCHARGED)
    feats = add_rolling_features(feats, config.COL_DISCHARGED)
    feats = add_net_flow_features(feats)
    feats = add_calendar_features(feats)

    for h in horizons:
        feats[f"target_h{h}"] = feats[target_col].shift(-h)

    return feats


def get_feature_columns(feats: pd.DataFrame) -> list[str]:
    """Feature columns = everything except raw target-adjacent leak-prone columns and target_h*."""
    exclude = {
        config.COL_HHS_CARE,
        config.COL_DISCHARGED,
        config.COL_TRANSFERRED,
        config.COL_APPREHENDED,
        config.COL_CBP_CUSTODY,
        config.NET_FLOW_COL,
    }
    return [c for c in feats.columns if c not in exclude and not c.startswith("target_h")]


def main() -> None:
    from src.data_loader import load_clean_dataset

    _, bday_df = load_clean_dataset()
    feats = build_feature_matrix(bday_df)
    feature_cols = get_feature_columns(feats)

    print(f"Feature matrix shape (pre-dropna): {feats.shape}")
    print(f"Number of feature columns: {len(feature_cols)}")
    print("Feature columns:", feature_cols)

    usable = feats.dropna(subset=feature_cols + [f"target_h{h}" for h in config.FORECAST_HORIZONS])
    print(f"Usable rows after dropping NaN (lag warm-up + tail horizon): {usable.shape[0]}")
    print("\nSample of feature matrix:")
    print(usable[feature_cols[:6] + ["target_h1"]].head())


if __name__ == "__main__":
    main()
