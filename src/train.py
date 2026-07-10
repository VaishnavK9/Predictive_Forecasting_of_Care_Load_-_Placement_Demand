"""Train and walk-forward-validate forecasting models for the HHS UAC series.

Validation design (documented, not incidental):
- Strict time-based split: the last TEST_SIZE_FRACTION of the reporting-day
  timeline is reserved as the out-of-sample test region. No model is ever
  fit on data from this region.
- Walk-forward (rolling-origin) validation: within the test region we roll
  the forecast origin forward across N_WALK_FORWARD_FOLDS positions. At
  each origin, every model is (re)trained using only data up to and
  including that origin (an expanding window), then asked to forecast
  1, 7, and 14 reporting-days ahead. This is the only source of the
  headline metrics computed later in evaluate.py -- no in-sample fit
  numbers are ever reported as forecast skill.
- Models compete on identical folds/horizons so their walk-forward errors
  are directly comparable.

Run as a script: `python -m src.train`.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src import config
from src.data_loader import load_clean_dataset
from src.feature_engineering import build_feature_matrix, get_feature_columns

warnings.filterwarnings("ignore")

SEASONAL_PERIOD = 5  # Sun-Thu reporting week
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, SEASONAL_PERIOD)
MOVING_AVG_WINDOW = 7
N_CI_SIMULATIONS = 200


@dataclass
class FoldResult:
    target: str
    model: str
    fold: int
    origin_date: pd.Timestamp
    horizon: int
    y_true: float
    y_pred: float
    y_lower: float = np.nan
    y_upper: float = np.nan


def fill_boundary_gaps(series: pd.Series) -> pd.Series:
    """Fill the handful of remaining boundary NaNs (edges the linear interpolation
    in data_loader could not reach) via forward/back-fill. Affects <2% of rows and
    is applied only so statistical models receive a fully-observed endog vector;
    it does not touch the systematically-absent Fri/Sat slots (already excluded
    from the reporting-day index entirely).
    """
    n_missing = series.isna().sum()
    if n_missing:
        print(f"  Filling {n_missing} boundary gap(s) in '{series.name}' via ffill/bfill "
              f"({100 * n_missing / len(series):.1f}% of rows).")
    return series.ffill().bfill()


def get_walk_forward_origins(n: int, max_horizon: int) -> list[int]:
    test_start = int(n * (1 - config.TEST_SIZE_FRACTION))
    last_origin = n - max_horizon - 1
    origins = np.linspace(test_start, last_origin, config.N_WALK_FORWARD_FOLDS)
    return sorted(set(int(round(o)) for o in origins))


def naive_persistence(train: pd.Series, horizon: int) -> float:
    return float(train.iloc[-1])


def moving_average(train: pd.Series, horizon: int, window: int = MOVING_AVG_WINDOW) -> float:
    return float(train.iloc[-window:].mean())


def sarima_forecast(train: pd.Series, horizon: int):
    model = SARIMAX(
        train, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    fc = fitted.get_forecast(steps=horizon)
    mean = float(fc.predicted_mean.iloc[-1])
    ci = fc.conf_int(alpha=0.05).iloc[-1]
    return mean, float(ci.iloc[0]), float(ci.iloc[1])


def ets_forecast(train: pd.Series, horizon: int):
    model = ExponentialSmoothing(
        train, trend="add", damped_trend=True, seasonal="add",
        seasonal_periods=SEASONAL_PERIOD, initialization_method="estimated",
    )
    fitted = model.fit()
    point = float(fitted.forecast(horizon).iloc[-1])
    try:
        sims = fitted.simulate(nsimulations=horizon, repetitions=N_CI_SIMULATIONS, error="add")
        final_step = sims.iloc[-1, :]
        lower, upper = float(np.percentile(final_step, 2.5)), float(np.percentile(final_step, 97.5))
    except Exception:
        lower, upper = np.nan, np.nan
    return point, lower, upper


def rf_forecast(X_train, y_train, x_origin) -> tuple[float, float, float]:
    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=config.RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    tree_preds = np.array([t.predict(x_origin.reshape(1, -1))[0] for t in model.estimators_])
    return float(tree_preds.mean()), float(np.percentile(tree_preds, 2.5)), float(np.percentile(tree_preds, 97.5))


def gbm_forecast(X_train, y_train, x_origin) -> tuple[float, float, float]:
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=config.RANDOM_STATE)
    model.fit(X_train, y_train)
    point = float(model.predict(x_origin.reshape(1, -1))[0])
    lo_model = GradientBoostingRegressor(loss="quantile", alpha=0.025, n_estimators=300, max_depth=3,
                                          learning_rate=0.05, random_state=config.RANDOM_STATE)
    hi_model = GradientBoostingRegressor(loss="quantile", alpha=0.975, n_estimators=300, max_depth=3,
                                          learning_rate=0.05, random_state=config.RANDOM_STATE)
    lo_model.fit(X_train, y_train)
    hi_model.fit(X_train, y_train)
    lower = float(lo_model.predict(x_origin.reshape(1, -1))[0])
    upper = float(hi_model.predict(x_origin.reshape(1, -1))[0])
    return point, lower, upper


def run_walk_forward_for_target(df: pd.DataFrame, target_col: str, horizons=config.FORECAST_HORIZONS) -> pd.DataFrame:
    print(f"\n{'=' * 70}\nWalk-forward validation for target: {target_col}\n{'=' * 70}")

    series = fill_boundary_gaps(df[target_col].copy())
    series.name = target_col

    feats = build_feature_matrix(df, target_col=target_col, horizons=horizons)
    feature_cols = get_feature_columns(feats)
    feats[feature_cols] = feats[feature_cols].ffill().bfill()

    n = len(series)
    max_h = max(horizons)
    origins = get_walk_forward_origins(n, max_h)
    print(f"n={n} reporting-day rows; walk-forward origins (row positions): {origins}")

    results: list[FoldResult] = []

    for fold_idx, origin_pos in enumerate(origins):
        origin_date = series.index[origin_pos]
        train_series = series.iloc[: origin_pos + 1]

        sarima_fit_cache, ets_fit_cache = None, None

        for h in horizons:
            target_pos = origin_pos + h
            if target_pos >= n:
                continue
            y_true = float(series.iloc[target_pos])

            # Baselines
            results.append(FoldResult(target_col, "naive_persistence", fold_idx, origin_date, h,
                                       y_true, naive_persistence(train_series, h)))
            results.append(FoldResult(target_col, "moving_average", fold_idx, origin_date, h,
                                       y_true, moving_average(train_series, h)))

            # SARIMA
            try:
                mean, lo, hi = sarima_forecast(train_series, h)
                results.append(FoldResult(target_col, "SARIMA", fold_idx, origin_date, h, y_true, mean, lo, hi))
            except Exception as e:
                print(f"  SARIMA failed at fold {fold_idx} h={h}: {e}")

            # ETS
            try:
                mean, lo, hi = ets_forecast(train_series, h)
                results.append(FoldResult(target_col, "ETS", fold_idx, origin_date, h, y_true, mean, lo, hi))
            except Exception as e:
                print(f"  ETS failed at fold {fold_idx} h={h}: {e}")

            # ML models: train only on rows whose target_h{h} is known no later than origin
            target_h_col = f"target_h{h}"
            ml_train = feats.iloc[: origin_pos + 1].dropna(subset=feature_cols + [target_h_col])
            if len(ml_train) >= 30:
                X_train = ml_train[feature_cols].values
                y_train = ml_train[target_h_col].values
                x_origin = feats.loc[origin_date, feature_cols].values.astype(float)

                try:
                    mean, lo, hi = rf_forecast(X_train, y_train, x_origin)
                    results.append(FoldResult(target_col, "RandomForest", fold_idx, origin_date, h, y_true, mean, lo, hi))
                except Exception as e:
                    print(f"  RandomForest failed at fold {fold_idx} h={h}: {e}")

                try:
                    mean, lo, hi = gbm_forecast(X_train, y_train, x_origin)
                    results.append(FoldResult(target_col, "GradientBoosting", fold_idx, origin_date, h, y_true, mean, lo, hi))
                except Exception as e:
                    print(f"  GradientBoosting failed at fold {fold_idx} h={h}: {e}")

        print(f"  Fold {fold_idx} (origin={origin_date.date()}) done.")

    results_df = pd.DataFrame([r.__dict__ for r in results])
    return results_df


def select_best_model(results_df: pd.DataFrame) -> str:
    mae_by_model = (
        results_df.assign(abs_err=lambda d: (d.y_true - d.y_pred).abs())
        .groupby("model")["abs_err"].mean()
        .sort_values()
    )
    print("\nMean walk-forward MAE by model (all horizons pooled):")
    print(mae_by_model.round(2))
    return mae_by_model.index[0]


def fit_final_model(df: pd.DataFrame, target_col: str, best_model_name: str, horizons=config.FORECAST_HORIZONS):
    """Refit the winning model type on all pre-holdout data (train+val region) and persist it."""
    series = fill_boundary_gaps(df[target_col].copy())
    n = len(series)
    test_start = int(n * (1 - config.TEST_SIZE_FRACTION))
    train_series = series.iloc[:test_start]

    feats = build_feature_matrix(df, target_col=target_col, horizons=horizons)
    feature_cols = get_feature_columns(feats)
    feats[feature_cols] = feats[feature_cols].ffill().bfill()

    safe_name = target_col.replace(" ", "_").replace("*", "")
    metadata = {
        "target": target_col,
        "best_model": best_model_name,
        "train_rows": int(test_start),
        "trained_through": str(train_series.index[-1].date()),
        "random_state": config.RANDOM_STATE,
        "horizons": horizons,
    }

    if best_model_name in ("SARIMA",):
        model = SARIMAX(train_series, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER,
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        metadata.update({"order": SARIMA_ORDER, "seasonal_order": SARIMA_SEASONAL_ORDER})
        joblib.dump(model, config.MODELS_DIR / f"{safe_name}_best_model.joblib")
    elif best_model_name in ("ETS",):
        model = ExponentialSmoothing(train_series, trend="add", damped_trend=True, seasonal="add",
                                      seasonal_periods=SEASONAL_PERIOD, initialization_method="estimated").fit()
        joblib.dump(model, config.MODELS_DIR / f"{safe_name}_best_model.joblib")
    elif best_model_name in ("RandomForest", "GradientBoosting"):
        # ML models only forecast at the discrete horizons they were trained on, so we
        # persist one fitted model per horizon (dict keyed by horizon) rather than a
        # single model -- needed for the dashboard's horizon selector to work for h=7/14.
        model = {}
        for h in horizons:
            target_h_col = f"target_h{h}"
            ml_train = feats.iloc[:test_start].dropna(subset=feature_cols + [target_h_col])
            X_train = ml_train[feature_cols].values
            y_train = ml_train[target_h_col].values
            if best_model_name == "RandomForest":
                m = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=config.RANDOM_STATE, n_jobs=-1)
            else:
                m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=config.RANDOM_STATE)
            m.fit(X_train, y_train)
            model[h] = m
        metadata["feature_columns"] = feature_cols
        metadata["model_type"] = "per_horizon_dict"
        joblib.dump(model, config.MODELS_DIR / f"{safe_name}_best_model.joblib")
    else:  # naive / moving average -- no persisted object needed beyond metadata
        model = None

    with open(config.MODELS_DIR / f"{safe_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Saved final '{best_model_name}' model for target '{target_col}' -> "
          f"{config.MODELS_DIR / (safe_name + '_best_model.joblib')}")
    return model, metadata


def main() -> None:
    _, bday_df = load_clean_dataset()

    all_results = []
    best_models = {}

    for target_col in (config.TARGET_CARE_LOAD, config.TARGET_DISCHARGE):
        results_df = run_walk_forward_for_target(bday_df, target_col)
        all_results.append(results_df)
        best_model_name = select_best_model(results_df)
        print(f"Best model for '{target_col}': {best_model_name}")
        _, metadata = fit_final_model(bday_df, target_col, best_model_name)
        best_models[target_col] = metadata

    combined = pd.concat(all_results, ignore_index=True)
    out_path = config.REPORTS_DIR / "walkforward_predictions.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved combined walk-forward predictions -> {out_path} ({len(combined)} rows)")

    with open(config.MODELS_DIR / "best_models_summary.json", "w") as f:
        json.dump(best_models, f, indent=2, default=str)


if __name__ == "__main__":
    main()
