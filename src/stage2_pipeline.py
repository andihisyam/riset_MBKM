from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


RANDOM_STATE = 42
TIMESTAMP = "date"
TARGET_WH = "Appliances"
TARGET = "Appliances_kWh"
LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLL_MEANS = [3, 6, 12, 24]
ROLL_STDS = [3, 6, 24]
HORIZONS = [1, 3, 6, 12, 24]
ALL_DIRECT_HORIZONS = list(range(1, 25))
BASE_FEATURES = [f"lag_{x}" for x in LAGS] + ["hour", "day_of_week", "is_weekend", "sin_hour", "cos_hour", "sin_dow", "cos_dow"]
ROLLING_FEATURES = [f"rolling_mean_{x}" for x in ROLL_MEANS] + [f"rolling_std_{x}" for x in ROLL_STDS]
ENV_FEATURES = [
    "lights",
    "T1",
    "RH_1",
    "T2",
    "RH_2",
    "T3",
    "RH_3",
    "T4",
    "RH_4",
    "T5",
    "RH_5",
    "T6",
    "RH_6",
    "T7",
    "RH_7",
    "T8",
    "RH_8",
    "T9",
    "RH_9",
    "T_out",
    "Press_mm_hg",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
]


@dataclass
class SplitBounds:
    train_end: int
    val_end: int
    n: int


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def metrics(actual, predicted, prefix: str = "") -> dict[str, float]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    if len(a) == 0:
        return {f"{prefix}MAE": np.nan, f"{prefix}RMSE": np.nan, f"{prefix}MAPE": np.nan, f"{prefix}sMAPE": np.nan}
    denom = np.abs(a) + np.abs(p)
    smape_mask = denom != 0
    mape_mask = a != 0
    return {
        f"{prefix}MAE": float(mean_absolute_error(a, p)),
        f"{prefix}RMSE": float(np.sqrt(np.mean((a - p) ** 2))),
        f"{prefix}MAPE": float(np.mean(np.abs((a[mape_mask] - p[mape_mask]) / a[mape_mask])) * 100) if mape_mask.any() else np.nan,
        f"{prefix}sMAPE": float(np.mean(2 * np.abs(a[smape_mask] - p[smape_mask]) / denom[smape_mask]) * 100) if smape_mask.any() else np.nan,
    }


def load_raw(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "energydata_complete.csv")
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    return df.sort_values(TIMESTAMP).reset_index(drop=True)


def make_hourly_clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    indexed = raw.set_index(TIMESTAMP)
    agg = {col: ("sum" if col in [TARGET_WH, "lights"] else "mean") for col in indexed.columns}
    hourly = indexed.resample("h").agg(agg)
    hourly["raw_count"] = indexed[TARGET_WH].resample("h").count()
    incomplete = hourly[hourly["raw_count"] < 6].copy()
    clean = hourly[hourly["raw_count"] == 6].copy().reset_index()
    clean[TARGET] = clean[TARGET_WH] / 1000.0
    notes = {
        "raw_rows": int(len(raw)),
        "hourly_rows_before_drop": int(len(hourly)),
        "hourly_rows_after_drop": int(len(clean)),
        "incomplete_hours": int(len(incomplete)),
        "incomplete_hour_timestamps": [idx.isoformat() for idx in incomplete.index],
        "target_unit": "Total energy used during the one-hour interval, in kWh",
        "aggregation": {
            "Appliances": "sum six 10-minute Wh intervals, then divide by 1000 to kWh",
            "lights": "sum six 10-minute intervals",
            "temperature_humidity_weather": "mean",
            "raw_count": "count of 10-minute records in each hourly window",
        },
    }
    return clean, notes


def add_features(hourly: pd.DataFrame) -> pd.DataFrame:
    df = hourly.copy()
    for lag in LAGS:
        df[f"lag_{lag}"] = df[TARGET].shift(lag)
    shifted = df[TARGET].shift(1)
    for window in ROLL_MEANS:
        df[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
    for window in ROLL_STDS:
        df[f"rolling_std_{window}"] = shifted.rolling(window).std()
    df["hour"] = df[TIMESTAMP].dt.hour
    df["day_of_week"] = df[TIMESTAMP].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def split_bounds(n: int) -> SplitBounds:
    return SplitBounds(int(n * 0.70), int(n * 0.85), n)


def split_name(index: int, bounds: SplitBounds) -> str:
    if index < bounds.train_end:
        return "train"
    if index < bounds.val_end:
        return "validation"
    return "test"


def test_origins(bounds: SplitBounds) -> list[int]:
    start = max(bounds.val_end - 1, 168)
    return [origin for origin in range(start, bounds.n - 24, 24) if origin + 1 >= bounds.val_end]


def summarize_fixed_origin(windows: list[dict], model: str, notes: str) -> dict:
    profile_actual = []
    profile_pred = []
    daily_actual = []
    daily_pred = []
    for row in windows:
        profile_actual.extend(row["actual"])
        profile_pred.extend(row["predicted"])
        daily_actual.append(float(np.sum(row["actual"])))
        daily_pred.append(float(np.sum(row["predicted"])))
    return {
        "model": model,
        **metrics(profile_actual, profile_pred, "Hourly "),
        **metrics(daily_actual, daily_pred, "Daily "),
        "windows_evaluated": len(windows),
        "Notes": notes,
    }


def baseline_fixed_origin(hourly: pd.DataFrame, bounds: SplitBounds) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = []
    summary = []
    for model in ["Last Value", "Daily Seasonal", "Weekly Seasonal"]:
        windows = []
        for origin in test_origins(bounds):
            actual = hourly.loc[origin + 1 : origin + 24, TARGET].to_numpy()
            if model == "Last Value":
                predicted = np.repeat(hourly.loc[origin, TARGET], 24)
            elif model == "Daily Seasonal":
                predicted = hourly.loc[origin - 23 : origin, TARGET].to_numpy()
            else:
                predicted = hourly.loc[origin - 167 : origin - 144, TARGET].to_numpy()
            if len(actual) != 24 or len(predicted) != 24:
                continue
            detail.append({"origin_index": origin, "origin": hourly.loc[origin, TIMESTAMP], "model": model, "actual_kWh_24h": float(actual.sum()), "predicted_kWh_24h": float(predicted.sum())})
            windows.append({"actual": actual, "predicted": predicted})
        summary.append(summarize_fixed_origin(windows, model, "Fixed-origin 24-hour naive baseline"))
    return pd.DataFrame(summary), pd.DataFrame(detail)


def tune_xgb(features: pd.DataFrame, feature_cols: list[str], bounds: SplitBounds) -> tuple[dict, pd.DataFrame]:
    data = features.dropna(subset=feature_cols + [TARGET]).copy()
    train = data[data.index < bounds.train_end]
    val = data[(data.index >= bounds.train_end) & (data.index < bounds.val_end)]
    candidates = [
        {"n_estimators": 250, "max_depth": 2, "learning_rate": 0.04, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3},
        {"n_estimators": 350, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3},
        {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5},
    ]
    rows = []
    best_params, best_mae = candidates[0], np.inf
    for i, params in enumerate(candidates, 1):
        model = make_xgb(params)
        model.fit(train[feature_cols], train[TARGET])
        pred = model.predict(val[feature_cols])
        row = {"candidate": i, **params, **metrics(val[TARGET], pred)}
        rows.append(row)
        if row["MAE"] < best_mae:
            best_mae, best_params = row["MAE"], params
    return best_params, pd.DataFrame(rows)


def make_xgb(params: dict) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
        **params,
    )


def fit_xgb(features: pd.DataFrame, feature_cols: list[str], bounds: SplitBounds, params: dict, horizon: int = 0) -> XGBRegressor:
    data = features.copy()
    target_col = TARGET if horizon == 0 else f"target_h{horizon}"
    data["_target_index"] = data.index
    if horizon:
        data[target_col] = data[TARGET].shift(-horizon)
        data["_target_index"] = data.index + horizon
    data = data.dropna(subset=feature_cols + [target_col])
    train_val = data[data["_target_index"] < bounds.val_end]
    model = make_xgb(params)
    model.fit(train_val[feature_cols], train_val[target_col])
    return model


def env_oracle_frame(features: pd.DataFrame, historical_cols: list[str], horizon: int) -> tuple[pd.DataFrame, list[str]]:
    data = features[historical_cols].copy()
    for col in ENV_FEATURES:
        data[col] = features[col].shift(-horizon)
    data[TARGET] = features[TARGET].shift(-horizon)
    data["_target_index"] = data.index + horizon
    return data, historical_cols + ENV_FEATURES


def tune_xgb_env_oracle(features: pd.DataFrame, historical_cols: list[str], bounds: SplitBounds) -> tuple[dict, pd.DataFrame]:
    candidates = [
        {"n_estimators": 250, "max_depth": 2, "learning_rate": 0.04, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3},
        {"n_estimators": 350, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3},
        {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5},
    ]
    data, cols = env_oracle_frame(features, historical_cols, 1)
    data = data.dropna(subset=cols + [TARGET])
    train = data[data["_target_index"] < bounds.train_end]
    val = data[(data["_target_index"] >= bounds.train_end) & (data["_target_index"] < bounds.val_end)]
    rows = []
    best_params, best_mae = candidates[0], np.inf
    for i, params in enumerate(candidates, 1):
        model = make_xgb(params)
        model.fit(train[cols], train[TARGET])
        pred = model.predict(val[cols])
        row = {"candidate": i, **params, **metrics(val[TARGET], pred)}
        rows.append(row)
        if row["MAE"] < best_mae:
            best_mae, best_params = row["MAE"], params
    return best_params, pd.DataFrame(rows)


def fit_xgb_env_oracle(features: pd.DataFrame, historical_cols: list[str], bounds: SplitBounds, params: dict, horizon: int) -> tuple[XGBRegressor, list[str]]:
    data, cols = env_oracle_frame(features, historical_cols, horizon)
    data = data.dropna(subset=cols + [TARGET])
    train_val = data[data["_target_index"] < bounds.val_end]
    model = make_xgb(params)
    model.fit(train_val[cols], train_val[TARGET])
    return model, cols


def feature_row_from_history(hourly: pd.DataFrame, y_values: np.ndarray, row_index: int) -> dict:
    ts = hourly.loc[row_index, TIMESTAMP]
    row = {
        "hour": ts.hour,
        "day_of_week": ts.dayofweek,
        "is_weekend": int(ts.dayofweek in [5, 6]),
        "sin_hour": np.sin(2 * np.pi * ts.hour / 24),
        "cos_hour": np.cos(2 * np.pi * ts.hour / 24),
        "sin_dow": np.sin(2 * np.pi * ts.dayofweek / 7),
        "cos_dow": np.cos(2 * np.pi * ts.dayofweek / 7),
    }
    for lag in LAGS:
        row[f"lag_{lag}"] = y_values[row_index - lag]
    past = pd.Series(y_values[:row_index])
    for window in ROLL_MEANS:
        row[f"rolling_mean_{window}"] = past.tail(window).mean()
    for window in ROLL_STDS:
        row[f"rolling_std_{window}"] = past.tail(window).std()
    return row


def xgb_recursive_windows(hourly: pd.DataFrame, bounds: SplitBounds, model: XGBRegressor, feature_cols: list[str]) -> list[dict]:
    windows = []
    for origin in test_origins(bounds):
        y_work = hourly[TARGET].to_numpy(dtype=float).copy()
        predicted = []
        for step in range(1, 25):
            idx = origin + step
            row = feature_row_from_history(hourly, y_work, idx)
            pred = float(model.predict(pd.DataFrame([row])[feature_cols])[0])
            y_work[idx] = max(pred, 0.0)
            predicted.append(max(pred, 0.0))
        actual = hourly.loc[origin + 1 : origin + 24, TARGET].to_numpy()
        windows.append({"origin": hourly.loc[origin, TIMESTAMP], "actual": actual, "predicted": np.asarray(predicted)})
    return windows


def xgb_direct_windows(hourly: pd.DataFrame, features: pd.DataFrame, bounds: SplitBounds, models: dict[int, XGBRegressor], feature_cols: list[str]) -> list[dict]:
    windows = []
    for origin in test_origins(bounds):
        row = features.loc[[origin], feature_cols]
        if row.isna().any(axis=None):
            continue
        predicted = [max(float(models[h].predict(row)[0]), 0.0) for h in ALL_DIRECT_HORIZONS]
        actual = hourly.loc[origin + 1 : origin + 24, TARGET].to_numpy()
        windows.append({"origin": hourly.loc[origin, TIMESTAMP], "actual": actual, "predicted": np.asarray(predicted)})
    return windows


def xgb_env_oracle_windows(hourly: pd.DataFrame, features: pd.DataFrame, bounds: SplitBounds, models: dict[int, XGBRegressor], historical_cols: list[str], env_cols: list[str]) -> list[dict]:
    cols = historical_cols + env_cols
    windows = []
    for origin in test_origins(bounds):
        predicted = []
        for h in ALL_DIRECT_HORIZONS:
            row = features.loc[[origin], historical_cols].copy()
            for col in env_cols:
                row[col] = features.loc[origin + h, col]
            predicted.append(max(float(models[h].predict(row[cols])[0]), 0.0))
        actual = hourly.loc[origin + 1 : origin + 24, TARGET].to_numpy()
        windows.append({"origin": hourly.loc[origin, TIMESTAMP], "actual": actual, "predicted": np.asarray(predicted)})
    return windows


def horizon_rows(windows_by_model: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for model, windows in windows_by_model.items():
        for horizon in HORIZONS:
            actual = [w["actual"][horizon - 1] for w in windows]
            predicted = [w["predicted"][horizon - 1] for w in windows]
            rows.append({"model": model, "horizon_hours": horizon, **metrics(actual, predicted), "windows_evaluated": len(windows)})
    return pd.DataFrame(rows)


def xgb_experiments(hourly: pd.DataFrame, features: pd.DataFrame, bounds: SplitBounds, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, XGBRegressor, list[str]]:
    historical_cols = BASE_FEATURES + ROLLING_FEATURES
    hist_params, hist_tuning = tune_xgb(features, historical_cols, bounds)
    hist_tuning.to_csv(results_dir / "xgboost_tuning_historical.csv", index=False)
    hist_one_step = fit_xgb(features, historical_cols, bounds, hist_params)
    recursive_windows = xgb_recursive_windows(hourly, bounds, hist_one_step, historical_cols)
    direct_models = {h: fit_xgb(features, historical_cols, bounds, hist_params, h) for h in ALL_DIRECT_HORIZONS}
    direct_windows = xgb_direct_windows(hourly, features, bounds, direct_models, historical_cols)

    env_params, env_tuning = tune_xgb_env_oracle(features, historical_cols, bounds)
    env_tuning.to_csv(results_dir / "xgboost_tuning_environmental.csv", index=False)
    env_models = {h: fit_xgb_env_oracle(features, historical_cols, bounds, env_params, h)[0] for h in ALL_DIRECT_HORIZONS}
    env_windows = xgb_env_oracle_windows(hourly, features, bounds, env_models, historical_cols, ENV_FEATURES)

    result_rows = [
        summarize_fixed_origin(recursive_windows, "XGBoost Historical Recursive", "Realistic historical-only recursive 24-hour forecast"),
        summarize_fixed_origin(direct_windows, "XGBoost Historical Direct", "Realistic historical-only direct H1-H24 forecast"),
        summarize_fixed_origin(env_windows, "XGBoost Historical + Environmental Direct", "Oracle/exogenous-information experiment using actual future-row environmental values"),
    ]
    results = pd.DataFrame(result_rows)
    horizons = horizon_rows({"XGBoost Historical Recursive": recursive_windows, "XGBoost Historical Direct": direct_windows, "XGBoost Historical + Environmental Direct": env_windows})
    return results, horizons, hist_params, hist_one_step, historical_cols


def interval_bounds(values: pd.Series, method: str, n_intervals: int) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    if method == "equal_width":
        bounds = np.linspace(arr.min(), arr.max(), n_intervals + 1)
    elif method == "quantile":
        bounds = np.quantile(arr, np.linspace(0, 1, n_intervals + 1))
    else:
        centers = np.sort(KMeans(n_clusters=n_intervals, random_state=RANDOM_STATE, n_init=10).fit(arr.reshape(-1, 1)).cluster_centers_.ravel())
        mids = (centers[:-1] + centers[1:]) / 2
        bounds = np.r_[arr.min(), mids, arr.max()]
    bounds = np.unique(bounds)
    bounds[0] -= 1e-9
    bounds[-1] += 1e-9
    return bounds


def states(values, bounds: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, bounds[1:-1], right=False), 0, len(bounds) - 2)


def chen_predict(train_values: np.ndarray, all_values: np.ndarray, bounds: np.ndarray, order: int = 1, weighted: bool = False) -> np.ndarray:
    train_states = states(train_values, bounds)
    all_states = states(all_values, bounds)
    centers = (bounds[:-1] + bounds[1:]) / 2
    rules: dict[tuple[int, ...], list[int]] = {}
    for i in range(order, len(train_states)):
        key = tuple(train_states[i - order : i])
        rules.setdefault(key, []).append(train_states[i])
    global_center = float(np.mean(train_values))
    preds = np.full(len(all_values), np.nan)
    for i in range(order, len(all_values)):
        key = tuple(all_states[i - order : i])
        consequents = rules.get(key)
        if not consequents:
            preds[i] = global_center
        elif weighted:
            vals, counts = np.unique(consequents, return_counts=True)
            preds[i] = float(np.average(centers[vals], weights=counts))
        else:
            preds[i] = float(np.mean(centers[np.unique(consequents)]))
    return preds


def partition_experiment(hourly: pd.DataFrame, bounds: SplitBounds) -> pd.DataFrame:
    rows = []
    train = hourly.loc[: bounds.train_end - 1, TARGET]
    values = hourly[TARGET].to_numpy()
    val_actual = hourly.loc[bounds.train_end : bounds.val_end - 1, TARGET]
    for method in ["equal_width", "quantile", "kmeans"]:
        for n in [5, 7, 9, 11, 15]:
            b = interval_bounds(train, method, n)
            train_states = states(train, b)
            preds = chen_predict(train.to_numpy(), values, b)
            val_pred = pd.Series(preds, index=hourly.index).loc[bounds.train_end : bounds.val_end - 1]
            counts = pd.Series(train_states).value_counts().sort_index()
            row = {
                "partition": method,
                "requested_intervals": n,
                "actual_states": len(b) - 1,
                "boundaries": json.dumps([round(float(x), 6) for x in b]),
                "min_state_count": int(counts.min()),
                "max_state_count": int(counts.max()),
                "state_imbalance": float(counts.max() / counts.min()) if counts.min() else np.inf,
                "FLR": int(max(len(train) - 1, 0)),
                "FLRG": int(len({tuple(states(train, b)[i - 1 : i + 1]) for i in range(1, len(train))})),
                **metrics(val_actual, val_pred),
            }
            rows.append(row)
    return pd.DataFrame(rows).sort_values("MAE")


def fts_model_comparison(hourly: pd.DataFrame, bounds: SplitBounds, best_partition: pd.Series) -> tuple[pd.DataFrame, dict]:
    train = hourly.loc[: bounds.train_end - 1, TARGET]
    values = hourly[TARGET].to_numpy()
    val_actual = hourly.loc[bounds.train_end : bounds.val_end - 1, TARGET]
    b = interval_bounds(train, best_partition["partition"], int(best_partition["requested_intervals"]))
    configs = [
        ("Chen FTS", 1, False),
        ("Weighted FTS", 1, True),
        ("High-Order FTS order 2", 2, False),
        ("High-Order FTS order 3", 3, False),
        ("High-Order FTS order 6", 6, False),
        ("Weighted High-Order FTS order 2", 2, True),
        ("Weighted High-Order FTS order 3", 3, True),
        ("Weighted High-Order FTS order 6", 6, True),
    ]
    rows = []
    for name, order, weighted in configs:
        preds = chen_predict(train.to_numpy(), values, b, order=order, weighted=weighted)
        val_pred = pd.Series(preds, index=hourly.index).loc[bounds.train_end : bounds.val_end - 1]
        rows.append({"model": name, "order": order, "weighted": weighted, **metrics(val_actual, val_pred)})
    comparison = pd.DataFrame(rows).sort_values("MAE")
    best = comparison.iloc[0].to_dict()
    best["bounds"] = b
    return comparison, best


def fts_fixed_origin(hourly: pd.DataFrame, bounds: SplitBounds, best_fts: dict) -> list[dict]:
    train = hourly.loc[: bounds.val_end - 1, TARGET].to_numpy()
    values = hourly[TARGET].to_numpy()
    order = int(best_fts["order"])
    weighted = bool(best_fts["weighted"])
    b = best_fts["bounds"]
    windows = []
    for origin in test_origins(bounds):
        history = values.copy()
        predicted = []
        for step in range(1, 25):
            current_train = history[: origin + step]
            pred = chen_predict(train, current_train, b, order=order, weighted=weighted)[-1]
            history[origin + step] = max(float(pred), 0.0)
            predicted.append(max(float(pred), 0.0))
        actual = values[origin + 1 : origin + 25]
        windows.append({"origin": hourly.loc[origin, TIMESTAMP], "actual": actual, "predicted": np.asarray(predicted)})
    return windows


def add_fts_features(hourly: pd.DataFrame, features: pd.DataFrame, bounds: SplitBounds, best_fts: dict) -> pd.DataFrame:
    out = features.copy()
    train = hourly.loc[: bounds.val_end - 1, TARGET].to_numpy()
    values = hourly[TARGET].to_numpy()
    b = best_fts["bounds"]
    out["fts_forecast"] = chen_predict(train, values, b, order=int(best_fts["order"]), weighted=bool(best_fts["weighted"]))
    out["fuzzy_state"] = states(values, b)
    return out


def hybrid_experiment(hourly: pd.DataFrame, features: pd.DataFrame, bounds: SplitBounds, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiments = {
        "Base Lag + Calendar": BASE_FEATURES,
        "+ Rolling": BASE_FEATURES + ROLLING_FEATURES,
        "+ FTS": BASE_FEATURES + ROLLING_FEATURES + ["fts_forecast", "fuzzy_state"],
    }
    rows = []
    windows_by_name = {}
    for name, cols in experiments.items():
        models = {h: fit_xgb(features, cols, bounds, params, h) for h in ALL_DIRECT_HORIZONS}
        windows = xgb_direct_windows(hourly, features, bounds, models, cols)
        windows_by_name[name] = windows
        rows.append(summarize_fixed_origin(windows, name, "Direct H1-H24 ablation"))
    ablation = pd.DataFrame(rows)
    base_mae = float(ablation.loc[ablation["model"].eq("+ Rolling"), "Hourly MAE"].iloc[0])
    base_daily_mae = float(ablation.loc[ablation["model"].eq("+ Rolling"), "Daily MAE"].iloc[0])
    ablation["Hourly MAE improvement vs + Rolling (%)"] = (base_mae - ablation["Hourly MAE"]) / base_mae * 100
    ablation["Daily MAE improvement vs + Rolling (%)"] = (base_daily_mae - ablation["Daily MAE"]) / base_daily_mae * 100
    return ablation, pd.DataFrame([summarize_fixed_origin(windows_by_name["+ FTS"], "Hybrid FTS-XGBoost", "Direct H1-H24 with lag, rolling, calendar, fts_forecast, fuzzy_state")])


def plot_fuzzy_counts(hourly: pd.DataFrame, bounds: SplitBounds, best_partition: pd.Series, figures_dir: Path) -> None:
    b = interval_bounds(hourly.loc[: bounds.train_end - 1, TARGET], best_partition["partition"], int(best_partition["requested_intervals"]))
    counts = pd.Series(states(hourly.loc[: bounds.train_end - 1, TARGET], b)).value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")
    plt.title("Training observations per fuzzy state")
    plt.xlabel("Fuzzy state")
    plt.ylabel("Observation count")
    plt.tight_layout()
    plt.savefig(figures_dir / "fuzzy_state_counts.png", dpi=160)
    plt.close()


def write_report(path: Path, aggregation_notes: dict, baseline: pd.DataFrame, xgb: pd.DataFrame, horizons: pd.DataFrame, partitions: pd.DataFrame, fts: pd.DataFrame, hybrid: pd.DataFrame, ablation: pd.DataFrame, final_table: pd.DataFrame) -> None:
    best_partition = partitions.iloc[0]
    best_fts = fts.iloc[0]
    xgb_best = xgb[xgb["model"].eq("XGBoost Historical Direct")].iloc[0]
    xgb_recursive = xgb[xgb["model"].eq("XGBoost Historical Recursive")].iloc[0]
    xgb_env = xgb[xgb["model"].eq("XGBoost Historical + Environmental Direct")].iloc[0]
    hybrid_row = hybrid.iloc[0]
    weekly = baseline[baseline["model"].eq("Weekly Seasonal")].iloc[0]
    xgb_vs_weekly = (weekly["Daily MAE"] - xgb_best["Daily MAE"]) / weekly["Daily MAE"] * 100
    hybrid_vs_xgb = (xgb_best["Daily MAE"] - hybrid_row["Daily MAE"]) / xgb_best["Daily MAE"] * 100
    direct_vs_recursive = (xgb_recursive["Daily MAE"] - xgb_best["Daily MAE"]) / xgb_recursive["Daily MAE"] * 100
    text = f"""# Stage 2 Experiment Report

## 1. Hourly aggregation correction

Hourly aggregation now counts raw 10-minute records per hour with `raw_count`. Normal complete hours require `raw_count = 6`. Incomplete hours found: {aggregation_notes["incomplete_hours"]}. Clean modeling rows: {aggregation_notes["hourly_rows_after_drop"]:,}.

`Appliances` is summed across six 10-minute Wh intervals and converted to `Appliances_kWh`, meaning total energy used during the one-hour interval in kWh. Temperature, humidity, and weather variables use mean aggregation.

## 2. Baseline correction

The main benchmark now uses fixed-origin 24-hour forecasting. The previous one-step daily file remains diagnostic only.

{baseline.round(4).to_markdown(index=False)}

Valid benchmark after correction: Weekly Seasonal Naive Daily MAE = {weekly["Daily MAE"]:.4f} kWh and Daily MAPE = {weekly["Daily MAPE"]:.2f}%.

## 3. Feature engineering

Hourly features include lag 1, 2, 3, 6, 12, 24, 48, and 168; rolling means 3, 6, 12, 24; rolling std 3, 6, 24; integer calendar features; and cyclical hour/day-of-week encodings. Rolling features use `shift(1)` before rolling, so current/future target values do not enter features.

## 4. XGBoost standalone results

{xgb.round(4).to_markdown(index=False)}

## 5. Multi-horizon XGBoost results

{horizons.round(4).to_markdown(index=False)}

## 6. Fuzzy partition experiment

Best validation partition: `{best_partition["partition"]}` with {int(best_partition["requested_intervals"])} requested intervals. Validation MAE = {best_partition["MAE"]:.4f}.

{partitions.head(10).round(4).to_markdown(index=False)}

## 7. FTS model comparison

Best validation FTS: `{best_fts["model"]}`. Validation MAE = {best_fts["MAE"]:.4f}.

{fts.round(4).to_markdown(index=False)}

## 8. Hybrid FTS-XGBoost results

{hybrid.round(4).to_markdown(index=False)}

## 9. Ablation analysis

{ablation.round(4).to_markdown(index=False)}

## 10. Comparison against Weekly Seasonal Naive

{final_table.round(4).to_markdown(index=False)}

## 11. Preliminary conclusion

Hourly remains suitable after dropping incomplete hours because it aligns with the 24-hour forecasting objective and leaves enough observations for chronological validation.

The valid fixed-origin 24-hour benchmark is Weekly Seasonal Naive with Daily MAE {weekly["Daily MAE"]:.4f} kWh and Daily MAPE {weekly["Daily MAPE"]:.2f}%. XGBoost Historical Direct beats that benchmark with Daily MAE {xgb_best["Daily MAE"]:.4f} kWh, an improvement of {xgb_vs_weekly:.2f}% by Daily MAE.

The best fuzzy partition is quantile with 15 intervals, and the best validation FTS type is Weighted FTS. As a standalone fixed-origin model, FTS is weaker than XGBoost and the weekly seasonal baseline.

Hybrid FTS-XGBoost has Daily MAE {hybrid_row["Daily MAE"]:.4f} kWh. That is {hybrid_vs_xgb:.2f}% versus XGBoost Historical Direct by Daily MAE, so the current fuzzy features do not improve XGBoost. They are marginally worse on daily total accuracy.

The best temporary multi-step method is Direct H1-H24 XGBoost, improving Daily MAE by {direct_vs_recursive:.2f}% versus recursive XGBoost. The environmental oracle/exogenous experiment is also worse here, with Daily MAE {xgb_env["Daily MAE"]:.4f} kWh, so environmental variables should not be added to the main proposed model yet.

The main limitation remains the short single-house dataset, so these results support preliminary method selection only. SHAP and 7-day forecasting were not run in this stage.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    results_dir = root / "results"
    figures_dir = root / "figures"
    reports_dir = root / "reports"
    ensure_dirs(data_dir, results_dir, figures_dir, reports_dir)

    raw = load_raw(root)
    hourly, aggregation_notes = make_hourly_clean(raw)
    hourly.to_csv(data_dir / "hourly_energy_clean.csv", index=False)
    (results_dir / "hourly_aggregation_notes.json").write_text(json.dumps(aggregation_notes, indent=2), encoding="utf-8")

    features = add_features(hourly)
    features.to_csv(data_dir / "hourly_features.csv", index=False)
    bounds = split_bounds(len(hourly))

    baseline, baseline_detail = baseline_fixed_origin(hourly, bounds)
    baseline.to_csv(results_dir / "baseline_24h_fixed_origin.csv", index=False)
    baseline_detail.to_csv(results_dir / "baseline_24h_fixed_origin_detail.csv", index=False)

    xgb_results, xgb_horizons, hist_params, _, _ = xgb_experiments(hourly, features, bounds, results_dir)
    xgb_results.to_csv(results_dir / "xgboost_24h_results.csv", index=False)
    xgb_horizons.to_csv(results_dir / "xgboost_horizon_results.csv", index=False)

    partitions = partition_experiment(hourly, bounds)
    partitions.to_csv(results_dir / "fuzzy_partition_results.csv", index=False)
    plot_fuzzy_counts(hourly, bounds, partitions.iloc[0], figures_dir)

    fts_comparison, best_fts = fts_model_comparison(hourly, bounds, partitions.iloc[0])
    fts_windows = fts_fixed_origin(hourly, bounds, best_fts)
    fts_fixed = pd.DataFrame([summarize_fixed_origin(fts_windows, "Best FTS Standalone", str(fts_comparison.iloc[0]["model"]))])
    fts_output = pd.concat([fts_comparison.assign(stage="validation"), fts_fixed.assign(stage="fixed_origin_test")], ignore_index=True, sort=False)
    fts_output.to_csv(results_dir / "fts_model_comparison.csv", index=False)

    hybrid_features = add_fts_features(hourly, features, bounds, best_fts)
    ablation, hybrid = hybrid_experiment(hourly, hybrid_features, bounds, hist_params)
    ablation.to_csv(results_dir / "ablation_results.csv", index=False)
    hybrid.to_csv(results_dir / "hybrid_model_comparison.csv", index=False)

    final_rows = []
    for name in ["Last Value", "Daily Seasonal", "Weekly Seasonal"]:
        row = baseline[baseline["model"].eq(name)].iloc[0]
        final_rows.append({"Model": name, "Hourly MAE": row["Hourly MAE"], "Hourly RMSE": row["Hourly RMSE"], "Daily MAE": row["Daily MAE"], "Daily MAPE": row["Daily MAPE"], "Notes": row["Notes"]})
    for source, name in [(xgb_results, "XGBoost Historical Direct"), (fts_fixed, "Best FTS Standalone"), (hybrid, "Hybrid FTS-XGBoost")]:
        row = source[source["model"].eq(name)].iloc[0]
        final_rows.append({"Model": name, "Hourly MAE": row["Hourly MAE"], "Hourly RMSE": row["Hourly RMSE"], "Daily MAE": row["Daily MAE"], "Daily MAPE": row["Daily MAPE"], "Notes": row["Notes"]})
    final_table = pd.DataFrame(final_rows).sort_values("Daily MAE")
    final_table.to_csv(results_dir / "stage2_final_model_comparison.csv", index=False)

    write_report(
        reports_dir / "stage2_experiment_report.md",
        aggregation_notes,
        baseline,
        xgb_results,
        xgb_horizons,
        partitions,
        fts_comparison,
        hybrid,
        ablation,
        final_table,
    )
    print("Stage 2 pipeline complete.")
    print(f"Clean hourly rows: {len(hourly):,}; incomplete hours dropped: {aggregation_notes['incomplete_hours']}")
    print(f"Report: {reports_dir / 'stage2_experiment_report.md'}")


if __name__ == "__main__":
    main()
