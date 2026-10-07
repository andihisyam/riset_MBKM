from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss


RANDOM_STATE = 42
TARGET = "Appliances"
TIMESTAMP = "date"
DROP_RANDOM_FEATURES = ["rv1", "rv2"]


@dataclass
class SplitConfig:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_energy_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    df = df.sort_values(TIMESTAMP).reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, config: SplitConfig = SplitConfig()) -> dict[str, pd.DataFrame]:
    if not math.isclose(config.train + config.validation + config.test, 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    n = len(df)
    train_end = int(n * config.train)
    validation_end = int(n * (config.train + config.validation))
    return {
        "train": df.iloc[:train_end].copy(),
        "validation": df.iloc[train_end:validation_end].copy(),
        "test": df.iloc[validation_end:].copy(),
    }


def smape(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    denom = np.abs(actual_arr) + np.abs(pred_arr)
    mask = denom != 0
    if not mask.any():
        return np.nan
    return float(np.mean(2 * np.abs(pred_arr[mask] - actual_arr[mask]) / denom[mask]) * 100)


def mape(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    mask = actual_arr != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((actual_arr[mask] - pred_arr[mask]) / actual_arr[mask])) * 100)


def mase(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    train_series: pd.Series | np.ndarray,
    seasonal_lag: int,
) -> float:
    train_arr = np.asarray(train_series, dtype=float)
    if len(train_arr) <= seasonal_lag:
        return np.nan
    scale = np.mean(np.abs(train_arr[seasonal_lag:] - train_arr[:-seasonal_lag]))
    if scale == 0:
        return np.nan
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))) / scale)


def regression_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    train_series: pd.Series | None = None,
    seasonal_lag: int | None = None,
) -> dict[str, float]:
    aligned = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if aligned.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan, "MASE": np.nan}
    errors = aligned["actual"] - aligned["predicted"]
    metrics = {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAPE": mape(aligned["actual"], aligned["predicted"]),
        "sMAPE": smape(aligned["actual"], aligned["predicted"]),
    }
    if train_series is not None and seasonal_lag is not None:
        metrics["MASE"] = mase(aligned["actual"], aligned["predicted"], train_series, seasonal_lag)
    else:
        metrics["MASE"] = np.nan
    return metrics


def audit_structure(df: pd.DataFrame) -> dict:
    intervals = df[TIMESTAMP].diff().dropna()
    return {
        "shape": list(df.shape),
        "date_min": df[TIMESTAMP].min().isoformat(),
        "date_max": df[TIMESTAMP].max().isoformat(),
        "most_common_interval": str(intervals.mode().iloc[0]) if not intervals.empty else None,
        "unique_intervals": {str(k): int(v) for k, v in intervals.value_counts().head(10).items()},
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_timestamps": int(df[TIMESTAMP].duplicated().sum()),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
    }


def target_summary(df: pd.DataFrame) -> dict[str, float]:
    y = df[TARGET]
    q = y.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    iqr = q.loc[0.75] - q.loc[0.25]
    spike_threshold = q.loc[0.75] + 1.5 * iqr
    return {
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "median": float(y.median()),
        "std": float(y.std()),
        "q25": float(q.loc[0.25]),
        "q75": float(q.loc[0.75]),
        "q90": float(q.loc[0.90]),
        "q95": float(q.loc[0.95]),
        "q99": float(q.loc[0.99]),
        "skewness": float(y.skew()),
        "iqr_spike_threshold": float(spike_threshold),
        "spike_count_iqr_rule": int((y > spike_threshold).sum()),
        "spike_share_iqr_rule": float((y > spike_threshold).mean()),
    }


def add_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out[TIMESTAMP].dt.hour
    out["day_of_week"] = out[TIMESTAMP].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6])
    out["date_only"] = out[TIMESTAMP].dt.date
    out["week"] = out[TIMESTAMP].dt.isocalendar().week.astype(int)
    return out


def stationarity_tests(series: pd.Series) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_stat, adf_p, _, _, adf_crit, _ = adfuller(series, autolag="AIC")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, _, kpss_crit = kpss(series, regression="c", nlags="auto")
    except ValueError as exc:
        kpss_stat, kpss_p, kpss_crit = np.nan, np.nan, {"error": str(exc)}
    return {
        "adf_statistic": float(adf_stat),
        "adf_p_value": float(adf_p),
        "adf_critical_values": {k: float(v) for k, v in adf_crit.items()},
        "kpss_statistic": float(kpss_stat) if not np.isnan(kpss_stat) else None,
        "kpss_p_value": float(kpss_p) if not np.isnan(kpss_p) else None,
        "kpss_critical_values": {k: float(v) for k, v in kpss_crit.items()} if isinstance(kpss_crit, dict) else kpss_crit,
    }


def predictor_analysis(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=[np.number])
    pearson_target = numeric.corr(method="pearson")[TARGET].sort_values(ascending=False)
    spearman_target = numeric.corr(method="spearman")[TARGET].sort_values(ascending=False)
    temperature_cols = [col for col in numeric.columns if col.startswith("T")]
    humidity_cols = [col for col in numeric.columns if col.startswith("RH")]
    temp_corr = numeric[temperature_cols].corr().abs()
    humidity_corr = numeric[humidity_cols].corr().abs()

    def high_corr_pairs(corr: pd.DataFrame, threshold: float = 0.85) -> list[dict]:
        pairs = []
        for i, col_a in enumerate(corr.columns):
            for col_b in corr.columns[i + 1 :]:
                value = corr.loc[col_a, col_b]
                if value >= threshold:
                    pairs.append({"feature_a": col_a, "feature_b": col_b, "abs_corr": float(value)})
        return sorted(pairs, key=lambda item: item["abs_corr"], reverse=True)

    return {
        "pearson_target_top": {k: float(v) for k, v in pearson_target.head(12).items()},
        "pearson_target_bottom": {k: float(v) for k, v in pearson_target.tail(12).items()},
        "spearman_target_top": {k: float(v) for k, v in spearman_target.head(12).items()},
        "spearman_target_bottom": {k: float(v) for k, v in spearman_target.tail(12).items()},
        "high_temperature_corr_pairs": high_corr_pairs(temp_corr),
        "high_humidity_corr_pairs": high_corr_pairs(humidity_corr),
        "rv1_rv2_equal": bool(np.allclose(df["rv1"], df["rv2"])),
        "rv1_appliances_pearson": float(numeric["rv1"].corr(numeric[TARGET], method="pearson")),
        "rv2_appliances_pearson": float(numeric["rv2"].corr(numeric[TARGET], method="pearson")),
    }


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    indexed = df.set_index(TIMESTAMP)
    agg: dict[str, str] = {}
    for col in indexed.columns:
        if col == TARGET:
            agg[col] = "sum"
        elif col == "lights":
            agg[col] = "sum"
        else:
            agg[col] = "mean"
    hourly = indexed.resample("h").agg(agg).dropna().reset_index()
    hourly["Appliances_kWh"] = hourly[TARGET] / 1000.0
    return hourly


def baseline_forecasts(df: pd.DataFrame, resolution: str, train_series: pd.Series) -> pd.DataFrame:
    if resolution == "hourly":
        daily_lag, weekly_lag = 24, 168
    elif resolution == "10min":
        daily_lag, weekly_lag = 144, 1008
    else:
        raise ValueError(f"Unknown resolution: {resolution}")
    return pd.DataFrame(
        {
            TIMESTAMP: df[TIMESTAMP],
            "actual": df[TARGET],
            "Last Value Naive": df[TARGET].shift(1),
            "Daily Seasonal Naive": df[TARGET].shift(daily_lag),
            "Weekly Seasonal Naive": df[TARGET].shift(weekly_lag),
        }
    )


def evaluate_baselines(
    full_df: pd.DataFrame,
    resolution: str,
    split_config: SplitConfig = SplitConfig(),
) -> pd.DataFrame:
    splits = chronological_split(full_df, split_config)
    train_series = splits["train"][TARGET]
    forecast_df = baseline_forecasts(full_df, resolution, train_series)
    rows = []
    if resolution == "hourly":
        seasonal_lag = 24
    else:
        seasonal_lag = 144
    for split_name, split_df in splits.items():
        split_forecasts = forecast_df.loc[split_df.index]
        for model in ["Last Value Naive", "Daily Seasonal Naive", "Weekly Seasonal Naive"]:
            metrics = regression_metrics(
                split_forecasts["actual"],
                split_forecasts[model],
                train_series=train_series,
                seasonal_lag=seasonal_lag,
            )
            rows.append(
                {
                    "resolution": resolution,
                    "split": split_name,
                    "model": model,
                    **metrics,
                    "n_evaluated": int(split_forecasts[["actual", model]].dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows)


def daily_total_accuracy(hourly: pd.DataFrame, test_index: Iterable[int]) -> pd.DataFrame:
    forecast_df = baseline_forecasts(hourly, "hourly", hourly[TARGET])
    test_forecasts = forecast_df.loc[list(test_index)].copy()
    test_forecasts["day"] = test_forecasts[TIMESTAMP].dt.date
    rows = []
    for model in ["Last Value Naive", "Daily Seasonal Naive", "Weekly Seasonal Naive"]:
        daily = test_forecasts.groupby("day").agg(actual=(TARGET if TARGET in test_forecasts else "actual", "sum"))
        model_daily = test_forecasts.dropna(subset=[model]).groupby("day").agg(actual=("actual", "sum"), predicted=(model, "sum"))
        if model_daily.empty:
            continue
        metrics = regression_metrics(model_daily["actual"] / 1000.0, model_daily["predicted"] / 1000.0)
        rows.append({"model": model, "split": "test", "unit": "kWh/day", **metrics, "days_evaluated": int(len(model_daily))})
    return pd.DataFrame(rows)


def evaluate_hourly_horizons(hourly: pd.DataFrame, split_config: SplitConfig = SplitConfig()) -> pd.DataFrame:
    splits = chronological_split(hourly, split_config)
    train_series = splits["train"][TARGET]
    rows = []
    horizons = [1, 6, 12, 24]
    for horizon in horizons:
        frame = pd.DataFrame(
            {
                TIMESTAMP: hourly[TIMESTAMP],
                "actual": hourly[TARGET],
                "Last Value Naive": hourly[TARGET].shift(horizon),
                "Daily Seasonal Naive": hourly[TARGET].shift(24),
                "Weekly Seasonal Naive": hourly[TARGET].shift(168),
            }
        )
        for split_name, split_df in splits.items():
            split_forecasts = frame.loc[split_df.index]
            for model in ["Last Value Naive", "Daily Seasonal Naive", "Weekly Seasonal Naive"]:
                metrics = regression_metrics(
                    split_forecasts["actual"],
                    split_forecasts[model],
                    train_series=train_series,
                    seasonal_lag=24,
                )
                rows.append(
                    {
                        "resolution": "hourly",
                        "split": split_name,
                        "horizon_hours": horizon,
                        "model": model,
                        **metrics,
                        "n_evaluated": int(split_forecasts[["actual", model]].dropna().shape[0]),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_rolling_24h_total(hourly: pd.DataFrame, split_config: SplitConfig = SplitConfig()) -> pd.DataFrame:
    splits = chronological_split(hourly, split_config)
    test_start = splits["test"].index.min()
    test_end = splits["test"].index.max()
    origins = range(max(168, test_start - 1), test_end - 24, 24)
    rows = []
    for origin in origins:
        actual_window = hourly.loc[origin + 1 : origin + 24, TARGET]
        if len(actual_window) != 24:
            continue
        forecasts = {
            "Last Value Naive": np.repeat(hourly.loc[origin, TARGET], 24),
            "Daily Seasonal Naive": hourly.loc[origin - 23 : origin, TARGET].to_numpy(),
            "Weekly Seasonal Naive": hourly.loc[origin - 167 : origin - 144, TARGET].to_numpy(),
        }
        actual_kwh = actual_window.sum() / 1000.0
        for model, pred_values in forecasts.items():
            if len(pred_values) != 24:
                continue
            rows.append(
                {
                    "origin": hourly.loc[origin, TIMESTAMP],
                    "target_start": hourly.loc[origin + 1, TIMESTAMP],
                    "target_end": hourly.loc[origin + 24, TIMESTAMP],
                    "model": model,
                    "actual_kWh_24h": actual_kwh,
                    "predicted_kWh_24h": float(np.sum(pred_values) / 1000.0),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame()
    summary_rows = []
    for model, group in detail.groupby("model"):
        metrics = regression_metrics(group["actual_kWh_24h"], group["predicted_kWh_24h"])
        summary_rows.append({"model": model, "split": "test", "unit": "kWh/24h", **metrics, "windows_evaluated": len(group)})
    return pd.DataFrame(summary_rows)


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_figures(df: pd.DataFrame, hourly: pd.DataFrame, figures_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(13, 4))
    plt.plot(df[TIMESTAMP], df[TARGET], linewidth=0.7)
    plt.title("Appliances energy consumption over time")
    plt.xlabel("Timestamp")
    plt.ylabel("Energy per 10-minute interval (Wh)")
    plt.tight_layout()
    plt.savefig(figures_dir / "target_time_series.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[TARGET], bins=60, ax=axes[0])
    axes[0].set_title("Distribution of Appliances")
    axes[0].set_xlabel("Energy per 10-minute interval (Wh)")
    sns.boxplot(x=df[TARGET], ax=axes[1])
    axes[1].set_title("Boxplot of Appliances")
    axes[1].set_xlabel("Energy per 10-minute interval (Wh)")
    plt.tight_layout()
    plt.savefig(figures_dir / "target_distribution.png", dpi=160)
    plt.close()

    calendar = add_calendar_columns(df)
    hourly_pattern = calendar.groupby("hour")[TARGET].mean()
    plt.figure(figsize=(9, 4))
    hourly_pattern.plot(marker="o")
    plt.title("Average 10-minute energy consumption by hour")
    plt.xlabel("Hour of day")
    plt.ylabel("Average energy (Wh)")
    plt.tight_layout()
    plt.savefig(figures_dir / "hourly_consumption_pattern.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=calendar, x="is_weekend", y=TARGET)
    plt.title("Weekday vs weekend Appliances distribution")
    plt.xlabel("Is weekend")
    plt.ylabel("Energy per 10-minute interval (Wh)")
    plt.tight_layout()
    plt.savefig(figures_dir / "weekday_weekend_comparison.png", dpi=160)
    plt.close()

    fig = plt.figure(figsize=(12, 4))
    plot_acf(df[TARGET], lags=1008, ax=plt.gca())
    plt.title("ACF of Appliances up to 1008 lags")
    plt.xlabel("Lag (10-minute intervals)")
    plt.ylabel("Autocorrelation")
    plt.tight_layout()
    plt.savefig(figures_dir / "acf_10min.png", dpi=160)
    plt.close(fig)

    numeric = df.select_dtypes(include=[np.number]).drop(columns=DROP_RANDOM_FEATURES, errors="ignore")
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, square=False)
    plt.title("Pearson correlation heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(13, 4))
    plt.plot(hourly[TIMESTAMP], hourly["Appliances_kWh"], linewidth=0.8)
    plt.title("Hourly total Appliances energy")
    plt.xlabel("Timestamp")
    plt.ylabel("Hourly energy (kWh)")
    plt.tight_layout()
    plt.savefig(figures_dir / "hourly_energy_time_series.png", dpi=160)
    plt.close()


def acf_key_lags(series: pd.Series, lags: list[int]) -> dict[str, float]:
    values = {}
    for lag in lags:
        values[f"lag_{lag}"] = float(series.autocorr(lag=lag)) if len(series) > lag else np.nan
    return values


def build_report(
    output_path: Path,
    audit: dict,
    target: dict,
    temporal: dict,
    stationarity: dict,
    predictors: dict,
    hourly_summary: dict,
    baseline_results: pd.DataFrame,
    daily_results: pd.DataFrame,
    horizon_results: pd.DataFrame,
    daily_24h_results: pd.DataFrame,
) -> None:
    best_test = (
        baseline_results[baseline_results["split"] == "test"]
        .sort_values(["resolution", "MAE"])
        .groupby("resolution")
        .head(1)
    )
    report = f"""# Preliminary Experiment Report

## 1. Dataset Characteristics

Dataset berisi {audit["shape"][0]:,} observasi dan {audit["shape"][1]} kolom. Periode data dimulai dari {audit["date_min"]} sampai {audit["date_max"]}. Interval paling umum adalah {audit["most_common_interval"]}. Duplicate row = {audit["duplicate_rows"]:,}, duplicate timestamp = {audit["duplicate_timestamps"]:,}.

Target `Appliances` memiliki median {target["median"]:.2f} Wh per 10 menit, mean {target["mean"]:.2f} Wh, maksimum {target["max"]:.2f} Wh, dan skewness {target["skewness"]:.2f}. Ini mengonfirmasi distribusi right-skewed. Batas IQR memberi {target["spike_count_iqr_rule"]:,} kandidat spike ({target["spike_share_iqr_rule"]:.2%}), tetapi spike tidak dihapus karena secara domain dapat merepresentasikan penggunaan appliance nyata.

`rv1` dan `rv2` identik: {predictors["rv1_rv2_equal"]}. Korelasi Pearson `rv1` terhadap target adalah {predictors["rv1_appliances_pearson"]:.4f}, dan `rv2` adalah {predictors["rv2_appliances_pearson"]:.4f}. Keduanya dikeluarkan dari proposed model.

## 2. Forecasting Problem Definition

Target praktis penelitian adalah memprediksi total konsumsi energi 24 jam berikutnya. Pada resolusi hourly, target model adalah total energi per jam dalam Wh, dengan output ringkas harian dihitung sebagai jumlah 24 prediksi hourly dan dikonversi ke kWh.

## 3. Why Hourly / Why 10-minute

Agregasi hourly menjumlahkan enam interval `Appliances` 10 menit menjadi energi per jam. Fitur temperatur, humidity, tekanan, angin, visibility, dew point, `rv1`, dan `rv2` diagregasi dengan mean. Jumlah observasi hourly menjadi {hourly_summary["shape"][0]:,}, dibanding {audit["shape"][0]:,} observasi 10 menit.

Keputusan sementara: hourly lebih selaras dengan target 24-hour dan 7-day forecasting karena horizon 24 jam hanya memerlukan 24 langkah, sedangkan data 10 menit memerlukan 144 langkah. Namun keputusan final tetap harus dibandingkan dengan model ML/FTS, bukan hanya baseline.

## 4. Temporal Analysis

Autocorrelation lag penting pada data 10 menit:

{pd.Series(temporal["acf_10min"]).to_markdown()}

Interpretasi awal: lag pendek masih kuat, dan lag 144 menunjukkan pola harian. Lag 1008 memberi indikasi pola mingguan, tetapi perlu hati-hati karena dataset hanya sekitar 4,5 bulan.

## 5. Stationarity

ADF p-value = {stationarity["adf_p_value"]:.6f}. KPSS p-value = {stationarity["kpss_p_value"] if stationarity["kpss_p_value"] is not None else "NA"}. Hasil ini tidak otomatis berarti differencing harus diterapkan. Untuk XGBoost dan FTS, fitur lag/seasonal serta validasi kronologis lebih relevan daripada memaksa differencing di awal.

## 6. Predictor Notes

Beberapa pasangan temperatur/humidity sangat berkorelasi. Ini perlu diperhatikan saat interpretasi SHAP karena fitur lingkungan yang saling berkorelasi dapat berbagi kontribusi model.

Top Pearson correlation terhadap target:

{pd.Series(predictors["pearson_target_top"]).to_markdown()}

## 7. Baseline Results

Baseline dihitung dengan split kronologis 70% train, 15% validation, dan 15% test. Tidak ada random split.

{baseline_results.round(4).to_markdown(index=False)}

Model baseline terbaik pada test untuk tiap resolusi:

{best_test.round(4).to_markdown(index=False)}

## 8. Daily Total Energy Accuracy

Pada resolusi hourly, hourly prediction one-step rolling dijumlahkan menjadi total harian kWh untuk test set. Ini berguna sebagai diagnosis awal, tetapi bukan klaim final untuk prediksi 24 jam penuh karena model `Last Value Naive` masih memakai aktual jam sebelumnya di dalam periode target.

{daily_results.round(4).to_markdown(index=False)}

Evaluasi yang lebih sesuai dengan target praktis adalah rolling 24-hour total forecast. Dari setiap forecast origin pada test set, baseline memprediksi 24 jam berikutnya lalu hasilnya dijumlahkan.

{daily_24h_results.round(4).to_markdown(index=False)}

## 9. Forecast Horizon Check

Baseline hourly juga dievaluasi untuk horizon 1, 6, 12, dan 24 jam.

{horizon_results[horizon_results["split"].eq("test")].round(4).to_markdown(index=False)}

## 10. Preliminary Research Conclusion

Hasil tahap awal mendukung penggunaan hourly sebagai kandidat resolusi utama karena lebih sesuai dengan target 24 jam dan tetap mempertahankan jumlah observasi yang cukup. Baseline seasonal wajib dipertahankan sebagai pembanding utama sebelum mengklaim XGBoost, FTS, atau hybrid lebih baik.

Belum ada dasar untuk menyimpulkan bahwa FTS-XGBoost adalah model terbaik. Tahap berikutnya harus menjalankan XGBoost standalone, eksperimen partition FTS, FTS standalone, lalu hybrid secara kronologis dengan validation set.

## 11. Next Experiments

1. Feature engineering hourly: lag, rolling past-only, calendar, dan cyclical encoding.
2. XGBoost historical-only sebagai model realistis.
3. Eksperimen fuzzy partition equal-width vs quantile untuk 5, 7, 9, 11, dan 15 interval.
4. Chen FTS, Weighted FTS, dan jika stabil High-Order FTS.
5. Hybrid FTS-XGBoost dan ablation study.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "energydata_complete.csv"
    results_dir = root / "results"
    figures_dir = root / "figures"
    reports_dir = root / "reports"
    ensure_dirs(results_dir, figures_dir, reports_dir)

    df = load_energy_data(data_path)
    audit = audit_structure(df)
    target = target_summary(df)
    stationarity = stationarity_tests(df[TARGET])
    predictors = predictor_analysis(df)
    hourly = aggregate_hourly(df)

    temporal = {
        "acf_10min": acf_key_lags(df[TARGET], [1, 6, 12, 144, 288, 1008]),
        "mean_by_hour": add_calendar_columns(df).groupby("hour")[TARGET].mean().to_dict(),
        "mean_by_weekend": add_calendar_columns(df).groupby("is_weekend")[TARGET].mean().to_dict(),
    }
    hourly_summary = {
        "shape": list(hourly.shape),
        "date_min": hourly[TIMESTAMP].min().isoformat(),
        "date_max": hourly[TIMESTAMP].max().isoformat(),
        "target_kwh_mean": float(hourly["Appliances_kWh"].mean()),
        "target_kwh_median": float(hourly["Appliances_kWh"].median()),
    }

    baseline_10min = evaluate_baselines(df, "10min")
    baseline_hourly = evaluate_baselines(hourly, "hourly")
    baseline_results = pd.concat([baseline_10min, baseline_hourly], ignore_index=True)

    hourly_splits = chronological_split(hourly)
    daily_results = daily_total_accuracy(hourly, hourly_splits["test"].index)
    horizon_results = evaluate_hourly_horizons(hourly)
    daily_24h_results = evaluate_rolling_24h_total(hourly)

    save_figures(df, hourly, figures_dir)
    save_json(audit, results_dir / "eda_structure_audit.json")
    save_json(target, results_dir / "target_summary.json")
    save_json(temporal, results_dir / "temporal_summary.json")
    save_json(stationarity, results_dir / "stationarity_tests.json")
    save_json(predictors, results_dir / "predictor_analysis.json")
    save_json(hourly_summary, results_dir / "hourly_summary.json")
    hourly.to_csv(results_dir / "hourly_energy.csv", index=False)
    baseline_results.to_csv(results_dir / "baseline_results.csv", index=False)
    daily_results.to_csv(results_dir / "daily_baseline_results.csv", index=False)
    horizon_results.to_csv(results_dir / "horizon_baseline_results.csv", index=False)
    daily_24h_results.to_csv(results_dir / "daily_24h_baseline_results.csv", index=False)

    build_report(
        reports_dir / "preliminary_experiment_report.md",
        audit,
        target,
        temporal,
        stationarity,
        predictors,
        hourly_summary,
        baseline_results,
        daily_results,
        horizon_results,
        daily_24h_results,
    )

    print("Preliminary pipeline complete.")
    print(f"Rows: 10-minute={len(df):,}, hourly={len(hourly):,}")
    print(f"Saved report: {reports_dir / 'preliminary_experiment_report.md'}")


if __name__ == "__main__":
    main()
