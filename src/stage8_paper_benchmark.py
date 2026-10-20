"""Paper-aligned 10-minute benchmark for Hybrid Weighted FTS-Random Forest.

The script is deliberately isolated from the hourly Stage 1-7 pipelines. It
uses the original 10-minute target and writes only to results/stage8_paper_benchmark.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "stage8_paper_benchmark"
FIG = OUT / "figures"
PRED = OUT / "predictions"
CFG = OUT / "configs"
RANDOM_STATE = 42
TARGET = "Appliances"
ENV = ["lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5", "RH_5", "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
TIME = ["NSM", "hour", "minute", "day_of_week", "is_weekend"]
HIST = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "rolling_mean_3", "rolling_mean_6", "rolling_std_3", "delta_1"]
PAPER_REFERENCE = [
    {"model": "Linear Regression", "MAE_Wh": 51.97, "RMSE_Wh": 93.18, "R2": 0.16, "MAPE_pct": 59.93},
    {"model": "SVM Radial", "MAE_Wh": 31.36, "RMSE_Wh": 70.74, "R2": 0.52, "MAPE_pct": 29.76},
    {"model": "Random Forest", "MAE_Wh": 31.85, "RMSE_Wh": 68.48, "R2": 0.54, "MAPE_pct": 31.39},
    {"model": "Gradient Boosting Machine", "MAE_Wh": 35.22, "RMSE_Wh": 66.65, "R2": 0.57, "MAPE_pct": 38.29},
]


def ensure_dirs() -> None:
    for p in [OUT, FIG, PRED, CFG]:
        p.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "energydata_complete.csv")
    df["date"] = pd.to_datetime(df["date"])
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def metric_row(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    denom = np.abs(a) + np.abs(p)
    nonzero = a != 0
    smask = denom != 0
    return {
        "MAE_Wh": float(mean_absolute_error(a, p)),
        "RMSE_Wh": float(np.sqrt(mean_squared_error(a, p))),
        "R2": float(r2_score(a, p)) if len(a) > 1 else np.nan,
        "MAPE_pct": float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100) if nonzero.any() else np.nan,
        "sMAPE_pct": float(np.mean(2 * np.abs(a[smask] - p[smask]) / denom[smask]) * 100) if smask.any() else np.nan,
        "bias_Wh": float(np.mean(p - a)),
        "n": int(len(a)),
    }


def audit_dataset(df: pd.DataFrame) -> dict:
    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier = (df[TARGET] < q1 - 1.5 * iqr) | (df[TARGET] > q3 + 1.5 * iqr)
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "timestamp_min": df.date.min().isoformat(),
        "timestamp_max": df.date.max().isoformat(),
        "duplicate_timestamps": int(df.date.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "target_min_Wh": float(df[TARGET].min()),
        "target_max_Wh": float(df[TARGET].max()),
        "target_mean_Wh": float(df[TARGET].mean()),
        "target_median_Wh": float(df[TARGET].median()),
        "target_std_Wh": float(df[TARGET].std()),
        "iqr_outliers": int(outlier.sum()),
        "expected_target_unit": "Wh per 10-minute interval",
        "resolution_check": str(df.date.diff().dropna().mode().iloc[0]),
    }


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["NSM"] = out.date.dt.hour * 3600 + out.date.dt.minute * 60 + out.date.dt.second
    out["hour"] = out.date.dt.hour
    out["minute"] = out.date.dt.minute
    out["day_of_week"] = out.date.dt.dayofweek
    out["is_weekend"] = out.day_of_week.isin([5, 6]).astype(int)
    for lag in [1, 2, 3, 6, 12]:
        out[f"lag_{lag}"] = out[TARGET].shift(lag)
    shifted = out[TARGET].shift(1)
    for window in [3, 6]:
        out[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
    out["rolling_std_3"] = shifted.rolling(3).std()
    out["delta_1"] = out[TARGET].shift(1).diff()
    return out


def fts_rules(y_train: np.ndarray, order: int, n_intervals: int = 15) -> tuple[np.ndarray, dict[tuple[int, ...], list[int]], float]:
    bounds = np.quantile(y_train, np.linspace(0, 1, n_intervals + 1))
    bounds = np.unique(bounds)
    bounds[0] -= 1e-9
    bounds[-1] += 1e-9
    states = lambda x: np.clip(np.digitize(np.asarray(x), bounds[1:-1], right=False), 0, len(bounds) - 2)
    train_states = states(y_train)
    rules: dict[tuple[int, ...], list[int]] = {}
    for i in range(order, len(train_states)):
        rules.setdefault(tuple(train_states[i - order : i]), []).append(int(train_states[i]))
    centers = (bounds[:-1] + bounds[1:]) / 2
    return bounds, rules, float(np.mean(y_train))


def fts_predict_one(history: np.ndarray, rules: dict, bounds: np.ndarray, fallback: float, order: int) -> tuple[float, bool]:
    if len(history) < order:
        return fallback, True
    states = np.clip(np.digitize(history, bounds[1:-1], right=False), 0, len(bounds) - 2)
    values = rules.get(tuple(states[-order:]))
    if not values:
        return fallback, True
    vals, counts = np.unique(values, return_counts=True)
    centers = (bounds[:-1] + bounds[1:]) / 2
    return float(np.average(centers[vals], weights=counts)), False


def fts_predictions(df: pd.DataFrame, train_end: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    y = df[TARGET].to_numpy(float)
    bounds, rules, fallback = fts_rules(y[:train_end], order)
    pred = np.full(len(df), np.nan)
    fallback_flag = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        pred[i], flag = fts_predict_one(y[:i], rules, bounds, fallback, order)
        fallback_flag[i] = int(flag)
    return pred, fallback_flag


def build_matrix(df: pd.DataFrame, strict: bool, include_history: bool, include_environment: bool = True) -> tuple[pd.DataFrame, list[str]]:
    out = pd.DataFrame(index=df.index)
    source = df.index - 1 if strict else df.index
    if include_environment:
        for col in ENV:
            out[col] = df[col].reindex(source).to_numpy()
    for col in TIME:
        out[col] = df[col].to_numpy()
    if include_history:
        for col in HIST:
            out[col] = df[col].to_numpy()
    return out, list(out.columns)


def candidate_params() -> list[dict]:
    return [
        {"n_estimators": 20, "max_depth": 12, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 1.0, "bootstrap": True},
        {"n_estimators": 30, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "bootstrap": True},
    ]


def fit_rf(x, y, params):
    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1, **params)
    model.fit(x, y)
    return model


def tune_params(x: pd.DataFrame, y: pd.Series, train_end: int, mode: str, fts: np.ndarray, target_start: int, target_stop: int) -> tuple[dict, list[dict]]:
    rows = []
    best, best_score = None, np.inf
    candidates = candidate_params()
    if mode == "cv":
        splitter = TimeSeriesSplit(n_splits=3)
        splits = list(splitter.split(x.iloc[:train_end]))
    else:
        splits = [(np.arange(train_end), np.arange(train_end, target_stop))]
    for i, params in enumerate(candidates, 1):
        scores = []
        for fold, (tr, va) in enumerate(splits, 1):
            model = fit_rf(x.iloc[tr], y.iloc[tr], params)
            pred = model.predict(x.iloc[va])
            scores.append(metric_row(y.iloc[va].to_numpy(), pred)["MAE_Wh"])
            rows.append({"candidate": i, "fold": fold, "selection_mode": mode, **params, "MAE_Wh": scores[-1]})
        score = float(np.mean(scores))
        if score < best_score:
            best_score, best = score, params
    return best or candidates[0], rows


def evaluate_variant(df, variant_name: str, strict: bool, include_history: bool, split_bounds: dict, selected_orders: list[int], track: str, include_environment: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    x, cols = build_matrix(df, strict, include_history, include_environment)
    valid = x.notna().all(axis=1) & df[TARGET].notna()
    warm = max(13, 1)
    train_end, val_end, n = split_bounds["train_end"], split_bounds["val_end"], len(df)
    rows, pred_rows, tuning_rows = [], [], []
    for order in selected_orders:
        fts_pred, fallback = fts_predictions(df, train_end, order)
        use = valid & (df.index >= warm) & np.isfinite(fts_pred)
        train_idx = np.where(use & (df.index < train_end))[0]
        val_idx = np.where(use & (df.index >= train_end) & (df.index < val_end))[0]
        test_idx = np.where(use & (df.index >= val_end))[0]
        if len(train_idx) < 100 or len(test_idx) == 0:
            continue
        selection_mode = "cv" if track == "A" else "validation"
        if track == "A":
            tune_x = x.loc[train_idx, cols].reset_index(drop=True)
            tune_y = df.loc[train_idx, TARGET].reset_index(drop=True)
            tune_train_end, tune_stop = len(train_idx), len(train_idx)
        else:
            tune_indices = np.r_[train_idx, val_idx]
            tune_x = x.loc[tune_indices, cols].reset_index(drop=True)
            tune_y = df.loc[tune_indices, TARGET].reset_index(drop=True)
            tune_train_end, tune_stop = len(train_idx), len(tune_indices)
        params, log = tune_params(tune_x, tune_y, tune_train_end, selection_mode, fts_pred, train_idx[0], tune_stop)
        for item in log:
            tuning_rows.append({"track": track, "variant": variant_name, "fts_order": order, **item})
        rf_direct = fit_rf(x.loc[train_idx, cols], df.loc[train_idx, TARGET], params)
        rf_resid = fit_rf(x.loc[train_idx, cols], df.loc[train_idx, TARGET].to_numpy() - fts_pred[train_idx], params)
        for split, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
            if len(idx) == 0:
                continue
            actual = df.loc[idx, TARGET].to_numpy(float)
            direct = np.maximum(rf_direct.predict(x.loc[idx, cols]), 0)
            residual = fts_pred[idx] + rf_resid.predict(x.loc[idx, cols])
            residual = np.maximum(residual, 0)
            fts_only = np.maximum(fts_pred[idx], 0)
            for model_name, pred in [("Weighted FTS", fts_only), ("Random Forest", direct), ("Hybrid FTS-RF residual", residual)]:
                rows.append({"track": track, "variant": variant_name, "fts_order": order, "split": split, "model": model_name, "n": len(idx), "fallback_rate": float(fallback[idx].mean()), **metric_row(actual, pred), "params": json.dumps(params, sort_keys=True)})
            for j, row_idx in enumerate(idx):
                pred_rows.append({"track": track, "variant": variant_name, "fts_order": order, "split": split, "index": int(row_idx), "timestamp": df.loc[row_idx, "date"], "actual_Wh": actual[j], "fts_prediction_Wh": fts_only[j], "rf_prediction_Wh": direct[j], "hybrid_prediction_Wh": residual[j], "residual_actual_Wh": actual[j] - fts_only[j], "fallback": int(fallback[row_idx])})
    return pd.DataFrame(rows), pd.DataFrame(pred_rows), tuning_rows


def write_feature_matrix() -> None:
    rows = []
    for col in ENV + ["lights"]:
        rows.append({"feature": col, "group": "environment", "available_before_prediction": "origin value only", "available_at_target": True, "used_in_track_a": True, "used_in_track_b": True})
    for col in ["NSM", "hour", "minute", "day_of_week", "is_weekend"]:
        rows.append({"feature": col, "group": "time", "available_before_prediction": True, "available_at_target": True, "used_in_track_a": True, "used_in_track_b": True})
    for col in HIST:
        rows.append({"feature": col, "group": "historical energy", "available_before_prediction": True, "available_at_target": False, "used_in_track_a": True, "used_in_track_b": True})
    pd.DataFrame(rows).to_csv(OUT / "FEATURE_AVAILABILITY_MATRIX.csv", index=False)
    leakage = []
    for col in ENV:
        leakage.append({"track": "B", "feature": col, "source_index_relative_to_target": -1, "future_actual_value_used": False, "reason": "origin value only"})
    for col in HIST:
        leakage.append({"track": "B", "feature": col, "source_index_relative_to_target": -1, "future_actual_value_used": False, "reason": "lag/rolling uses history through origin"})
    for col in TIME:
        leakage.append({"track": "B", "feature": col, "source_index_relative_to_target": 0, "future_actual_value_used": False, "reason": "calendar timestamp known before target observation"})
    pd.DataFrame(leakage).to_csv(OUT / "FEATURE_LEAKAGE_AUDIT.csv", index=False)


def write_plots(df: pd.DataFrame, all_preds: pd.DataFrame, order_df: pd.DataFrame, best_key: dict) -> None:
    plt.figure(figsize=(8, 4)); plt.hist(df[TARGET], bins=40, color="#4472c4"); plt.title("Distribusi Appliances (Wh per 10 menit)"); plt.xlabel("Wh"); plt.ylabel("Frekuensi"); plt.tight_layout(); plt.savefig(FIG / "01_target_distribution.png", dpi=150); plt.close()
    best = all_preds[(all_preds.track == "B") & (all_preds.variant == best_key["variant"]) & (all_preds.fts_order == best_key["fts_order"]) & (all_preds.split == "test")].head(500)
    if not best.empty:
        plt.figure(figsize=(12, 4)); plt.plot(pd.to_datetime(best.timestamp), best.actual_Wh, label="Aktual"); plt.plot(pd.to_datetime(best.timestamp), best.hybrid_prediction_Wh, label="Hybrid"); plt.legend(); plt.title("Aktual vs Hybrid pada resolusi 10 menit"); plt.tight_layout(); plt.savefig(FIG / "02_actual_vs_hybrid_10min.png", dpi=150); plt.close()
    piv = order_df[(order_df.track == "B") & (order_df.model == "Hybrid FTS-RF residual") & (order_df.split == "test")].pivot_table(index="fts_order", columns="variant", values="MAE_Wh")
    if not piv.empty: piv.plot(kind="bar", figsize=(8, 4)); plt.ylabel("MAE (Wh)"); plt.title("MAE Hybrid menurut orde FTS dan fitur"); plt.tight_layout(); plt.savefig(FIG / "03_fts_order_comparison.png", dpi=150); plt.close()
    if not best.empty:
        for col, title, fname in [("fts_prediction_Wh", "FTS sebelum koreksi RF", "04_fts_before_rf.png"), ("hybrid_prediction_Wh", "Hybrid sesudah koreksi RF", "05_fts_after_rf.png")]:
            plt.figure(figsize=(12, 4)); plt.plot(pd.to_datetime(best.timestamp), best.actual_Wh, label="Aktual"); plt.plot(pd.to_datetime(best.timestamp), best[col], label=title); plt.legend(); plt.title(title); plt.tight_layout(); plt.savefig(FIG / fname, dpi=150); plt.close()
        best["hour"] = pd.to_datetime(best.timestamp).dt.hour
        err = best.assign(abs_error=np.abs(best.actual_Wh - best.hybrid_prediction_Wh)).groupby("hour").abs_error.mean()
        err.plot(kind="bar", figsize=(9, 4), title="MAE Hybrid berdasarkan jam"); plt.ylabel("MAE (Wh)"); plt.tight_layout(); plt.savefig(FIG / "06_error_by_hour.png", dpi=150); plt.close()
        plt.figure(figsize=(5, 5)); plt.scatter(best.actual_Wh, best.hybrid_prediction_Wh, s=8, alpha=.4); plt.xlabel("Aktual (Wh)"); plt.ylabel("Prediksi (Wh)"); plt.title("Aktual vs prediksi"); plt.tight_layout(); plt.savefig(FIG / "07_actual_predicted_scatter.png", dpi=150); plt.close()
        plt.figure(figsize=(8, 4)); plt.hist(best.actual_Wh - best.hybrid_prediction_Wh, bins=40); plt.title("Distribusi residual Hybrid"); plt.xlabel("Aktual - prediksi (Wh)"); plt.tight_layout(); plt.savefig(FIG / "08_residual_distribution.png", dpi=150); plt.close()


def write_reports(df: pd.DataFrame, audit: dict, track_a: pd.DataFrame, track_b: pd.DataFrame, order_df: pd.DataFrame, tuning: pd.DataFrame, best_key: dict) -> None:
    paper = pd.DataFrame(PAPER_REFERENCE); paper.to_csv(OUT / "PAPER_REFERENCE_RESULTS.csv", index=False)
    with (OUT / "DATASET_AUDIT.md").open("w", encoding="utf-8") as f:
        f.write("# Audit Dataset Stage 8\n\n")
        f.write("Dataset 10-menit asli dipakai tanpa agregasi hourly.\n\n```json\n" + json.dumps(audit, indent=2) + "\n```\n")
        f.write("\nTarget `Appliances` diperlakukan sebagai Wh per interval 10 menit. Outlier dicatat dengan aturan IQR dan tidak dihapus.\n")
    (OUT / "PAPER_PROTOCOL_AUDIT.md").write_text("""# Audit Protokol Paper\n\nPaper menggunakan data 10-menit, 75% training dan 25% testing melalui CARET `createDataPartition`, repeated 10-fold CV dengan 3 repeat, serta predictor lingkungan, lights, NSM, status hari, dan hari minggu. Seed/split persis dari paper tidak tersedia dalam PDF yang digunakan, sehingga Track A di sini adalah approximate reproduction.\n\nFTS membutuhkan urutan timestamp. Karena itu Track A menggunakan pembagian kronologis 75/25 agar transisi FTS tidak menyambungkan baris acak. Angka paper hanya dipakai sebagai reference table, bukan sebagai hasil eksperimen ulang. Track B menggunakan split kronologis train/validation/test dan informasi lingkungan pada forecast origin.\n""", encoding="utf-8")
    track_a.to_csv(OUT / "TRACK_A_RESULTS.csv", index=False); track_b.to_csv(OUT / "TRACK_B_RESULTS.csv", index=False)
    order_df.to_csv(OUT / "FTS_ORDER_COMPARISON.csv", index=False); tuning.to_csv(OUT / "HYPERPARAMETER_TUNING.csv", index=False)
    best = track_b[(track_b.model == "Hybrid FTS-RF residual") & (track_b.split == "test") & (track_b.variant == best_key["variant"]) & (track_b.fts_order == best_key["fts_order"])].iloc[0]
    fts = track_b[(track_b.model == "Weighted FTS") & (track_b.split == "test") & (track_b.variant == best_key["variant"]) & (track_b.fts_order == best_key["fts_order"])].iloc[0]
    rf = track_b[(track_b.model == "Random Forest") & (track_b.split == "test") & (track_b.variant == best_key["variant"]) & (track_b.fts_order == best_key["fts_order"])].iloc[0]
    contribution = pd.DataFrame([{"component": "Weighted FTS", "MAE_Wh": fts.MAE_Wh}, {"component": "Random Forest same features", "MAE_Wh": rf.MAE_Wh}, {"component": "Hybrid FTS-RF residual", "MAE_Wh": best.MAE_Wh}, {"component": "RF correction improvement vs FTS (%)", "MAE_Wh": (fts.MAE_Wh - best.MAE_Wh) / fts.MAE_Wh * 100}])
    contribution.to_csv(OUT / "RESIDUAL_CONTRIBUTION.csv", index=False)
    ref = paper.assign(comparison="paper reported testing")
    candidate = track_a[(track_a.model.isin(["Random Forest", "Hybrid FTS-RF residual"])) & (track_a.split == "test")][["variant", "fts_order", "model", "MAE_Wh", "RMSE_Wh", "R2", "MAPE_pct"]].copy(); candidate["comparison"] = "Stage 8 Track A approximate"
    pd.concat([ref.rename(columns={"model": "method"}), candidate.rename(columns={"model": "method"})], ignore_index=True, sort=False).to_csv(OUT / "PAPER_COMPARISON.csv", index=False)
    (OUT / "STAGE8_README.md").write_text("""# Stage 8 Paper-Aligned Benchmark\n\nScript: `src/stage8_paper_benchmark.py`. Track A uses 10-minute data, 75/25 chronological split, and paper-style environmental/time predictors. It is approximate because the original CARET seed/split was not recovered. Track B uses chronological rolling one-step prediction with validation-selected FTS order and Random Forest parameters.\n\nThe paper reference values are stored in `PAPER_REFERENCE_RESULTS.csv`; they are not reproduced results. Main forecasting conclusions must use Track B.\n""", encoding="utf-8")
    (OUT / "PAPER_COMPARISON.md").write_text("# Perbandingan dengan Paper\n\nAngka paper berasal dari Tabel 5 dan memakai protokol yang tidak identik. Track A diberi label approximate; Track B adalah benchmark kronologis utama. Lihat `PAPER_COMPARISON.csv` untuk angka dan `PAPER_PROTOCOL_AUDIT.md` untuk batasan.\n", encoding="utf-8")
    (OUT / "REPRODUCIBILITY.md").write_text("# Reproducibility\n\nJalankan `python run_stage8.py` dari root repository. Seed Random Forest: 42. Quantile interval FTS: 15. Output terisolasi di `results/stage8_paper_benchmark/`; Stage 1-7.1 tidak ditimpa.\n", encoding="utf-8")
    (OUT / "STAGE8_FINAL_REPORT.md").write_text(f"""# Stage 8 Final Report\n\n## Verdict\n\nStage 8 berhasil dijalankan pada data 10-menit asli. Track A merupakan approximate reproduction karena split random CARET dan seed paper tidak tersedia. Track B menjadi evaluasi utama karena menjaga urutan waktu dan memisahkan train, validation, serta test.\n\n## Konfigurasi terpilih Track B\n\n- Variant: `{best_key['variant']}`\n- FTS order: `{best_key['fts_order']}`\n- Seleksi: validation MAE hybrid; test tidak dipakai memilih konfigurasi.\n- Test Hybrid MAE: **{best.MAE_Wh:.4f} Wh**\n- Test FTS-only MAE: **{fts.MAE_Wh:.4f} Wh**\n- Test RF direct MAE: **{rf.MAE_Wh:.4f} Wh**\n- Perbaikan residual terhadap FTS: **{(fts.MAE_Wh - best.MAE_Wh) / fts.MAE_Wh * 100:.2f}%**\n\n## Interpretasi\n\nHasil Stage 8 tidak boleh dipakai untuk menyatakan hybrid mengungguli paper secara mutlak karena unit, split, fitur, dan protokol paper tidak identik. Fitur lingkungan pada Track B hanya memakai nilai forecast origin; nilai lingkungan target tidak digunakan sebagai forecasting utama. Fallback FTS dilaporkan pada CSV prediksi dan ringkasan model.\n""", encoding="utf-8")


def error_analysis(pred: pd.DataFrame, best_key: dict, df: pd.DataFrame) -> None:
    p = pred[(pred.track == "B") & (pred.variant == best_key["variant"]) & (pred.fts_order == best_key["fts_order"]) & (pred.split == "test")].copy()
    p["timestamp"] = pd.to_datetime(p.timestamp); p["hour"] = p.timestamp.dt.hour; p["day_of_week"] = p.timestamp.dt.dayofweek; p["abs_error"] = abs(p.actual_Wh - p.hybrid_prediction_Wh); p["signed_error"] = p.hybrid_prediction_Wh - p.actual_Wh
    q = df.loc[: int(len(df) * .6), TARGET].quantile([.75, .95]).to_numpy()
    p["regime"] = np.select([p.actual_Wh <= q[0], p.actual_Wh >= q[1]], ["low", "high"], default="middle")
    p["spike"] = p.actual_Wh >= q[1]
    rows = []
    for label, group in p.groupby("hour"):
        rows.append({"analysis": "hour", "group": int(label), **metric_row(group.actual_Wh, group.hybrid_prediction_Wh)})
    for label, group in p.groupby("day_of_week"):
        rows.append({"analysis": "day_of_week", "group": int(label), **metric_row(group.actual_Wh, group.hybrid_prediction_Wh)})
    for label, group in p.groupby("regime"):
        rows.append({"analysis": "regime", "group": label, **metric_row(group.actual_Wh, group.hybrid_prediction_Wh)})
    for label, group in p.groupby("fallback"):
        rows.append({"analysis": "fallback", "group": int(label), **metric_row(group.actual_Wh, group.hybrid_prediction_Wh)})
    pd.DataFrame(rows).to_csv(OUT / "ERROR_ANALYSIS.csv", index=False)


def main() -> None:
    ensure_dirs(); df = add_features(load_data()); audit = audit_dataset(df); (OUT / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8"); write_feature_matrix()
    n = len(df); a_train = 14803 if n == 19735 else int(round(n * .75)); a_bounds = {"train_end": a_train, "val_end": a_train, "n": n}; b_train = int(n * .60); b_val = int(n * .80); b_bounds = {"train_end": b_train, "val_end": b_val, "n": n}
    track_a_1, pred_a_1, tune_a_1 = evaluate_variant(df, "A1_common_predictors", False, False, a_bounds, [1, 2, 3], "A")
    track_a_2, pred_a_2, tune_a_2 = evaluate_variant(df, "A2_historical_energy_augmented", False, True, a_bounds, [1, 2, 3], "A")
    track_b_1, pred_b_1, tune_b_1 = evaluate_variant(df, "B1_history_time", True, True, b_bounds, [1, 2, 3], "B", False)
    track_b_2, pred_b_2, tune_b_2 = evaluate_variant(df, "B2_history_time_environment_origin", True, True, b_bounds, [1, 2, 3], "B", True)
    track_a = pd.concat([track_a_1, track_a_2], ignore_index=True); track_b = pd.concat([track_b_1, track_b_2], ignore_index=True); preds = pd.concat([pred_a_1, pred_a_2, pred_b_1, pred_b_2], ignore_index=True); tuning = pd.DataFrame(tune_a_1 + tune_a_2 + tune_b_1 + tune_b_2)
    track_a.to_csv(OUT / "TRACK_A_RESULTS.csv", index=False); track_b.to_csv(OUT / "TRACK_B_RESULTS.csv", index=False); preds.to_csv(PRED / "all_predictions.csv", index=False); tuning.to_csv(OUT / "HYPERPARAMETER_TUNING.csv", index=False)
    selected = track_b[(track_b.model == "Hybrid FTS-RF residual") & (track_b.split == "validation")].sort_values("MAE_Wh").iloc[0]; best_key = {"variant": selected.variant, "fts_order": int(selected.fts_order)}
    order_df = track_b.copy(); write_reports(df, audit, track_a, track_b, order_df, tuning, best_key); error_analysis(preds, best_key, df); write_plots(df, preds, order_df, best_key)
    (CFG / "stage8_config.json").write_text(json.dumps({"random_state": RANDOM_STATE, "fts_intervals": 15, "track_a_train_rows": a_train, "track_b_train_rows": b_train, "track_b_validation_end": b_val, "selected_track_b": best_key, "paper_reference": "Candanedo et al. 2017 Table 5"}, indent=2), encoding="utf-8")
    archive = ROOT / "stage8_paper_benchmark.zip"; shutil.make_archive(str(archive.with_suffix("")), "zip", OUT)
    print(json.dumps({"output": str(OUT), "zip": str(archive), "track_b_selected": best_key}, indent=2))


if __name__ == "__main__":
    main()
