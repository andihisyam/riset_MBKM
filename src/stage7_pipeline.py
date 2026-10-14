from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.stage2_pipeline import (
    ALL_DIRECT_HORIZONS,
    BASE_FEATURES,
    RANDOM_STATE,
    ROLLING_FEATURES,
    TARGET,
    TIMESTAMP,
    interval_bounds,
    metrics,
    split_bounds,
    states,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "stage7"
FIGURES = ROOT / "figures" / "stage7"
REPORTS = ROOT / "reports" / "stage7"
MODELS = ROOT / "models" / "stage7"
CONFIGS = ROOT / "configs" / "stage7"
HORIZONS = ALL_DIRECT_HORIZONS
HIST_COLS = BASE_FEATURES + ROLLING_FEATURES
FEATURE_COLS = HIST_COLS
FIXED_THRESHOLD = 0.31


@dataclass
class Bounds:
    train_end: int
    val_end: int
    n: int


def ensure_dirs() -> None:
    for p in [RESULTS, FIGURES, REPORTS, MODELS, CONFIGS]:
        p.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[stage7] {msg}", flush=True)


def rmse(a, p) -> float:
    a = np.asarray(a, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def summarize(model: str, frame: pd.DataFrame, notes: str = "") -> dict:
    daily = frame.groupby(["split", "origin_index", "origin"], as_index=False).agg(actual_daily=("actual", "sum"), predicted_daily=("predicted", "sum"))
    test = frame[frame["split"].eq("test")]
    test_daily = daily[daily["split"].eq("test")]
    out = {"model": model, **metrics(test["actual"], test["predicted"], "Hourly "), **metrics(test_daily["actual_daily"], test_daily["predicted_daily"], "Daily "), "windows": int(test_daily["origin_index"].nunique()), "Notes": notes}
    out["Bias"] = float((test_daily["predicted_daily"] - test_daily["actual_daily"]).mean())
    return out


def make_origins(bounds: Bounds, split: str) -> list[int]:
    if split == "validation":
        start = max(bounds.train_end - 1, 168)
        end = bounds.val_end - 24
    else:
        start = max(bounds.val_end - 1, 168)
        end = bounds.n - 24
    return list(range(start, end, 24))


def target_frame(features: pd.DataFrame, bounds: Bounds, horizon: int, fit_until: int) -> pd.DataFrame:
    df = features[FEATURE_COLS + [TIMESTAMP]].copy()
    df["target"] = features[TARGET].shift(-horizon)
    df["target_index"] = df.index + horizon
    return df[(df["target_index"] < fit_until)].dropna(subset=FEATURE_COLS + ["target"])


def model_candidates() -> dict[str, list[dict]]:
    return {
        "XGBoost": [
            {"n_estimators": 250, "max_depth": 2, "learning_rate": 0.04, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3},
            {"n_estimators": 350, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3},
        ],
        "RandomForest": [
            {"n_estimators": 180, "max_depth": 8, "min_samples_leaf": 2},
            {"n_estimators": 260, "max_depth": 12, "min_samples_leaf": 2},
        ],
        "LightGBM": [
            {"n_estimators": 180, "learning_rate": 0.04, "num_leaves": 15, "min_child_samples": 15},
            {"n_estimators": 260, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 20},
        ],
        "CatBoost": [
            {"iterations": 180, "depth": 4, "learning_rate": 0.04, "l2_leaf_reg": 3},
            {"iterations": 260, "depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 5},
        ],
        "SVR": [
            {"C": 5.0, "epsilon": 0.03, "gamma": "scale"},
            {"C": 20.0, "epsilon": 0.02, "gamma": "scale"},
        ],
    }


def make_model(name: str, params: dict):
    if name == "XGBoost":
        return XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=1, verbosity=0, **params)
    if name == "RandomForest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1, **params)
    if name == "LightGBM":
        return LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **params)
    if name == "CatBoost":
        return CatBoostRegressor(random_seed=RANDOM_STATE, loss_function="RMSE", verbose=False, thread_count=1, **params)
    if name == "SVR":
        return make_pipeline(StandardScaler(), SVR(**params))
    raise ValueError(name)


def fit_horizon_models(features: pd.DataFrame, bounds: Bounds, alg: str, params: dict, fit_until: int) -> tuple[dict[int, object], float]:
    models = {}
    t0 = time.perf_counter()
    for h in HORIZONS:
        frame = target_frame(features, bounds, h, fit_until)
        model = make_model(alg, params)
        model.fit(frame[FEATURE_COLS], frame["target"])
        models[h] = model
    return models, time.perf_counter() - t0


def predict_direct(hourly: pd.DataFrame, features: pd.DataFrame, models: dict[int, object], alg: str, origins: list[int], split: str, train_seconds: float, params: dict) -> tuple[pd.DataFrame, float]:
    rows = []
    t0 = time.perf_counter()
    for o in origins:
        x = features.loc[[o], FEATURE_COLS]
        for h in HORIZONS:
            pred = max(float(models[h].predict(x)[0]), 0.0)
            rows.append({"split": split, "family": "Direct", "algorithm": alg, "fts_order": "", "model": f"{alg} Direct", "origin_index": o, "origin": hourly.loc[o, TIMESTAMP], "horizon": h, "actual": float(hourly.loc[o + h, TARGET]), "predicted": pred, "train_seconds": train_seconds, "inference_seconds": np.nan, "params": json.dumps(params, sort_keys=True)})
    infer = time.perf_counter() - t0
    out = pd.DataFrame(rows)
    out["inference_seconds"] = infer
    return out, infer


def build_rules(values: np.ndarray, bounds_arr: np.ndarray, order: int):
    st = states(values, bounds_arr)
    rules: dict[tuple[int, ...], list[int]] = {}
    for i in range(order, len(st)):
        rules.setdefault(tuple(int(x) for x in st[i - order : i]), []).append(int(st[i]))
    centers = (bounds_arr[:-1] + bounds_arr[1:]) / 2
    return rules, centers, float(np.mean(values))


def fts_next(history: list[float], rules, centers, fallback: float, order: int, bounds_arr: np.ndarray) -> tuple[float, bool]:
    if len(history) < order:
        return fallback, True
    key = tuple(int(x) for x in states(np.asarray(history[-order:]), bounds_arr))
    consequents = rules.get(key)
    if not consequents:
        return fallback, True
    vals, counts = np.unique(consequents, return_counts=True)
    return float(np.average(centers[vals], weights=counts)), False


def fts_matrix(hourly: pd.DataFrame, origins: list[int], order: int, bounds_arr: np.ndarray, rule_until: int) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    values = hourly[TARGET].to_numpy(dtype=float)
    rules, centers, fallback = build_rules(values[:rule_until], bounds_arr, order)
    out, diag = {}, []
    for o in origins:
        history = list(values[: o + 1])
        preds, misses = [], 0
        for h in HORIZONS:
            value, missed = fts_next(history, rules, centers, fallback, order, bounds_arr)
            value = max(value, 0.0)
            preds.append(value)
            history.append(value)
            misses += int(missed)
        out[o] = np.asarray(preds)
        diag.append({"origin_index": o, "order": order, "unseen_or_fallback_steps": misses})
    return out, pd.DataFrame(diag)


def fts_rows(hourly: pd.DataFrame, matrix: dict[int, np.ndarray], order: int, origins: list[int], split: str) -> pd.DataFrame:
    rows = []
    for o in origins:
        for h in HORIZONS:
            rows.append({"split": split, "family": "FTS", "algorithm": "WeightedFTS", "fts_order": order, "model": f"Weighted FTS order {order}", "origin_index": o, "origin": hourly.loc[o, TIMESTAMP], "horizon": h, "actual": float(hourly.loc[o + h, TARGET]), "predicted": float(matrix[o][h - 1])})
    return pd.DataFrame(rows)


def rolling_fts_for_residual_training(hourly: pd.DataFrame, order: int, bounds_arr: np.ndarray, fit_until: int) -> dict[int, np.ndarray]:
    values = hourly[TARGET].to_numpy(dtype=float)
    out = {}
    for o in range(max(168, order), fit_until - 24):
        rules, centers, fallback = build_rules(values[: o + 1], bounds_arr, order)
        history = list(values[: o + 1])
        preds = []
        for _ in HORIZONS:
            value, _ = fts_next(history, rules, centers, fallback, order, bounds_arr)
            value = max(value, 0.0)
            preds.append(value)
            history.append(value)
        out[o] = np.asarray(preds)
    return out


def residual_training_frame(hourly: pd.DataFrame, features: pd.DataFrame, matrix: dict[int, np.ndarray], fit_until: int, horizon: int) -> pd.DataFrame:
    rows = []
    for o, preds in matrix.items():
        target_index = o + horizon
        if target_index >= fit_until:
            continue
        rows.append({"origin_index": o, "residual": float(hourly.loc[target_index, TARGET] - preds[horizon - 1])})
    base = pd.DataFrame(rows).set_index("origin_index")
    frame = features.loc[base.index, FEATURE_COLS].copy()
    frame["residual"] = base["residual"]
    return frame.dropna(subset=FEATURE_COLS + ["residual"])


def fit_residual_models(hourly: pd.DataFrame, features: pd.DataFrame, alg: str, params: dict, order: int, bounds_arr: np.ndarray, fit_until: int, train_fts: dict[int, np.ndarray] | None = None) -> tuple[dict[int, object], float]:
    if train_fts is None:
        train_fts = rolling_fts_for_residual_training(hourly, order, bounds_arr, fit_until)
    models = {}
    t0 = time.perf_counter()
    for h in HORIZONS:
        frame = residual_training_frame(hourly, features, train_fts, fit_until, h)
        model = make_model(alg, params)
        model.fit(frame[FEATURE_COLS], frame["residual"])
        models[h] = model
    return models, time.perf_counter() - t0


def predict_residual(hourly: pd.DataFrame, features: pd.DataFrame, models: dict[int, object], alg: str, order: int, fts_pred: dict[int, np.ndarray], origins: list[int], split: str, train_seconds: float, params: dict) -> tuple[pd.DataFrame, float]:
    rows = []
    t0 = time.perf_counter()
    for o in origins:
        x = features.loc[[o], FEATURE_COLS]
        for h in HORIZONS:
            residual_hat = float(models[h].predict(x)[0])
            fts_value = float(fts_pred[o][h - 1])
            pred = max(fts_value + residual_hat, 0.0)
            rows.append({"split": split, "family": "Residual", "algorithm": alg, "fts_order": order, "model": f"Residual FTS{order}-{alg}", "origin_index": o, "origin": hourly.loc[o, TIMESTAMP], "horizon": h, "actual": float(hourly.loc[o + h, TARGET]), "fts_prediction": fts_value, "residual_prediction": residual_hat, "predicted": pred, "train_seconds": train_seconds, "inference_seconds": np.nan, "params": json.dumps(params, sort_keys=True)})
    infer = time.perf_counter() - t0
    out = pd.DataFrame(rows)
    out["inference_seconds"] = infer
    return out, infer


def daily_mae(frame: pd.DataFrame) -> float:
    d = frame.groupby("origin_index").agg(actual=("actual", "sum"), predicted=("predicted", "sum"))
    return float(np.mean(np.abs(d["actual"] - d["predicted"])))


def choose_threshold(direct: pd.DataFrame, residual: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    candidates = np.unique(np.r_[np.linspace(0.15, 0.90, 16), np.quantile(direct["predicted"], np.linspace(0.05, 0.95, 19)), FIXED_THRESHOLD])
    rows = []
    for th in candidates:
        hy = apply_gate(direct, residual, th, label="validation_candidate")
        rows.append({"threshold": float(th), "validation_daily_mae": daily_mae(hy), "residual_share": float((direct["predicted"] <= th).mean())})
    out = pd.DataFrame(rows).sort_values(["validation_daily_mae", "threshold"])
    return float(out.iloc[0]["threshold"]), out


def apply_gate(direct: pd.DataFrame, residual: pd.DataFrame, threshold: float, label: str) -> pd.DataFrame:
    d = direct.sort_values(["origin_index", "horizon"]).reset_index(drop=True)
    r = residual.sort_values(["origin_index", "horizon"]).reset_index(drop=True)
    use_resid = d["predicted"] <= threshold
    out = d.copy()
    out["family"] = "Hybrid"
    out["model"] = label
    out["threshold"] = threshold
    out["gate_condition"] = use_resid
    out["selected_expert"] = np.where(use_resid, r["model"], d["model"])
    out["direct_prediction"] = d["predicted"]
    out["residual_prediction"] = r["predicted"]
    out["predicted"] = np.where(use_resid, r["predicted"], d["predicted"])
    out["fts_order"] = r["fts_order"].iloc[0]
    return out


def write_preflight(hourly: pd.DataFrame, features: pd.DataFrame, bounds: Bounds, fts_orders: pd.DataFrame) -> None:
    report = f"""# Stage 7 Preflight Audit

## Artefak yang digunakan

- Dataset hourly: `data/hourly_energy_clean.csv`
- Fitur: `data/hourly_features.csv`
- Target: `{TARGET}` dalam kWh
- Jumlah observasi hourly lengkap: {len(hourly):,}
- Feature columns: {', '.join(FEATURE_COLS)}

## Split kronologis

- Train: index 0 sampai {bounds.train_end - 1}; tanggal {hourly.loc[0, TIMESTAMP]} sampai {hourly.loc[bounds.train_end - 1, TIMESTAMP]}
- Validation: index {bounds.train_end} sampai {bounds.val_end - 1}; tanggal {hourly.loc[bounds.train_end, TIMESTAMP]} sampai {hourly.loc[bounds.val_end - 1, TIMESTAMP]}
- Test: index {bounds.val_end} sampai {bounds.n - 1}; tanggal {hourly.loc[bounds.val_end, TIMESTAMP]} sampai {hourly.loc[bounds.n - 1, TIMESTAMP]}

## Audit horizon dan fitur

Seluruh model Stage 7 memakai origin yang sama untuk H1-H24. Fitur lag dan rolling sudah tersedia pada forecast origin karena dihitung dari nilai historis yang digeser satu jam. Tidak ada temperatur/kelembapan masa depan yang dimasukkan ke Direct ML Stage 7.

## FTS dan threshold historis

Stage 2 memilih quantile 15 interval dan Weighted FTS. Stage 7 membandingkan order 1, 2, dan 3 dengan jumlah interval yang sama. Threshold tetap 0.31 kWh dipakai sebagai pembanding Gate B historis; threshold validasi dipilih hanya dari validation Daily MAE.

## Catatan validitas

Periode test Stage 7 sudah pernah dipakai dalam Stage 1-6, sehingga Stage 7 adalah evaluasi lanjutan eksploratif, bukan holdout independen sepenuhnya. Residual FTS-ML memakai prediksi FTS rolling untuk membentuk residual training agar tidak memakai prediksi FTS in-sample yang terlalu optimistis.

## Reproduksi order FTS validasi

{fts_orders.to_markdown(index=False)}
"""
    (REPORTS / "stage7_preflight_audit.md").write_text(report, encoding="utf-8")


def plot_outputs(direct, residual, hybrid, horizon, daily, ranking, threshold_analysis, tuning):
    plt.figure(figsize=(9, 5))
    direct.sort_values("Daily MAE").plot(x="model", y="Daily MAE", kind="bar", legend=False, ax=plt.gca(), color="#1f77b4")
    plt.title("Perbandingan Daily MAE Model Direct")
    plt.ylabel("Daily MAE (kWh)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / "stage7_01_direct_daily_mae.png", dpi=160)
    plt.close()

    combo = pd.concat([direct.assign(type="Direct"), residual.assign(type="Residual"), hybrid.assign(type=hybrid["threshold_type"])], ignore_index=True)
    pivot = combo.pivot_table(index="algorithm", columns="type", values="Daily MAE", aggfunc="min")
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.title("Direct vs Residual vs Hybrid per Algoritma")
    plt.ylabel("Daily MAE (kWh)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES / "stage7_02_direct_residual_hybrid.png", dpi=160)
    plt.close()

    best_hybrid = hybrid[hybrid["threshold_type"].eq("Validation threshold")].pivot_table(index="algorithm", columns="fts_order", values="Daily MAE", aggfunc="min")
    plt.figure(figsize=(8, 5))
    plt.imshow(best_hybrid, aspect="auto", cmap="viridis_r")
    plt.colorbar(label="Daily MAE (kWh)")
    plt.xticks(range(len(best_hybrid.columns)), best_hybrid.columns)
    plt.yticks(range(len(best_hybrid.index)), best_hybrid.index)
    plt.title("Heatmap Daily MAE Hybrid: Algoritma vs Orde FTS")
    for i in range(best_hybrid.shape[0]):
        for j in range(best_hybrid.shape[1]):
            plt.text(j, i, f"{best_hybrid.iloc[i, j]:.2f}", ha="center", va="center", color="white")
    plt.tight_layout()
    plt.savefig(FIGURES / "stage7_03_hybrid_heatmap.png", dpi=160)
    plt.close()

    hy = hybrid[hybrid["threshold_type"].eq("Validation threshold")].copy()
    hy.sort_values("improvement_vs_direct_pct").plot(x="model", y="improvement_vs_direct_pct", kind="barh", figsize=(9, 7), legend=False, color="#2ca02c")
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Improvement Hybrid terhadap Direct")
    plt.xlabel("Improvement Daily MAE (%)")
    plt.tight_layout()
    plt.savefig(FIGURES / "stage7_04_hybrid_improvement.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    for name, group in horizon[horizon["model"].isin(ranking["model"].head(5))].groupby("model"):
        plt.plot(group["horizon"], group["MAE"], marker="o", label=name)
    plt.title("MAE berdasarkan Horizon H1-H24")
    plt.xlabel("Horizon")
    plt.ylabel("MAE (kWh)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "stage7_05_horizon_mae.png", dpi=160)
    plt.close()

    best_models = ranking["model"].head(5).tolist()
    first_origin = daily["origin_index"].min()
    hourly_rows = pd.read_csv(RESULTS / "stage7_hourly_predictions.csv")
    plt.figure(figsize=(10, 5))
    actual = hourly_rows[(hourly_rows["origin_index"].eq(first_origin)) & (hourly_rows["model"].eq(best_models[0]))].sort_values("horizon")
    plt.plot(actual["horizon"], actual["actual"], color="black", linewidth=2.5, label="Aktual")
    for m in best_models:
        g = hourly_rows[(hourly_rows["origin_index"].eq(first_origin)) & (hourly_rows["model"].eq(m))].sort_values("horizon")
        if not g.empty:
            plt.plot(g["horizon"], g["predicted"], marker="o", label=m)
    plt.title("Aktual vs Prediksi 24 Jam untuk Model Terbaik")
    plt.xlabel("Horizon")
    plt.ylabel("kWh")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES / "stage7_06_actual_vs_prediction_24h.png", dpi=160)
    plt.close()

    best = ranking.iloc[0]["model"]
    g = daily[daily["model"].eq(best)].copy()
    plt.figure(figsize=(9, 5))
    plt.plot(pd.to_datetime(g["origin"]), g["actual_daily"], marker="o", label="Aktual")
    plt.plot(pd.to_datetime(g["origin"]), g["predicted_daily"], marker="o", label=best)
    plt.title("Prediksi Total Energi Harian vs Aktual")
    plt.ylabel("kWh")
    plt.xticks(rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "stage7_07_daily_total_actual_vs_pred.png", dpi=160)
    plt.close()

    err = daily[daily["model"].isin(best_models)].copy()
    err["abs_error"] = (err["actual_daily"] - err["predicted_daily"]).abs()
    err.boxplot(column="abs_error", by="model", figsize=(10, 5), rot=25)
    plt.title("Distribusi Absolute Error antar Forecast Origin")
    plt.suptitle("")
    plt.ylabel("Absolute Error Harian (kWh)")
    plt.tight_layout()
    plt.savefig(FIGURES / "stage7_08_error_distribution.png", dpi=160)
    plt.close()

    threshold_analysis.groupby(["algorithm", "fts_order", "threshold_type"])["test_residual_share"].max().reset_index().pivot_table(index="algorithm", columns="threshold_type", values="test_residual_share", aggfunc="mean").plot(kind="bar", figsize=(9, 5))
    plt.title("Frekuensi Pemilihan Residual oleh Threshold")
    plt.ylabel("Share residual")
    plt.tight_layout()
    plt.savefig(FIGURES / "stage7_09_residual_selection_frequency.png", dpi=160)
    plt.close()

    tuning.groupby(["algorithm", "family"])["train_seconds"].sum().reset_index().pivot(index="algorithm", columns="family", values="train_seconds").plot(kind="bar", figsize=(9, 5))
    plt.title("Perbandingan Waktu Komputasi Training")
    plt.ylabel("Detik")
    plt.tight_layout()
    plt.savefig(FIGURES / "stage7_10_compute_time.png", dpi=160)
    plt.close()


def run_stage7() -> None:
    ensure_dirs()
    hourly = pd.read_csv(ROOT / "data" / "hourly_energy_clean.csv", parse_dates=[TIMESTAMP])
    features = pd.read_csv(ROOT / "data" / "hourly_features.csv", parse_dates=[TIMESTAMP])
    b0 = split_bounds(len(hourly))
    bounds = Bounds(b0.train_end, b0.val_end, b0.n)
    val_origins, test_origins = make_origins(bounds, "validation"), make_origins(bounds, "test")
    CONFIGS.joinpath("stage7_config.json").write_text(json.dumps({"random_state": RANDOM_STATE, "feature_cols": FEATURE_COLS, "fixed_threshold": FIXED_THRESHOLD, "val_origins": val_origins, "test_origins": test_origins}, indent=2), encoding="utf-8")

    log("Phase 1: FTS order audit")
    train_bounds = interval_bounds(hourly.loc[: bounds.train_end - 1, TARGET], "quantile", 15)
    fts_rows_all, fts_order_rows, fts_diag_rows = [], [], []
    for order in [1, 2, 3]:
        rules, _, _ = build_rules(hourly.loc[: bounds.train_end - 1, TARGET].to_numpy(), train_bounds, order)
        val_matrix, val_diag = fts_matrix(hourly, val_origins, order, train_bounds, bounds.train_end)
        test_matrix, test_diag = fts_matrix(hourly, test_origins, order, train_bounds, bounds.val_end)
        val_frame = fts_rows(hourly, val_matrix, order, val_origins, "validation")
        test_frame = fts_rows(hourly, test_matrix, order, test_origins, "test")
        both = pd.concat([val_frame, test_frame], ignore_index=True)
        fts_rows_all.append(both)
        row = summarize(f"Weighted FTS order {order}", both, "Quantile 15, weighted recursive H1-H24")
        row.update({"order": order, "n_patterns": len(rules), "rare_patterns": int(sum(1 for v in rules.values() if len(v) <= 2)), "validation_daily_mae": daily_mae(val_frame), "validation_rmse_hourly": rmse(val_frame["actual"], val_frame["predicted"]), "fallback_steps_validation": int(val_diag["unseen_or_fallback_steps"].sum()), "fallback_steps_test": int(test_diag["unseen_or_fallback_steps"].sum())})
        fts_order_rows.append(row)
        fts_diag_rows.append(pd.concat([val_diag.assign(split="validation"), test_diag.assign(split="test")], ignore_index=True))
    fts_order_df = pd.DataFrame(fts_order_rows)
    fts_order_df.to_csv(RESULTS / "stage7_fts_order_comparison.csv", index=False)
    pd.concat(fts_diag_rows, ignore_index=True).to_csv(RESULTS / "stage7_fts_order_diagnostics.csv", index=False)
    write_preflight(hourly, features, bounds, fts_order_df[["order", "validation_daily_mae", "Daily MAE", "n_patterns", "rare_patterns", "fallback_steps_validation", "fallback_steps_test"]])

    log("Phase 2: Direct ML benchmark")
    candidates = model_candidates()
    tuning_rows, direct_frames, direct_summaries, best_direct_params = [], [], [], {}
    direct_val_by_alg, direct_test_by_alg = {}, {}
    for alg, params_list in candidates.items():
        best = None
        for idx, params in enumerate(params_list, 1):
            models, train_seconds = fit_horizon_models(features, bounds, alg, params, bounds.train_end)
            val_pred, infer_seconds = predict_direct(hourly, features, models, alg, val_origins, "validation", train_seconds, params)
            score = daily_mae(val_pred)
            tuning_rows.append({"phase": "direct", "family": "Direct", "algorithm": alg, "candidate": idx, "fts_order": "", "validation_daily_mae": score, "train_seconds": train_seconds, "inference_seconds": infer_seconds, "params": json.dumps(params, sort_keys=True)})
            if best is None or score < best["score"]:
                best = {"score": score, "params": params}
        best_direct_params[alg] = best["params"]
        final_models, train_seconds = fit_horizon_models(features, bounds, alg, best["params"], bounds.val_end)
        val_models, _ = fit_horizon_models(features, bounds, alg, best["params"], bounds.train_end)
        val_pred, _ = predict_direct(hourly, features, val_models, alg, val_origins, "validation", train_seconds, best["params"])
        test_pred, infer_seconds = predict_direct(hourly, features, final_models, alg, test_origins, "test", train_seconds, best["params"])
        combined = pd.concat([val_pred, test_pred], ignore_index=True)
        direct_val_by_alg[alg], direct_test_by_alg[alg] = val_pred, test_pred
        direct_frames.append(combined)
        row = summarize(f"{alg} Direct", combined, "Direct H1-H24 ML")
        row.update({"algorithm": alg, "family": "Direct", "best_params": json.dumps(best["params"], sort_keys=True), "validation_daily_mae": best["score"], "train_seconds": train_seconds, "inference_seconds": infer_seconds})
        direct_summaries.append(row)
        joblib.dump({"algorithm": alg, "params": best["params"]}, MODELS / f"direct_{alg}_manifest.joblib")
        log(f"Direct {alg}: validation Daily MAE={best['score']:.4f}")
    direct_df = pd.DataFrame(direct_summaries)
    direct_df.to_csv(RESULTS / "stage7_direct_comparison.csv", index=False)

    log("Phase 3-4: Residual FTS-ML benchmark")
    residual_frames, residual_summaries, best_residual_params = [], [], {}
    residual_val, residual_test = {}, {}
    fts_val_matrices = {order: fts_matrix(hourly, val_origins, order, train_bounds, bounds.train_end)[0] for order in [1, 2, 3]}
    fts_test_matrices = {order: fts_matrix(hourly, test_origins, order, train_bounds, bounds.val_end)[0] for order in [1, 2, 3]}
    residual_train_cache = {
        (order, bounds.train_end): rolling_fts_for_residual_training(hourly, order, train_bounds, bounds.train_end)
        for order in [1, 2, 3]
    }
    residual_train_cache.update(
        {
            (order, bounds.val_end): rolling_fts_for_residual_training(hourly, order, train_bounds, bounds.val_end)
            for order in [1, 2, 3]
        }
    )
    for order in [1, 2, 3]:
        for alg, params_list in candidates.items():
            best = None
            for idx, params in enumerate(params_list, 1):
                models, train_seconds = fit_residual_models(hourly, features, alg, params, order, train_bounds, bounds.train_end, residual_train_cache[(order, bounds.train_end)])
                val_pred, infer_seconds = predict_residual(hourly, features, models, alg, order, fts_val_matrices[order], val_origins, "validation", train_seconds, params)
                score = daily_mae(val_pred)
                tuning_rows.append({"phase": "residual", "family": "Residual", "algorithm": alg, "candidate": idx, "fts_order": order, "validation_daily_mae": score, "train_seconds": train_seconds, "inference_seconds": infer_seconds, "params": json.dumps(params, sort_keys=True)})
                if best is None or score < best["score"]:
                    best = {"score": score, "params": params}
            best_residual_params[(alg, order)] = best["params"]
            val_models, _ = fit_residual_models(hourly, features, alg, best["params"], order, train_bounds, bounds.train_end, residual_train_cache[(order, bounds.train_end)])
            final_models, train_seconds = fit_residual_models(hourly, features, alg, best["params"], order, train_bounds, bounds.val_end, residual_train_cache[(order, bounds.val_end)])
            val_pred, _ = predict_residual(hourly, features, val_models, alg, order, fts_val_matrices[order], val_origins, "validation", train_seconds, best["params"])
            test_pred, infer_seconds = predict_residual(hourly, features, final_models, alg, order, fts_test_matrices[order], test_origins, "test", train_seconds, best["params"])
            combined = pd.concat([val_pred, test_pred], ignore_index=True)
            residual_val[(alg, order)], residual_test[(alg, order)] = val_pred, test_pred
            residual_frames.append(combined)
            row = summarize(f"Residual FTS{order}-{alg}", combined, "Weighted FTS + ML residual correction")
            row.update({"algorithm": alg, "fts_order": order, "family": "Residual", "best_params": json.dumps(best["params"], sort_keys=True), "validation_daily_mae": best["score"], "train_seconds": train_seconds, "inference_seconds": infer_seconds})
            residual_summaries.append(row)
            log(f"Residual FTS{order}-{alg}: validation Daily MAE={best['score']:.4f}")
    residual_df = pd.DataFrame(residual_summaries)
    residual_df.to_csv(RESULTS / "stage7_residual_comparison.csv", index=False)

    log("Phase 5: Hybrid threshold benchmark")
    hybrid_frames, hybrid_summaries, threshold_rows = [], [], []
    direct_summary_map = {r["algorithm"]: r for r in direct_summaries}
    for order in [1, 2, 3]:
        for alg in candidates:
            d_val, r_val = direct_val_by_alg[alg], residual_val[(alg, order)]
            d_test, r_test = direct_test_by_alg[alg], residual_test[(alg, order)]
            best_th, th_table = choose_threshold(d_val, r_val)
            for threshold_type, th in [("Fixed 0.31", FIXED_THRESHOLD), ("Validation threshold", best_th)]:
                val_h = apply_gate(d_val, r_val, th, f"Hybrid FTS{order}-{alg} {threshold_type}")
                val_h["split"] = "validation"
                test_h = apply_gate(d_test, r_test, th, f"Hybrid FTS{order}-{alg} {threshold_type}")
                test_h["split"] = "test"
                combined = pd.concat([val_h, test_h], ignore_index=True)
                hybrid_frames.append(combined)
                row = summarize(f"Hybrid FTS{order}-{alg} {threshold_type}", combined, "Threshold: residual if Direct prediction <= threshold")
                row.update({"algorithm": alg, "fts_order": order, "threshold_type": threshold_type, "threshold": th, "validation_daily_mae": daily_mae(val_h), "test_residual_share": float(test_h["gate_condition"].mean()), "validation_residual_share": float(val_h["gate_condition"].mean())})
                row["improvement_vs_direct_pct"] = (direct_summary_map[alg]["Daily MAE"] - row["Daily MAE"]) / direct_summary_map[alg]["Daily MAE"] * 100
                hybrid_summaries.append(row)
            th_table["algorithm"] = alg
            th_table["fts_order"] = order
            threshold_rows.append(th_table)
            log(f"Hybrid FTS{order}-{alg}: best validation threshold={best_th:.4f}")
    hybrid_df = pd.DataFrame(hybrid_summaries)
    hybrid_df.to_csv(RESULTS / "stage7_hybrid_comparison.csv", index=False)
    threshold_analysis = pd.concat(threshold_rows, ignore_index=True)
    fixed_actual = hybrid_df[["algorithm", "fts_order", "threshold_type", "threshold", "validation_daily_mae", "Daily MAE", "test_residual_share", "validation_residual_share", "improvement_vs_direct_pct"]].copy()
    fixed_actual.to_csv(RESULTS / "stage7_threshold_analysis.csv", index=False)

    log("Phase 6: outputs, figures, reports")
    hourly_predictions = pd.concat(direct_frames + residual_frames + hybrid_frames + fts_rows_all, ignore_index=True, sort=False)
    hourly_predictions["abs_error"] = (hourly_predictions["actual"] - hourly_predictions["predicted"]).abs()
    hourly_predictions.to_csv(RESULTS / "stage7_hourly_predictions.csv", index=False)
    daily_predictions = hourly_predictions.groupby(["model", "family", "algorithm", "fts_order", "split", "origin_index", "origin"], dropna=False, as_index=False).agg(actual_daily=("actual", "sum"), predicted_daily=("predicted", "sum"))
    daily_predictions["daily_abs_error"] = (daily_predictions["actual_daily"] - daily_predictions["predicted_daily"]).abs()
    daily_predictions.to_csv(RESULTS / "stage7_daily_predictions.csv", index=False)

    horizon_rows = []
    for (model, family, alg, order, split, h), group in hourly_predictions.groupby(["model", "family", "algorithm", "fts_order", "split", "horizon"], dropna=False):
        if split == "test":
            horizon_rows.append({"model": model, "family": family, "algorithm": alg, "fts_order": order, "horizon": h, **metrics(group["actual"], group["predicted"])})
    horizon_df = pd.DataFrame(horizon_rows)
    horizon_df.to_csv(RESULTS / "stage7_horizon_metrics.csv", index=False)

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(RESULTS / "stage7_tuning_log.csv", index=False)
    all_summary = pd.concat([direct_df, residual_df, hybrid_df, fts_order_df], ignore_index=True, sort=False)
    ranking = all_summary.sort_values("Daily MAE").reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(RESULTS / "stage7_final_model_ranking.csv", index=False)
    threshold_analysis.merge(fixed_actual, on=["algorithm", "fts_order"], how="left").to_csv(RESULTS / "stage7_threshold_grid_validation.csv", index=False)
    plot_outputs(direct_df, residual_df, hybrid_df, horizon_df, daily_predictions[daily_predictions["split"].eq("test")], ranking, fixed_actual, tuning_df)

    versions = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "xgboost": __import__("xgboost").__version__,
        "lightgbm": __import__("lightgbm").__version__,
        "catboost": __import__("catboost").__version__,
    }
    best_direct = direct_df.sort_values("Daily MAE").iloc[0]
    best_order = fts_order_df.sort_values("validation_daily_mae").iloc[0]
    best_residual = residual_df.assign(improvement=lambda d: d.apply(lambda r: (direct_df[direct_df["algorithm"].eq(r["algorithm"])]["Daily MAE"].iloc[0] - r["Daily MAE"]) / direct_df[direct_df["algorithm"].eq(r["algorithm"])]["Daily MAE"].iloc[0] * 100, axis=1)).sort_values("improvement", ascending=False).iloc[0]
    best_hybrid = hybrid_df.sort_values("Daily MAE").iloc[0]
    xgb_stage5 = pd.read_csv(ROOT / "results" / "stage5_final_model_comparison.csv")
    xgb_old = float(xgb_stage5[xgb_stage5["model"].eq("XGBoost Direct")]["Daily MAE"].iloc[0])
    stat_rows = []
    if len(daily_predictions[daily_predictions["model"].eq(best_direct["model"])]) and len(daily_predictions[daily_predictions["model"].eq(best_hybrid["model"])]):
        a = daily_predictions[(daily_predictions["split"].eq("test")) & (daily_predictions["model"].eq(best_direct["model"]))].sort_values("origin_index")["daily_abs_error"].to_numpy()
        c = daily_predictions[(daily_predictions["split"].eq("test")) & (daily_predictions["model"].eq(best_hybrid["model"]))].sort_values("origin_index")["daily_abs_error"].to_numpy()
        if len(a) == len(c) and len(a) > 0:
            try:
                stat_rows.append({"comparison": f"{best_direct['model']} vs {best_hybrid['model']}", "wilcoxon_p": float(wilcoxon(a, c).pvalue), "n": len(a)})
            except ValueError as exc:
                stat_rows.append({"comparison": f"{best_direct['model']} vs {best_hybrid['model']}", "wilcoxon_p": np.nan, "n": len(a), "note": str(exc)})
    pd.DataFrame(stat_rows).to_csv(RESULTS / "stage7_pairwise_stats.csv", index=False)

    repro = f"""# Stage 7 Reproducibility Report

- Runtime completed: {pd.Timestamp.now().isoformat()}
- Library versions: `{json.dumps(versions)}`
- Random seed: {RANDOM_STATE}
- Split: chronological 70/15/15
- Validation origins: {len(val_origins)}
- Test origins: {len(test_origins)}
- Models: {', '.join(candidates)}
- FTS orders: 1, 2, 3
- Fixed threshold: {FIXED_THRESHOLD}
- Files generated under `results/stage7`, `figures/stage7`, `reports/stage7`, `models/stage7`, and `configs/stage7`.
"""
    (REPORTS / "stage7_reproducibility_report.md").write_text(repro, encoding="utf-8")

    final = f"""# Stage 7 Final Report

## Ringkasan hasil

- Algoritma Direct terbaik tanpa FTS: **{best_direct['model']}** dengan Daily MAE **{best_direct['Daily MAE']:.4f} kWh**.
- Orde FTS terbaik berdasarkan validasi: **Weighted FTS order {int(best_order['order'])}** dengan validation Daily MAE **{best_order['validation_daily_mae']:.4f} kWh**.
- Algoritma yang paling terbantu Residual FTS-ML: **{best_residual['model']}** dengan improvement terhadap Direct algoritma terkait **{best_residual['improvement']:.2f}%**.
- Hybrid berbasis ambang terbaik: **{best_hybrid['model']}** dengan Daily MAE **{best_hybrid['Daily MAE']:.4f} kWh** dan improvement terhadap Direct algoritma terkait **{best_hybrid.get('improvement_vs_direct_pct', np.nan):.2f}%**.
- XGBoost Direct Stage sebelumnya: **{xgb_old:.4f} kWh**. Model terbaik Stage 7 {'mengalahkan' if best_hybrid['Daily MAE'] < xgb_old else 'tidak mengalahkan'} angka tersebut pada protokol Stage 7.

## Interpretasi

Primary metric adalah Daily MAE. Perbedaan Hourly MAE tetap dilaporkan karena model yang baik pada total harian belum tentu paling baik per jam. Stage 7 tidak menganggap hybrid pasti lebih baik; hasil final mengikuti ranking CSV.

## Stabilitas dan risiko

Periode test sudah pernah disentuh pada Stage 1-6, sehingga hasil ini adalah evaluasi eksploratif lanjutan. Jumlah origin non-overlap tetap terbatas, sehingga uji statistik harus dibaca hati-hati dan tidak cukup untuk klaim signifikan bila p-value tidak mendukung.

## Biaya komputasi

Waktu training dan inference per kandidat dicatat dalam `stage7_tuning_log.csv`. Biaya tambahan residual/hybrid perlu dibandingkan dengan besarnya improvement Daily MAE.

## File utama

- `stage7_direct_comparison.csv`
- `stage7_fts_order_comparison.csv`
- `stage7_residual_comparison.csv`
- `stage7_hybrid_comparison.csv`
- `stage7_final_model_ranking.csv`
- `stage7_hourly_predictions.csv`
- `stage7_daily_predictions.csv`
- `stage7_threshold_analysis.csv`
"""
    (REPORTS / "stage7_final_report.md").write_text(final, encoding="utf-8")
    log("Stage 7 complete")


if __name__ == "__main__":
    run_stage7()
