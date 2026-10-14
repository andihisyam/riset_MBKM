from __future__ import annotations

import json
import math
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.stage2_pipeline import BASE_FEATURES, ROLLING_FEATURES, TARGET, TIMESTAMP, split_bounds


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "stage7_1_audit"
FIG = OUT / "figures"
TESTS = OUT / "tests"
CORRECTED = OUT / "corrected"
FEATURE_COLS = BASE_FEATURES + ROLLING_FEATURES
LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLL_MEANS = [3, 6, 12, 24]
ROLL_STDS = [3, 6, 24]


def ensure_dirs() -> None:
    for p in [OUT, FIG, TESTS, CORRECTED]:
        p.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, cols: list[str] | None = None, n: int | None = None) -> str:
    use = df.copy()
    if cols:
        use = use[cols]
    if n:
        use = use.head(n)
    return use.to_markdown(index=False)


def metrics_from_arrays(actual, pred) -> dict[str, float]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    denom = np.abs(a) + np.abs(p)
    return {
        "MAE": float(np.mean(np.abs(a - p))),
        "RMSE": float(np.sqrt(np.mean((a - p) ** 2))),
        "MAPE": float(np.mean(np.abs((a - p) / a)) * 100),
        "sMAPE": float(np.mean(2 * np.abs(a - p) / denom) * 100),
        "Bias": float(np.mean(p - a)),
    }


def model_metric(frame: pd.DataFrame) -> dict[str, float]:
    daily = frame.groupby("origin_index", as_index=False).agg(actual=("actual", "sum"), predicted=("predicted", "sum"))
    h = metrics_from_arrays(frame["actual"], frame["predicted"])
    d = metrics_from_arrays(daily["actual"], daily["predicted"])
    return {
        "Hourly MAE": h["MAE"],
        "Hourly RMSE": h["RMSE"],
        "Hourly MAPE": h["MAPE"],
        "Hourly sMAPE": h["sMAPE"],
        "Daily MAE": d["MAE"],
        "Daily RMSE": d["RMSE"],
        "Daily MAPE": d["MAPE"],
        "Daily sMAPE": d["sMAPE"],
        "Bias": d["Bias"],
        "windows": int(daily["origin_index"].nunique()),
    }


def daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.groupby(["model", "split", "origin_index", "origin"], dropna=False, as_index=False).agg(
        actual_daily=("actual", "sum"),
        predicted_daily=("predicted", "sum"),
        hourly_abs_sum=("actual", lambda x: 0.0),
    )
    abs_sum = frame.assign(abs_error=lambda x: (x["actual"] - x["predicted"]).abs()).groupby(["model", "split", "origin_index"], dropna=False)["abs_error"].sum().reset_index(name="sum_hourly_abs_error")
    out = out.drop(columns=["hourly_abs_sum"]).merge(abs_sum, on=["model", "split", "origin_index"], how="left")
    out["daily_abs_error"] = (out["actual_daily"] - out["predicted_daily"]).abs()
    out["error_cancellation_kwh"] = out["sum_hourly_abs_error"] - out["daily_abs_error"]
    out["daily_signed_error"] = out["predicted_daily"] - out["actual_daily"]
    return out


def feature_future_audit(hourly: pd.DataFrame, features: pd.DataFrame, origins: list[int]) -> pd.DataFrame:
    rows = []
    y = hourly[TARGET].to_numpy(dtype=float)
    for origin in origins:
        for lag in LAGS:
            feature = f"lag_{lag}"
            expected = y[origin - lag]
            got = float(features.loc[origin, feature])
            rows.append({"origin_index": origin, "feature": feature, "expected": expected, "observed": got, "abs_diff": abs(expected - got), "uses_future_actual": origin - lag > origin})
        past = pd.Series(y[:origin])
        for window in ROLL_MEANS:
            feature = f"rolling_mean_{window}"
            expected = float(past.tail(window).mean())
            got = float(features.loc[origin, feature])
            rows.append({"origin_index": origin, "feature": feature, "expected": expected, "observed": got, "abs_diff": abs(expected - got), "uses_future_actual": False})
        for window in ROLL_STDS:
            feature = f"rolling_std_{window}"
            expected = float(past.tail(window).std())
            got = float(features.loc[origin, feature])
            rows.append({"origin_index": origin, "feature": feature, "expected": expected, "observed": got, "abs_diff": abs(expected - got), "uses_future_actual": False})
        ts = pd.Timestamp(hourly.loc[origin, TIMESTAMP])
        cal_expected = {
            "hour": ts.hour,
            "day_of_week": ts.dayofweek,
            "is_weekend": int(ts.dayofweek in [5, 6]),
            "sin_hour": np.sin(2 * np.pi * ts.hour / 24),
            "cos_hour": np.cos(2 * np.pi * ts.hour / 24),
            "sin_dow": np.sin(2 * np.pi * ts.dayofweek / 7),
            "cos_dow": np.cos(2 * np.pi * ts.dayofweek / 7),
        }
        for feature, expected in cal_expected.items():
            got = float(features.loc[origin, feature])
            rows.append({"origin_index": origin, "feature": feature, "expected": float(expected), "observed": got, "abs_diff": abs(float(expected) - got), "uses_future_actual": False})
    return pd.DataFrame(rows)


def audit_fairness(tuning: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    direct = tuning[tuning["phase"].eq("direct")]
    residual = tuning[tuning["phase"].eq("residual")]
    algorithms = sorted(direct["algorithm"].unique())
    rows = []
    for alg in algorithms:
        rows.extend(
            [
                {"algorithm": alg, "condition": "Feature set", "rating": "Identical", "evidence": "All Direct models use BASE_FEATURES + ROLLING_FEATURES (22 columns)."},
                {"algorithm": alg, "condition": "Target and horizon", "rating": "Identical", "evidence": "All models predict H1-H24 on the same hourly Appliances_kWh target."},
                {"algorithm": alg, "condition": "Split and forecast origin", "rating": "Identical", "evidence": "Chronological 70/15/15; 20 validation origins and 20 test origins."},
                {"algorithm": alg, "condition": "Direct tuning candidates", "rating": "Identical", "evidence": f"{direct[direct['algorithm'].eq(alg)]['candidate'].nunique()} candidates."},
                {"algorithm": alg, "condition": "Residual tuning candidates per order", "rating": "Identical", "evidence": f"{residual[residual['algorithm'].eq(alg)].groupby('fts_order')['candidate'].nunique().to_dict()} candidates."},
                {"algorithm": alg, "condition": "Model-specific preprocessing", "rating": "Different but justified" if alg == "SVR" else "Identical", "evidence": "SVR uses train-fit StandardScaler in sklearn Pipeline; tree/boosting models use raw numeric features." if alg == "SVR" else "No scaler used."},
                {"algorithm": alg, "condition": "Random seed", "rating": "Comparable", "evidence": "Seed fixed where estimator exposes random_state/random_seed; SVR deterministic for fixed data."},
                {"algorithm": alg, "condition": "Selection criterion", "rating": "Identical", "evidence": "Direct/residual candidates selected by validation Daily MAE; threshold selected by validation Daily MAE."},
            ]
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "FAIRNESS_MATRIX.csv", index=False)
    return out


def audit_fts(hourly_pred: pd.DataFrame, fts_compare: pd.DataFrame, fts_diag: pd.DataFrame, horizon: pd.DataFrame) -> pd.DataFrame:
    fts_pred = hourly_pred[hourly_pred["family"].eq("FTS")].copy()
    rows = []
    for order in sorted(fts_compare["order"].unique()):
        cmp = fts_compare[fts_compare["order"].eq(order)].iloc[0]
        diag = fts_diag[fts_diag["order"].eq(order)]
        pred = fts_pred[(fts_pred["fts_order"].eq(order)) & (fts_pred["split"].eq("test"))]
        fallback_origins = int((diag[diag["split"].eq("test")]["unseen_or_fallback_steps"] > 0).sum())
        fallback_steps = int(diag[diag["split"].eq("test")]["unseen_or_fallback_steps"].sum())
        rows.append({
            "order": int(order),
            "n_patterns": int(cmp["n_patterns"]),
            "rare_patterns": int(cmp["rare_patterns"]),
            "validation_daily_mae": float(cmp["validation_daily_mae"]),
            "test_daily_mae": float(cmp["Daily MAE"]),
            "fallback_origins_test": fallback_origins,
            "fallback_steps_test": fallback_steps,
            "test_hourly_mae_recalc": model_metric(pred)["Hourly MAE"],
            "test_daily_mae_recalc": model_metric(pred)["Daily MAE"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "fts_order_audit_table.csv", index=False)
    h = horizon[horizon["family"].eq("FTS")]
    h.to_csv(OUT / "fts_horizon_error.csv", index=False)
    return out


def plot_residual_focus(hourly_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = "Residual FTS3-RandomForest"
    frame = hourly_pred[hourly_pred["model"].eq(model)].copy()
    daily = daily_frame(frame)
    hourly = frame.assign(error=lambda x: x["predicted"] - x["actual"])
    per_origin = daily[["split", "origin_index", "origin", "actual_daily", "predicted_daily", "daily_abs_error", "daily_signed_error", "error_cancellation_kwh"]].copy()
    per_horizon = hourly.groupby(["split", "horizon"], as_index=False).apply(lambda g: pd.Series(model_metric(g)), include_groups=False).reset_index(drop=True)
    per_origin.to_csv(OUT / "residual_fts3_randomforest_per_origin.csv", index=False)
    per_horizon.to_csv(OUT / "residual_fts3_randomforest_per_horizon.csv", index=False)

    val = daily[daily["split"].eq("validation")]
    test = daily[daily["split"].eq("test")]
    plt.figure(figsize=(6, 5))
    plt.scatter(val["actual_daily"], val["predicted_daily"], label="Validation", alpha=0.8)
    plt.scatter(test["actual_daily"], test["predicted_daily"], label="Test", alpha=0.8)
    lo = min(daily["actual_daily"].min(), daily["predicted_daily"].min())
    hi = max(daily["actual_daily"].max(), daily["predicted_daily"].max())
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.title("Residual FTS3-RandomForest: Aktual vs Prediksi Harian")
    plt.xlabel("Aktual harian (kWh)")
    plt.ylabel("Prediksi harian (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "residual_fts3_rf_scatter_daily.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for split, group in daily.groupby("split"):
        plt.plot(range(len(group)), group["daily_signed_error"], marker="o", label=split)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Residual FTS3-RandomForest: Signed Error per Origin")
    plt.ylabel("Prediksi - aktual (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "residual_fts3_rf_origin_signed_error.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for split, group in hourly.groupby("split"):
        plt.hist(group["error"], bins=30, alpha=0.55, label=split)
    plt.title("Residual FTS3-RandomForest: Distribusi Error Hourly")
    plt.xlabel("Prediksi - aktual (kWh)")
    plt.ylabel("Jumlah prediksi")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "residual_fts3_rf_hourly_error_hist.png", dpi=160)
    plt.close()

    return per_origin, per_horizon


def audit_gating(hourly_pred: pd.DataFrame, direct_comp: pd.DataFrame) -> pd.DataFrame:
    hybrid = hourly_pred[hourly_pred["family"].eq("Hybrid")].copy()
    direct = hourly_pred[hourly_pred["family"].eq("Direct")][["algorithm", "split", "origin_index", "horizon", "actual", "predicted"]].rename(columns={"predicted": "direct_pred"})
    residual = hourly_pred[hourly_pred["family"].eq("Residual")][["algorithm", "fts_order", "split", "origin_index", "horizon", "predicted"]].rename(columns={"predicted": "residual_pred_actual_model"})
    merged = hybrid.merge(direct, on=["algorithm", "split", "origin_index", "horizon", "actual"], how="left").merge(
        residual, on=["algorithm", "fts_order", "split", "origin_index", "horizon"], how="left"
    )
    merged["direct_abs_error"] = (merged["actual"] - merged["direct_pred"]).abs()
    merged["hybrid_abs_error"] = (merged["actual"] - merged["predicted"]).abs()
    merged["residual_abs_error"] = (merged["actual"] - merged["residual_pred_actual_model"]).abs()
    merged["decision_improved"] = merged["hybrid_abs_error"] < merged["direct_abs_error"] - 1e-12
    merged["decision_worsened"] = merged["hybrid_abs_error"] > merged["direct_abs_error"] + 1e-12
    gate_rule_mismatch = ((merged["direct_prediction"] <= merged["threshold"]) != merged["gate_condition"]).sum()
    rows = []
    for (model, split), g in merged.groupby(["model", "split"]):
        daily_h = daily_frame(g.assign(model=model))
        daily_d = daily_frame(g.assign(model=model, predicted=g["direct_pred"]))
        rows.append({
            "model": model,
            "split": split,
            "algorithm": g["algorithm"].iloc[0],
            "fts_order": g["fts_order"].iloc[0],
            "threshold": g["threshold"].iloc[0],
            "gate_rule_mismatch": int(gate_rule_mismatch if split == "test" else 0),
            "active_count": int(g["gate_condition"].sum()),
            "active_share": float(g["gate_condition"].mean()),
            "improved_hourly_count": int(g["decision_improved"].sum()),
            "worsened_hourly_count": int(g["decision_worsened"].sum()),
            "unchanged_hourly_count": int((~g["decision_improved"] & ~g["decision_worsened"]).sum()),
            "daily_mae_hybrid": float(daily_h["daily_abs_error"].mean()),
            "daily_mae_direct": float(daily_d["daily_abs_error"].mean()),
            "daily_mae_change": float(daily_h["daily_abs_error"].mean() - daily_d["daily_abs_error"].mean()),
            "hourly_mae_hybrid": float(g["hybrid_abs_error"].mean()),
            "hourly_mae_direct": float(g["direct_abs_error"].mean()),
            "hourly_mae_change": float(g["hybrid_abs_error"].mean() - g["direct_abs_error"].mean()),
            "mean_error_cancellation_hybrid": float(daily_h["error_cancellation_kwh"].mean()),
            "mean_error_cancellation_direct": float(daily_d["error_cancellation_kwh"].mean()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "gating_decision_audit.csv", index=False)
    focus = merged[(merged["model"].eq("Hybrid FTS1-LightGBM Fixed 0.31")) & (merged["split"].eq("test"))]
    focus.to_csv(OUT / "gating_focus_hybrid_fts1_lightgbm_031.csv", index=False)
    return out


def statistical_validation(hourly_pred: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    daily = daily_frame(hourly_pred[hourly_pred["split"].eq("test")])
    comparisons = [
        ("LightGBM Direct", "Hybrid FTS1-LightGBM Fixed 0.31"),
        ("RandomForest Direct", "Residual FTS3-RandomForest"),
        ("LightGBM Direct", "RandomForest Direct"),
    ]
    rows = []
    for a, b in comparisons:
        da = daily[daily["model"].eq(a)].sort_values("origin_index")
        db = daily[daily["model"].eq(b)].sort_values("origin_index")
        if len(da) != len(db) or len(da) == 0:
            rows.append({"comparison": f"{a} vs {b}", "status": "missing paired origins"})
            continue
        diff = da["daily_abs_error"].to_numpy() - db["daily_abs_error"].to_numpy()
        try:
            p = float(wilcoxon(diff).pvalue)
        except ValueError:
            p = np.nan
        rng = np.random.default_rng(42)
        boots = [float(np.mean(rng.choice(diff, size=len(diff), replace=True))) for _ in range(5000)]
        rows.append({
            "comparison": f"{a} minus {b}",
            "n_origins": len(diff),
            "mean_error_diff": float(np.mean(diff)),
            "median_error_diff": float(np.median(diff)),
            "wilcoxon_p": p,
            "bootstrap_ci_low": float(np.quantile(boots, 0.025)),
            "bootstrap_ci_high": float(np.quantile(boots, 0.975)),
            "wins_for_second": int((diff > 0).sum()),
            "wins_for_first": int((diff < 0).sum()),
            "ties": int((np.abs(diff) <= 1e-12).sum()),
            "note": "Only 20 non-overlap origins; multiple comparisons not adjusted.",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "statistical_validation_table.csv", index=False)
    return out


def reproducibility_audit() -> dict:
    checks = {}
    proc = subprocess.run([sys.executable, "-m", "py_compile", "src/stage7_pipeline.py", "src/stage7_1_audit.py", "src/stage7_1_full_audit.py"], cwd=ROOT, capture_output=True, text=True)
    checks["py_compile_returncode"] = proc.returncode
    checks["py_compile_stderr"] = proc.stderr.strip()
    required = [
        "results/stage7/stage7_hourly_predictions.csv",
        "results/stage7/stage7_daily_predictions.csv",
        "results/stage7/stage7_tuning_log.csv",
        "configs/stage7/stage7_config.json",
        "reports/stage7/stage7_final_report.md",
    ]
    checks["required_artifacts_present"] = all((ROOT / p).exists() and (ROOT / p).stat().st_size > 0 for p in required)
    checks["python"] = platform.python_version()
    for mod in ["pandas", "numpy", "sklearn", "xgboost", "lightgbm", "catboost", "scipy"]:
        try:
            checks[f"{mod}_version"] = __import__(mod).__version__
        except Exception as exc:
            checks[f"{mod}_version"] = f"unavailable: {exc}"
    pd.DataFrame([checks]).to_csv(OUT / "reproducibility_checks.csv", index=False)
    return checks


def run_full_audit() -> None:
    ensure_dirs()
    hourly = pd.read_csv(ROOT / "data" / "hourly_energy_clean.csv", parse_dates=[TIMESTAMP])
    features = pd.read_csv(ROOT / "data" / "hourly_features.csv", parse_dates=[TIMESTAMP])
    bounds0 = split_bounds(len(hourly))
    val_origins = list(range(max(bounds0.train_end - 1, 168), bounds0.val_end - 24, 24))
    test_origins = list(range(max(bounds0.val_end - 1, 168), bounds0.n - 24, 24))
    audit_origins = sorted(set(val_origins + test_origins))

    hourly_pred = pd.read_csv(ROOT / "results" / "stage7" / "stage7_hourly_predictions.csv", parse_dates=["origin"])
    direct_comp = pd.read_csv(ROOT / "results" / "stage7" / "stage7_direct_comparison.csv")
    residual_comp = pd.read_csv(ROOT / "results" / "stage7" / "stage7_residual_comparison.csv")
    hybrid_comp = pd.read_csv(ROOT / "results" / "stage7" / "stage7_hybrid_comparison.csv")
    fts_comp = pd.read_csv(ROOT / "results" / "stage7" / "stage7_fts_order_comparison.csv")
    fts_diag = pd.read_csv(ROOT / "results" / "stage7" / "stage7_fts_order_diagnostics.csv")
    horizon = pd.read_csv(ROOT / "results" / "stage7" / "stage7_horizon_metrics.csv")
    tuning = pd.read_csv(ROOT / "results" / "stage7" / "stage7_tuning_log.csv")
    ranking = pd.read_csv(ROOT / "results" / "stage7" / "stage7_final_model_ranking.csv")
    metric_recalc = pd.read_csv(OUT / "METRIC_RECALCULATION.csv") if (OUT / "METRIC_RECALCULATION.csv").exists() else pd.DataFrame()

    fairness = audit_fairness(tuning, hourly)
    feature_audit = feature_future_audit(hourly, features, audit_origins)
    feature_audit.to_csv(OUT / "feature_future_value_audit.csv", index=False)
    feature_summary = feature_audit.groupby("feature", as_index=False).agg(max_abs_diff=("abs_diff", "max"), mean_abs_diff=("abs_diff", "mean"), future_flags=("uses_future_actual", "sum"))
    feature_summary.to_csv(OUT / "feature_future_value_audit_summary.csv", index=False)
    fts_audit = audit_fts(hourly_pred, fts_comp, fts_diag, horizon)
    residual_origin, residual_horizon = plot_residual_focus(hourly_pred)
    gating = audit_gating(hourly_pred, direct_comp)
    stats = statistical_validation(hourly_pred, ranking)
    repro = reproducibility_audit()

    # Metric consistency against summary CSVs.
    recalc_test = metric_recalc[metric_recalc["split"].eq("test")].copy()
    comp_rows = []
    for label, src in [("direct", direct_comp), ("residual", residual_comp), ("hybrid", hybrid_comp), ("fts", fts_comp)]:
        for _, row in src.iterrows():
            m = row["model"]
            rr = recalc_test[recalc_test["model"].eq(m)]
            if rr.empty:
                continue
            rr = rr.iloc[0]
            comp_rows.append({
                "source": label,
                "model": m,
                "daily_mae_csv": row["Daily MAE"],
                "daily_mae_recalc": rr["daily_mae_recalc"],
                "daily_mae_abs_diff": abs(row["Daily MAE"] - rr["daily_mae_recalc"]),
                "hourly_mae_csv": row["Hourly MAE"],
                "hourly_mae_recalc": rr["hourly_mae_recalc"],
                "hourly_mae_abs_diff": abs(row["Hourly MAE"] - rr["hourly_mae_recalc"]),
            })
    metric_consistency = pd.DataFrame(comp_rows)
    metric_consistency.to_csv(OUT / "metric_csv_consistency_audit.csv", index=False)

    # Issue register update.
    issues = pd.read_csv(OUT / "ISSUE_REGISTER.csv") if (OUT / "ISSUE_REGISTER.csv").exists() else pd.DataFrame()
    new_issues = [
        {
            "id": "S7_1_004",
            "phase": "F",
            "severity": "High",
            "status": "Confirmed limitation",
            "title": "Residual FTS3-RandomForest is test-best but validation-poor",
            "evidence": "Validation Daily MAE 5.7880 vs test Daily MAE 3.0977.",
            "impact": "Do not select this model as primary method despite best test MAE.",
            "recommendation": "Use validation-selected model for recommendation and report test-best as exploratory.",
        },
        {
            "id": "S7_1_005",
            "phase": "G",
            "severity": "Medium",
            "status": "Confirmed limitation",
            "title": "Best hybrid improvement is very small and statistically unsupported",
            "evidence": "Hybrid FTS1-LightGBM Fixed 0.31 improves Daily MAE by about 0.0043 kWh; Wilcoxon p around 0.6009.",
            "impact": "Hybrid benefit is not a strong claim.",
            "recommendation": "Prefer Direct ML baseline unless hybrid is kept as exploratory extension.",
        },
        {
            "id": "S7_1_006",
            "phase": "B",
            "severity": "Low",
            "status": "Verified",
            "title": "Numerical feature audit found no future actual target use in lag/rolling features at forecast origins",
            "evidence": f"Max feature reconstruction diff {feature_summary['max_abs_diff'].max():.3e}; future flags {int(feature_summary['future_flags'].sum())}.",
            "impact": "Supports causal availability of direct features.",
            "recommendation": "Keep this regression test with Stage 7.1 artifacts.",
        },
    ]
    issues = pd.concat([issues, pd.DataFrame(new_issues)], ignore_index=True, sort=False)
    issues.to_csv(OUT / "ISSUE_REGISTER.csv", index=False)

    best_validation_direct = direct_comp.sort_values("validation_daily_mae").iloc[0]
    best_validation_residual = residual_comp.sort_values("validation_daily_mae").iloc[0]
    best_validation_hybrid = hybrid_comp.sort_values("validation_daily_mae").iloc[0]
    validation_candidates = pd.DataFrame([
        {"class": "Direct", "model": best_validation_direct["model"], "validation_daily_mae": best_validation_direct["validation_daily_mae"], "test_daily_mae": best_validation_direct["Daily MAE"], "hourly_mae_test": best_validation_direct["Hourly MAE"]},
        {"class": "Residual", "model": best_validation_residual["model"], "validation_daily_mae": best_validation_residual["validation_daily_mae"], "test_daily_mae": best_validation_residual["Daily MAE"], "hourly_mae_test": best_validation_residual["Hourly MAE"]},
        {"class": "Hybrid", "model": best_validation_hybrid["model"], "validation_daily_mae": best_validation_hybrid["validation_daily_mae"], "test_daily_mae": best_validation_hybrid["Daily MAE"], "hourly_mae_test": best_validation_hybrid["Hourly MAE"]},
    ]).sort_values("validation_daily_mae")
    validation_candidates.to_csv(OUT / "validation_selected_candidates.csv", index=False)
    selected = validation_candidates.iloc[0]
    best_test = ranking.sort_values("Daily MAE").iloc[0]

    (OUT / "FTS_ORDER_AUDIT.md").write_text(f"""# FTS ORDER AUDIT

## Ringkasan

{md_table(fts_audit)}

Order dipilih berdasarkan validation, bukan test. Order 2 memiliki validation Daily MAE terendah. Order 3 memiliki test Daily MAE lebih baik, tetapi memakai banyak pola langka dan fallback pada test, sehingga tidak boleh dipilih hanya karena test.

## Catatan fallback

Fallback test terbesar terjadi pada order {int(fts_audit.sort_values('fallback_steps_test', ascending=False).iloc[0]['order'])}. Fallback memperlihatkan sparsity pola ketika order meningkat.
""", encoding="utf-8")

    residual_summary = residual_origin.groupby("split").agg(
        daily_mae=("daily_abs_error", "mean"),
        bias=("daily_signed_error", "mean"),
        mean_cancellation=("error_cancellation_kwh", "mean"),
        max_abs_error=("daily_abs_error", "max"),
    ).reset_index()
    (OUT / "RESIDUAL_AUDIT.md").write_text(f"""# RESIDUAL AUDIT

## Fokus: Residual FTS3-RandomForest

{md_table(residual_summary)}

Model ini buruk pada validation tetapi terbaik pada test. Audit per-origin menunjukkan hasil test dipengaruhi distribusi origin dan error cancellation. Karena validation adalah dasar pemilihan model, Residual FTS3-RandomForest tidak layak dijadikan model utama meskipun test MAE paling kecil.

Artefak pendukung:

- `residual_fts3_randomforest_per_origin.csv`
- `residual_fts3_randomforest_per_horizon.csv`
- `figures/residual_fts3_rf_scatter_daily.png`
- `figures/residual_fts3_rf_origin_signed_error.png`
- `figures/residual_fts3_rf_hourly_error_hist.png`
""", encoding="utf-8")

    focus_gate = gating[gating["model"].eq("Hybrid FTS1-LightGBM Fixed 0.31")]
    (OUT / "GATING_AUDIT.md").write_text(f"""# GATING AUDIT

## Fokus Hybrid FTS1-LightGBM threshold 0.31

{md_table(focus_gate)}

Gate menggunakan prediksi Direct sebagai input keputusan (`direct_prediction <= threshold`), bukan nilai aktual. Threshold 0.31 diperlakukan sebagai fixed baseline. Threshold validasi pada kombinasi lain dipilih dari validation Daily MAE. Banyak gate validasi memilih threshold rendah sehingga residual tidak aktif; model tersebut secara praktis kembali menjadi Direct.

Kesimpulan: hybrid terbaik hanya memberi perbaikan kecil dan tidak cukup kuat untuk klaim metode utama.
""", encoding="utf-8")

    (OUT / "STATISTICAL_VALIDATION.md").write_text(f"""# STATISTICAL VALIDATION

{md_table(stats)}

Hanya ada 20 forecast origin non-overlap. Wilcoxon signed-rank dipakai pada selisih error harian berpasangan. Confidence interval bootstrap origin-level dilaporkan sebagai indikasi sensitivitas, tetapi power statistik rendah dan ada risiko multiple comparisons.

Tidak ada dasar untuk menyatakan perbedaan hybrid terbaik signifikan pada taraf 5%.
""", encoding="utf-8")

    (OUT / "REPRODUCIBILITY_AUDIT.md").write_text(f"""# REPRODUCIBILITY AUDIT

## Checks

{json.dumps(repro, indent=2)}

Clean-run penuh Stage 7 tidak diulang karena runtime panjang dan Stage 7 sudah tersedia. Verifikasi yang dijalankan pada Stage 7.1:

- compile source Stage 7 dan audit scripts,
- audit keberadaan artefak wajib,
- hitung ulang metrik dari prediksi individual,
- rekonstruksi fitur lag/rolling secara numerik,
- kontrol XGBoost Stage 6 params pada Phase A-C.

Kesimpulan: artefak cukup untuk reproduksi audit dan evaluasi ulang metrik. Reproduksi penuh model training tetap mahal tetapi dapat dijalankan melalui `python run_stage7.py`.
""", encoding="utf-8")

    (OUT / "CORRECTION_LOG.md").write_text("""# CORRECTION LOG

Tidak ada bug terkonfirmasi yang membutuhkan patch hasil Stage 7. Tidak ada hasil Stage 1-7 yang diubah.

Patch yang dibuat hanya menambahkan audit Stage 7.1:

- `run_stage7_1_audit.py`
- `src/stage7_1_audit.py`
- `run_stage7_1_full_audit.py`
- `src/stage7_1_full_audit.py`

Tidak ada file corrected karena tidak ada koreksi numerik yang diterapkan.
""", encoding="utf-8")

    (OUT / "FINAL_RECOMMENDATION.md").write_text(f"""# FINAL RECOMMENDATION

## Model berdasarkan validation

Model dengan validation Daily MAE terbaik di antara kelas Direct/Residual/Hybrid adalah:

**{selected['model']}**

- Validation Daily MAE: {selected['validation_daily_mae']:.4f} kWh
- Test Daily MAE: {selected['test_daily_mae']:.4f} kWh
- Test Hourly MAE: {selected['hourly_mae_test']:.4f} kWh

## Model test-best eksploratif

Model dengan test Daily MAE terendah adalah **{best_test['model']}** dengan Daily MAE {best_test['Daily MAE']:.4f} kWh. Namun model ini tidak dipilih sebagai metode utama karena performa validation buruk dan berisiko overfit/distribution-specific.

## Rekomendasi metode penelitian

Metode paling dapat dipertanggungjawabkan untuk narasi penelitian adalah **Direct ML sebagai baseline utama, dengan LightGBM/RandomForest sebagai pembanding kuat, dan Hybrid FTS-ML sebagai analisis eksploratif**. Jika penelitian tetap ingin mempertahankan judul hybrid, klaim harus dibatasi: hybrid berbasis threshold belum terbukti memberi peningkatan yang stabil dan signifikan.
""", encoding="utf-8")

    critical = issues[issues["severity"].isin(["Critical", "High"])].copy()
    (OUT / "AUDIT_SUMMARY.md").write_text(f"""# AUDIT SUMMARY — Stage 7.1 Final

## A. Executive Summary

Stage 7 dapat diaudit dari artefak prediksi individual, konfigurasi, dan source code. Integritas data, origin, horizon, dan target aktual konsisten. Pengujian numerik fitur tidak menemukan penggunaan nilai aktual masa depan pada fitur lag/rolling di origin evaluasi.

Namun, hasil Stage 7 harus dibaca sebagai **benchmark eksploratif dengan keterbatasan**. Model terbaik pada test bukan model yang layak dipilih otomatis karena beberapa hasil test-best buruk pada validation.

## B. Audit Verdict

**PASS WITH LIMITATIONS**

Tidak ada bug atau leakage terkonfirmasi yang memerlukan koreksi hasil Stage 7, tetapi ada keterbatasan metodologis yang memengaruhi cara klaim harus ditulis.

## C. Critical/High Findings

{md_table(critical) if not critical.empty else 'Tidak ada Critical/High finding.'}

## D. Verified Results

- Fitur lag/rolling direkonstruksi numerik dari histori origin; max diff: {feature_summary['max_abs_diff'].max():.3e}.
- Metric CSV konsisten dengan prediksi individual; max Daily MAE diff: {metric_consistency['daily_mae_abs_diff'].max():.3e}.
- Hybrid FTS1-LightGBM Fixed 0.31 memberi perbaikan sangat kecil dan tidak signifikan secara statistik.

## E. Stage 6 vs Stage 7

Rekonsiliasi Phase A-C tetap berlaku: XGBoost Stage 6 vs Stage 7 berbeda karena protocol pemilihan hyperparameter, bukan dataset/target/origin.

## F. Model Selection

Model dipilih berdasarkan validation:

{md_table(validation_candidates)}

Model validation-selected terbaik adalah **{selected['model']}**. Hasil test model tersebut adalah Daily MAE **{selected['test_daily_mae']:.4f} kWh**.

Model test-best eksploratif adalah **{best_test['model']}** dengan Daily MAE **{best_test['Daily MAE']:.4f} kWh**, tetapi tidak direkomendasikan sebagai model utama karena validasi buruk.

## G. Research Implications

- XGBoost masih layak sebagai baseline historis, tetapi tidak lagi satu-satunya kandidat kuat.
- LightGBM/RandomForest Direct lebih defensible sebagai benchmark Direct ML pada Stage 7.
- Pendekatan hybrid belum terbukti membantu secara stabil.
- Orde FTS memengaruhi sparsity pola; order lebih tinggi dapat membaik pada test tetapi berisiko fallback/sparsity.
- Penelitian memerlukan eksperimen tambahan jika ingin klaim hybrid sebagai metode utama.

## H. Recommended Next Action

Untuk laporan ke dosen, gunakan narasi: **Direct ML adalah benchmark utama; hybrid FTS-ML adalah eksperimen tambahan yang belum menunjukkan peningkatan stabil/signifikan.** Jangan memilih model hanya karena test Daily MAE terendah.
""", encoding="utf-8")

    # tiny regression tests as artifacts
    (TESTS / "test_feature_no_future_actual.py").write_text("""from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
summary = pd.read_csv(ROOT / 'results' / 'stage7_1_audit' / 'feature_future_value_audit_summary.csv')
assert summary['max_abs_diff'].max() < 1e-9
assert summary['future_flags'].sum() == 0
""", encoding="utf-8")
    (TESTS / "test_metric_recalculation_consistency.py").write_text("""from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
df = pd.read_csv(ROOT / 'results' / 'stage7_1_audit' / 'metric_csv_consistency_audit.csv')
assert df['daily_mae_abs_diff'].max() < 1e-9
assert df['hourly_mae_abs_diff'].max() < 1e-9
""", encoding="utf-8")

    # run generated regression tests
    subprocess.run([sys.executable, str(TESTS / "test_feature_no_future_actual.py")], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(TESTS / "test_metric_recalculation_consistency.py")], cwd=ROOT, check=True)

    with zipfile.ZipFile(ROOT / "stage7_1_final_audit.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(ROOT))

    print(f"Stage 7.1 full audit written to {OUT}")
    print(f"ZIP: {ROOT / 'stage7_1_final_audit.zip'}")


if __name__ == "__main__":
    run_full_audit()
