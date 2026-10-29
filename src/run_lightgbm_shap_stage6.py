"""Reproduce LightGBM Direct and Residual FTS1 models for SHAP audit."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.stage2_pipeline import TARGET, TIMESTAMP, interval_bounds, split_bounds
from src.stage7_pipeline import (
    FEATURE_COLS,
    Bounds,
    fit_horizon_models,
    fit_residual_models,
    fts_matrix,
    make_origins,
    predict_direct,
    predict_residual,
    rolling_fts_for_residual_training,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "stage6_lightgbm_shap"
FIG = ROOT / "figures" / "stage6_lightgbm_shap"
MODEL_OUT = ROOT / "models" / "stage6_lightgbm_shap"
HOURLY_PRED = ROOT / "results" / "stage7" / "stage7_hourly_predictions.csv"

DIRECT_PARAMS = {"n_estimators": 260, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 20}
RESIDUAL_PARAMS = {"n_estimators": 180, "learning_rate": 0.04, "num_leaves": 15, "min_child_samples": 15}
THRESHOLD = 0.31
ORDER = 1


def expected_value(explainer) -> float:
    return float(np.asarray(explainer.expected_value).reshape(-1)[0])


def compute_shap(models, features, origins, branch: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    checks = []
    X = features.loc[origins, FEATURE_COLS]
    for horizon, model in models.items():
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(X, check_additivity=False))
        prediction = np.asarray(model.predict(X), dtype=float)
        base = expected_value(explainer)
        errors = base + values.sum(axis=1) - prediction
        for row_idx, origin in enumerate(origins):
            for col_idx, feature in enumerate(FEATURE_COLS):
                rows.append({
                    "branch": branch,
                    "horizon": horizon,
                    "origin_index": origin,
                    "feature": feature,
                    "feature_value": float(X.iloc[row_idx, col_idx]),
                    "shap_value": float(values[row_idx, col_idx]),
                    "model_output": float(prediction[row_idx]),
                    "expected_value": base,
                })
        checks.append({
            "branch": branch,
            "horizon": horizon,
            "n_rows": len(X),
            "max_additivity_error": float(np.max(np.abs(errors))),
            "mean_abs_additivity_error": float(np.mean(np.abs(errors))),
        })
    return pd.DataFrame(rows), pd.DataFrame(checks)


def plot_importance(shap_data: pd.DataFrame, branch: str, path: Path) -> None:
    importance = shap_data.assign(abs_shap=shap_data["shap_value"].abs()).groupby("feature")["abs_shap"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(importance.index, importance.values, color="#2f6f9f")
    ax.set_title(f"SHAP Global Importance - {branch}")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Fitur")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_beeswarm(shap_data: pd.DataFrame, branch: str, path: Path) -> None:
    pivot_shap = shap_data.pivot_table(index=["origin_index", "horizon"], columns="feature", values="shap_value").reindex(columns=FEATURE_COLS)
    pivot_feature = shap_data.pivot_table(index=["origin_index", "horizon"], columns="feature", values="feature_value").reindex(columns=FEATURE_COLS)
    shap.summary_plot(pivot_shap.to_numpy(), pivot_feature, feature_names=FEATURE_COLS, show=False, max_display=15, plot_size=(10, 7))
    plt.title(f"SHAP Summary - {branch}", pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    for path in [OUT, FIG, MODEL_OUT]:
        path.mkdir(parents=True, exist_ok=True)

    hourly = pd.read_csv(ROOT / "data" / "hourly_energy_clean.csv", parse_dates=[TIMESTAMP])
    features = pd.read_csv(ROOT / "data" / "hourly_features.csv", parse_dates=[TIMESTAMP])
    split = split_bounds(len(hourly))
    bounds = Bounds(split.train_end, split.val_end, split.n)
    test_origins = make_origins(bounds, "test")
    train_bounds = interval_bounds(hourly.loc[: bounds.train_end - 1, TARGET], "quantile", 15)

    direct_models, direct_train_seconds = fit_horizon_models(features, bounds, "LightGBM", DIRECT_PARAMS, bounds.val_end)
    residual_train = rolling_fts_for_residual_training(hourly, ORDER, train_bounds, bounds.val_end)
    residual_models, residual_train_seconds = fit_residual_models(
        hourly, features, "LightGBM", RESIDUAL_PARAMS, ORDER, train_bounds, bounds.val_end, residual_train
    )
    fts_test, fts_diag = fts_matrix(hourly, test_origins, ORDER, train_bounds, bounds.val_end)

    direct_pred, _ = predict_direct(hourly, features, direct_models, "LightGBM", test_origins, "test", direct_train_seconds, DIRECT_PARAMS)
    residual_pred, _ = predict_residual(hourly, features, residual_models, "LightGBM", ORDER, fts_test, test_origins, "test", residual_train_seconds, RESIDUAL_PARAMS)
    direct_pred = direct_pred.sort_values(["origin_index", "horizon"]).reset_index(drop=True)
    residual_pred = residual_pred.sort_values(["origin_index", "horizon"]).reset_index(drop=True)
    use_residual = direct_pred["predicted"] <= THRESHOLD
    hybrid_pred = direct_pred.copy()
    hybrid_pred["predicted"] = np.where(use_residual, residual_pred["predicted"], direct_pred["predicted"])
    hybrid_pred["selected_expert"] = np.where(use_residual, "Residual FTS1-LightGBM", "LightGBM Direct")
    hybrid_pred["gate_condition"] = use_residual
    hybrid_pred["threshold"] = THRESHOLD

    original = pd.read_csv(HOURLY_PRED)
    original = original[(original["model"] == "Hybrid FTS1-LightGBM Fixed 0.31") & (original["split"] == "test")].sort_values(["origin_index", "horizon"]).reset_index(drop=True)
    comparisons = pd.DataFrame({
        "field": ["direct_prediction", "residual_prediction", "hybrid_prediction"],
        "max_abs_difference": [
            float(np.max(np.abs(direct_pred["predicted"] - original["direct_prediction"]))),
            float(np.max(np.abs(residual_pred["predicted"] - original["residual_prediction"]))),
            float(np.max(np.abs(hybrid_pred["predicted"] - original["predicted"]))),
        ],
    })

    direct_shap, direct_checks = compute_shap(direct_models, features, test_origins, "LightGBM Direct")
    residual_shap, residual_checks = compute_shap(residual_models, features, test_origins, "Residual FTS1-LightGBM")
    all_shap = pd.concat([direct_shap, residual_shap], ignore_index=True)
    all_checks = pd.concat([direct_checks, residual_checks], ignore_index=True)

    all_shap.to_csv(OUT / "lightgbm_shap_values_test.csv", index=False)
    all_checks.to_csv(OUT / "lightgbm_shap_additivity.csv", index=False)
    comparisons.to_csv(OUT / "lightgbm_reproduction_comparison.csv", index=False)
    hybrid_pred.to_csv(OUT / "hybrid_fts1_lightgbm_gate_trace_test.csv", index=False)
    fts_diag.to_csv(OUT / "fts1_test_diagnostics.csv", index=False)
    all_shap.assign(abs_shap=lambda d: d["shap_value"].abs()).groupby(["branch", "feature"], as_index=False)["abs_shap"].mean().rename(columns={"abs_shap": "mean_abs_shap"}).sort_values(["branch", "mean_abs_shap"], ascending=[True, False]).to_csv(OUT / "lightgbm_shap_global_importance.csv", index=False)
    json.dump({"direct_params": DIRECT_PARAMS, "residual_params": RESIDUAL_PARAMS, "threshold": THRESHOLD, "fts_order": ORDER, "test_origins": test_origins, "feature_cols": FEATURE_COLS, "direct_train_seconds": direct_train_seconds, "residual_train_seconds": residual_train_seconds, "specialist_activation_count": int(use_residual.sum()), "specialist_activation_share": float(use_residual.mean())}, open(OUT / "lightgbm_shap_config.json", "w", encoding="utf-8"), indent=2)
    (OUT / "lightgbm_shap_reproduction_report.md").write_text(
        "# Reproduksi SHAP Hybrid FTS1-LightGBM Fixed 0.31\n\n"
        "Analisis ini mereproduksi dua cabang model pada konfigurasi Stage 7: LightGBM Direct dan Residual FTS1-LightGBM. Threshold gate dikunci pada 0,31 kWh dan tidak dituning ulang.\n\n"
        f"- Forecast origin test: {len(test_origins)}\n"
        f"- Baris SHAP per cabang: {len(test_origins) * 24}\n"
        f"- Aktivasi residual: {int(use_residual.sum())} dari {len(use_residual)} ({use_residual.mean() * 100:.2f}%)\n"
        f"- Maksimum selisih reproduksi hybrid terhadap Stage 7: {comparisons['max_abs_difference'].max():.12g}\n"
        f"- Maksimum error additivity Direct: {direct_checks['max_additivity_error'].max():.12g}\n"
        f"- Maksimum error additivity Residual: {residual_checks['max_additivity_error'].max():.12g}\n\n"
        "SHAP residual menjelaskan output korektor residual, bukan nilai FTS awal atau hasil clipping prediksi akhir. Gate dijelaskan terpisah melalui decision trace.\n",
        encoding="utf-8",
    )
    joblib.dump(direct_models, MODEL_OUT / "lightgbm_direct_horizon_models.joblib")
    joblib.dump(residual_models, MODEL_OUT / "lightgbm_residual_fts1_horizon_models.joblib")
    plot_importance(direct_shap, "LightGBM Direct", FIG / "lightgbm_direct_shap_importance.png")
    plot_importance(residual_shap, "Residual FTS1-LightGBM", FIG / "residual_fts1_lightgbm_shap_importance.png")
    plot_beeswarm(direct_shap, "LightGBM Direct", FIG / "lightgbm_direct_shap_summary.png")
    plot_beeswarm(residual_shap, "Residual FTS1-LightGBM", FIG / "residual_fts1_lightgbm_shap_summary.png")
    print(f"Saved SHAP audit to {OUT}")


if __name__ == "__main__":
    main()
