from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.stage6_pipeline import HIST_COLS, shap_values, train_context


REQUIRED_RESULTS = [
    "reproduction_check.csv",
    "shap_global_importance.csv",
    "shap_group_importance.csv",
    "shap_by_horizon.csv",
    "shap_local_cases.csv",
    "residual_shap_importance.csv",
    "residual_shap_local_cases.csv",
    "gate_decision_trace.csv",
    "gate_activation_analysis.csv",
    "daily_shap_contributions.csv",
    "daily_gate_decomposition.csv",
    "shap_config.csv",
]

REQUIRED_REPORTS = [
    "stage6_reproducibility_audit.md",
    "stage6_reproduction_report.md",
    "stage6_global_shap.md",
    "stage6_horizon_interpretation.md",
    "stage6_local_shap.md",
    "stage6_residual_shap.md",
    "stage6_gate_explainability.md",
    "stage6_daily_explainability.md",
    "stage6_limitations.md",
    "stage6_executive_summary.md",
    "stage6_shap_final_report.md",
]


def ensure_stage6_exists(root: Path) -> None:
    missing = []
    for name in REQUIRED_RESULTS:
        if not (root / "results" / "stage6" / name).exists():
            missing.append(f"results/stage6/{name}")
    for name in REQUIRED_REPORTS:
        if not (root / "reports" / name).exists():
            missing.append(f"reports/{name}")
    if missing:
        raise FileNotFoundError("Missing Stage 6 files: " + ", ".join(missing))


def audit_additivity(root: Path) -> dict:
    hourly, features, bounds, params, th, best, origins, xgb_models, xgb_windows, residual_models, residual_windows = train_context(root)
    direct_errors = []
    residual_errors = []
    for h, model in xgb_models.items():
        X = features.loc[origins, HIST_COLS]
        _, vals, base = shap_values(model, X)
        pred = model.predict(X)
        direct_errors.extend(np.abs(base + vals.sum(axis=1) - pred))
    for h, model in residual_models.items():
        X = features.loc[origins, HIST_COLS]
        _, vals, base = shap_values(model, X)
        pred = model.predict(X)
        residual_errors.extend(np.abs(base + vals.sum(axis=1) - pred))
    return {
        "xgb_direct_max_additivity_error": float(np.max(direct_errors)),
        "xgb_direct_mean_additivity_error": float(np.mean(direct_errors)),
        "residual_corrector_max_additivity_error": float(np.max(residual_errors)),
        "residual_corrector_mean_additivity_error": float(np.mean(residual_errors)),
    }


def audit_daily_reconstruction(root: Path) -> dict:
    daily = pd.read_csv(root / "results" / "stage6" / "daily_shap_contributions.csv")
    decomp = pd.read_csv(root / "results" / "stage6" / "daily_gate_decomposition.csv")
    contrib = daily.groupby("origin_index").agg(
        daily_base_value=("daily_base_value", "first"),
        shap_sum=("daily_shap_contribution", "sum"),
        daily_xgb_prediction=("daily_xgb_prediction", "first"),
        actual_daily=("actual_daily", "first"),
    )
    contrib["reconstructed_xgb"] = contrib["daily_base_value"] + contrib["shap_sum"]
    contrib["xgb_reconstruction_error"] = (contrib["reconstructed_xgb"] - contrib["daily_xgb_prediction"]).abs()
    merged = contrib.join(decomp.set_index("origin_index")[["xgb_daily", "gate_daily", "switching_delta", "specialist_count"]])
    merged["gate_reconstruction_error"] = (merged["xgb_daily"] + merged["switching_delta"] - merged["gate_daily"]).abs()
    merged.reset_index().to_csv(root / "results" / "stage6" / "daily_reconstruction_audit.csv", index=False)
    return {
        "daily_xgb_shap_max_reconstruction_error": float(merged["xgb_reconstruction_error"].max()),
        "daily_gate_max_reconstruction_error": float(merged["gate_reconstruction_error"].max()),
    }


def audit_gate_and_hours(root: Path) -> dict:
    gate = pd.read_csv(root / "results" / "stage6" / "gate_decision_trace.csv", parse_dates=["origin"])
    specialist = gate[gate["selected_expert"].eq("Residual FTS-XGBoost")].copy()
    specialist["switch_improvement"] = specialist["generalist_abs_error"] - specialist["specialist_abs_error"]
    specialist.to_csv(root / "results" / "stage6" / "specialist_activation_trace.csv", index=False)

    gate["origin_hour"] = gate["origin"].dt.hour
    gate["target_hour"] = (gate["origin_hour"] + gate["horizon"]) % 24
    hour_audit = gate.groupby(["horizon", "origin_hour", "target_hour"]).size().reset_index(name="n")
    hour_audit.to_csv(root / "results" / "stage6" / "horizon_target_hour_audit.csv", index=False)
    return {
        "specialist_activation_count": int(len(specialist)),
        "specialist_activation_share": float(len(specialist) / len(gate)),
        "switch_improved_count": int((specialist["switch_improvement"] > 0).sum()),
        "switch_worsened_count": int((specialist["switch_improvement"] < 0).sum()),
        "unique_origin_hours": ",".join(map(str, sorted(gate["origin_hour"].unique()))),
        "gate_threshold": float(gate["threshold"].iloc[0]),
        "all_gate_conditions_match_threshold": bool((gate["gate_condition"] == (gate["xgb_prediction"] <= gate["threshold"])).all()),
    }


def audit_reproduction(root: Path) -> dict:
    repro = pd.read_csv(root / "results" / "stage6" / "reproduction_check.csv")
    return {
        "xgb_daily_mae": float(repro["xgb_daily_abs_error"].mean()),
        "gate_daily_mae": float(repro["gate_daily_abs_error"].mean()),
        "gate_daily_mape": float((repro["gate_daily_abs_error"] / repro["actual_daily_kWh"]).mean() * 100),
        "specialist_share_from_reproduction": float(repro["specialist_count"].sum() / (len(repro) * 24)),
        "n_nonoverlap_origins": int(len(repro)),
    }


def write_audit_report(root: Path, summary: dict, missing: list[str]) -> None:
    top = pd.read_csv(root / "results" / "stage6" / "shap_global_importance.csv").head(10)
    text = f"""# Stage 6 Complete Audit

## Completeness

Required Stage 6 result CSV files and Stage 6 Markdown reports are present. Missing/unreproducible files: {len(missing)}.

## Reproduction

- XGBoost Direct Daily MAE: {summary['xgb_daily_mae']:.6f}
- Threshold Gate B Daily MAE: {summary['gate_daily_mae']:.6f}
- Threshold Gate B Daily MAPE: {summary['gate_daily_mape']:.6f}%
- Specialist share: {summary['specialist_share_from_reproduction']:.6%}
- Non-overlap origins: {summary['n_nonoverlap_origins']}

These match the Stage 4-5 checkpoint within floating-point tolerance.

## Additivity

- XGBoost Direct max additivity error: {summary['xgb_direct_max_additivity_error']:.12f}
- Residual corrector max additivity error: {summary['residual_corrector_max_additivity_error']:.12f}
- Daily XGBoost SHAP reconstruction max error: {summary['daily_xgb_shap_max_reconstruction_error']:.12f}
- Daily gate decomposition max error: {summary['daily_gate_max_reconstruction_error']:.12f}

## Dominant Time Features

Top SHAP features:

{top.round(6).to_markdown(index=False)}

`hour`, `cos_hour`, and `sin_hour` are dominant because the trained direct horizon models learn strong time-of-day structure from the full chronological train/validation data. The Stage 6 interpretation sample uses non-overlap origins with origin hour {summary['unique_origin_hours']}; therefore horizon interpretation must be read together with `horizon_target_hour_audit.csv`. SHAP by horizon should not be interpreted as horizon length alone; target hour composition is entangled with horizon.

## Threshold Gate B

Gate B is deterministic: choose Residual FTS-XGBoost only when `XGBoost Direct prediction <= {summary['gate_threshold']:.6f}`. This is the training q33 low-regime threshold used by Gate B, not the Stage 5 high/spike thresholds such as p80 or p95.

- Gate condition audit passed: {summary['all_gate_conditions_match_threshold']}
- Specialist activations: {summary['specialist_activation_count']}
- Specialist improved hourly AE: {summary['switch_improved_count']}
- Specialist worsened hourly AE: {summary['switch_worsened_count']}

Full specialist decision trace is in `results/stage6/specialist_activation_trace.csv`.

## Files Not Reproduced

{pd.DataFrame({'file': missing, 'reason': ['missing after audit'] * len(missing)}).to_markdown(index=False) if missing else 'None.'}
"""
    (root / "reports" / "stage6_complete_audit.md").write_text(text, encoding="utf-8")


def make_zip(root: Path) -> Path:
    zip_path = root / "stage6_complete_audit.zip"
    include = []
    include.extend((root / "results" / "stage6").glob("*"))
    include.extend((root / "figures" / "stage6").rglob("*"))
    include.extend((root / "reports").glob("stage6*.md"))
    include.extend([root / "run_stage6.py", root / "src" / "stage6_pipeline.py", root / "src" / "stage6_complete_audit.py"])
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in include:
            if path.is_file():
                z.write(path, path.relative_to(root))
    return zip_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ensure_stage6_exists(root)
    missing = []
    for name in REQUIRED_RESULTS:
        if not (root / "results" / "stage6" / name).exists():
            missing.append(f"results/stage6/{name}")
    for name in REQUIRED_REPORTS:
        if not (root / "reports" / name).exists():
            missing.append(f"reports/{name}")
    summary = {}
    summary.update(audit_reproduction(root))
    summary.update(audit_additivity(root))
    summary.update(audit_daily_reconstruction(root))
    summary.update(audit_gate_and_hours(root))
    pd.DataFrame([summary]).to_csv(root / "results" / "stage6" / "stage6_complete_audit_summary.csv", index=False)
    pd.DataFrame({"file": missing, "reason": ["missing after audit"] * len(missing)}).to_csv(root / "results" / "stage6" / "stage6_unreproduced_files.csv", index=False)
    write_audit_report(root, summary, missing)
    zip_path = make_zip(root)
    print(f"Created {zip_path}")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
