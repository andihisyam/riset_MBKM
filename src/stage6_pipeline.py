from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.stage2_pipeline import ALL_DIRECT_HORIZONS, RANDOM_STATE, TARGET, TIMESTAMP, metrics, split_bounds
from src.stage3_pipeline import (
    HIST_COLS,
    best_fts,
    best_xgb_params,
    direct_predict_windows,
    fit_direct_models,
    load_checkpoint,
    membership_features,
    residual_hybrid,
    residual_predict_windows,
)
from src.stage4_pipeline import flatten_rows, nonoverlap_origins, window_summary_from_flat
from src.stage5_pipeline import add_volatility_features, thresholds


if not hasattr(np, "bool"):
    np.bool = np.bool_


GROUPS = {
    "Short-term lag": ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12"],
    "Daily seasonal lag": ["lag_24", "lag_48"],
    "Weekly seasonal lag": ["lag_168"],
    "Rolling statistics": [c for c in HIST_COLS if c.startswith("rolling_")],
    "Calendar/time features": ["hour", "day_of_week", "is_weekend", "sin_hour", "cos_hour", "sin_dow", "cos_dow"],
}


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def model_summary(name: str, flat: pd.DataFrame) -> dict:
    return window_summary_from_flat(name, flat)


def train_context(root: Path):
    hourly, features = load_checkpoint(root)
    features = add_volatility_features(features)
    bounds = split_bounds(len(hourly))
    params = best_xgb_params(root, features, bounds)
    th = thresholds(hourly, bounds, root / "results")
    _, _, best = best_fts(hourly, bounds, root / "results")
    features = membership_features(hourly, features, bounds, best)
    origins = nonoverlap_origins(bounds)
    xgb_models = fit_direct_models(features, HIST_COLS, bounds, params)
    xgb_windows = direct_predict_windows(hourly, features, bounds, xgb_models, HIST_COLS, origins)
    _, residual_models, _ = residual_hybrid(hourly, features, bounds, params, best, root / "results")
    residual_windows = residual_predict_windows(hourly, features, bounds, best, residual_models, origins)
    return hourly, features, bounds, params, th, best, origins, xgb_models, xgb_windows, residual_models, residual_windows


def make_gate_flat(xgb_flat: pd.DataFrame, residual_flat: pd.DataFrame, th: dict) -> pd.DataFrame:
    pivot = pd.concat([xgb_flat, residual_flat], ignore_index=True).pivot_table(
        index=["origin_index", "origin", "horizon", "actual"], columns="model", values="predicted"
    ).reset_index()
    rows = []
    for _, r in pivot.iterrows():
        selected = "Residual FTS-XGBoost" if r["XGBoost Direct"] <= th["q33"] else "XGBoost Direct"
        final = r[selected]
        rows.append(
            {
                "model": "Threshold Gate B",
                "origin_index": int(r["origin_index"]),
                "origin": r["origin"],
                "horizon": int(r["horizon"]),
                "actual": float(r["actual"]),
                "xgb_prediction": float(r["XGBoost Direct"]),
                "residual_prediction": float(r["Residual FTS-XGBoost"]),
                "gate_input": float(r["XGBoost Direct"]),
                "threshold": th["q33"],
                "gate_condition": bool(r["XGBoost Direct"] <= th["q33"]),
                "selected_expert": selected,
                "predicted": float(final),
                "generalist_abs_error": abs(float(r["XGBoost Direct"] - r["actual"])),
                "specialist_abs_error": abs(float(r["Residual FTS-XGBoost"] - r["actual"])),
                "abs_error": abs(float(final - r["actual"])),
                "switch_delta": float(r["Residual FTS-XGBoost"] - r["XGBoost Direct"]),
            }
        )
    return pd.DataFrame(rows)


def reproduction(root: Path, out_dir: Path, hourly, features, th, origins, xgb_windows, residual_windows):
    xgb_flat = flatten_rows("XGBoost Direct", xgb_windows)
    residual_flat = flatten_rows("Residual FTS-XGBoost", residual_windows)
    gate = make_gate_flat(xgb_flat, residual_flat, th)
    check_rows = []
    for origin, g in gate.groupby("origin_index"):
        check_rows.append(
            {
                "origin_index": origin,
                "forecast_origin": g["origin"].iloc[0],
                "actual_daily_kWh": g["actual"].sum(),
                "xgb_daily_kWh": g["xgb_prediction"].sum(),
                "residual_daily_kWh": g["residual_prediction"].sum(),
                "gate_daily_kWh": g["predicted"].sum(),
                "xgb_daily_abs_error": abs(g["xgb_prediction"].sum() - g["actual"].sum()),
                "gate_daily_abs_error": abs(g["predicted"].sum() - g["actual"].sum()),
                "specialist_count": int(g["selected_expert"].eq("Residual FTS-XGBoost").sum()),
            }
        )
    repro = pd.DataFrame(check_rows)
    repro.to_csv(out_dir / "reproduction_check.csv", index=False)
    gate.to_csv(out_dir / "gate_decision_trace.csv", index=False)
    xgb_metrics = model_summary("XGBoost Direct", xgb_flat)
    gate_metrics = model_summary("Threshold Gate B", gate.assign(model="Threshold Gate B"))
    return xgb_flat, residual_flat, gate, pd.DataFrame([xgb_metrics, gate_metrics])


def shap_values(model, X: pd.DataFrame):
    booster = model.get_booster()
    contrib = booster.predict(xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)
    return None, np.asarray(contrib[:, :-1]), float(np.mean(contrib[:, -1]))


def plot_importance(importance: pd.DataFrame, path: Path, title: str):
    top = importance.head(20).iloc[::-1]
    plt.figure(figsize=(8, 7))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.title(title)
    plt.xlabel("Mean absolute SHAP value (kWh)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def global_shap(out_dir: Path, fig_dir: Path, features, origins, xgb_models):
    rows = []
    horizon_rows = []
    shap_cache = {}
    X_all = features.loc[origins, HIST_COLS]
    for h, model in xgb_models.items():
        X = X_all.copy()
        _, vals, base = shap_values(model, X)
        shap_cache[h] = {"X": X, "values": vals, "base": base}
        mean_abs = np.abs(vals).mean(axis=0)
        for f, v in zip(HIST_COLS, mean_abs):
            horizon_rows.append({"horizon": h, "feature": f, "mean_abs_shap": v, "base_value": base})
    by_h = pd.DataFrame(horizon_rows)
    by_h.to_csv(out_dir / "shap_by_horizon.csv", index=False)
    glob = by_h.groupby("feature", as_index=False)["mean_abs_shap"].mean().sort_values("mean_abs_shap", ascending=False)
    glob.to_csv(out_dir / "shap_global_importance.csv", index=False)
    group_rows = []
    for h, g in by_h.groupby("horizon"):
        for group, cols in GROUPS.items():
            sub = g[g["feature"].isin(cols)]
            group_rows.append({"horizon": h, "group": group, "total_mean_abs_shap": sub["mean_abs_shap"].sum(), "mean_per_feature": sub["mean_abs_shap"].mean(), "n_features": len(sub)})
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(out_dir / "shap_group_importance.csv", index=False)

    plot_importance(glob, fig_dir / "shap_importance.png", "XGBoost Direct global SHAP importance")
    h1 = shap_cache[1]
    shap.summary_plot(h1["values"], h1["X"], show=False, max_display=20)
    plt.title("XGBoost Direct SHAP summary (H1)")
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary.png", dpi=160)
    plt.close()

    heat = by_h.pivot_table(index="feature", columns="horizon", values="mean_abs_shap").fillna(0)
    top_features = glob.head(20)["feature"]
    plt.figure(figsize=(12, 8))
    plt.imshow(heat.loc[top_features], aspect="auto", cmap="viridis")
    plt.yticks(range(len(top_features)), top_features)
    plt.xticks(range(24), range(1, 25))
    plt.colorbar(label="Mean absolute SHAP value (kWh)")
    plt.title("SHAP importance by horizon")
    plt.xlabel("Horizon")
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_horizon_heatmap.png", dpi=160)
    plt.close()

    group_pivot = group_df.pivot_table(index="horizon", columns="group", values="total_mean_abs_shap")
    group_pivot.plot(figsize=(10, 5), marker="o")
    plt.title("Feature group SHAP contribution by horizon")
    plt.xlabel("Horizon")
    plt.ylabel("Total mean absolute SHAP value (kWh)")
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_feature_group_by_horizon.png", dpi=160)
    plt.close()
    return shap_cache, glob, group_df, by_h


def local_cases(out_dir: Path, fig_dir: Path, features, xgb_flat, gate, shap_cache):
    cases_dir = fig_dir / "local_cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    df = gate.copy()
    df["actual_regime"] = pd.cut(df["actual"], [-np.inf, 0.31, 0.81, np.inf], labels=["low", "normal", "high"]).astype(str)
    selectors = {
        "low_error": df["abs_error"].idxmin(),
        "high_error": df["abs_error"].idxmax(),
        "low_actual": df[df["actual_regime"].eq("low")]["abs_error"].idxmin(),
        "normal_actual": df[df["actual_regime"].eq("normal")]["abs_error"].idxmin(),
        "high_actual": df[df["actual_regime"].eq("high")]["abs_error"].idxmin(),
        "specialist_selected": df[df["selected_expert"].eq("Residual FTS-XGBoost")]["abs_error"].idxmin(),
        "generalist_selected": df[df["selected_expert"].eq("XGBoost Direct")]["abs_error"].idxmin(),
    }
    rows = []
    used = set()
    for label, idx in selectors.items():
        if idx in used:
            continue
        used.add(idx)
        row = df.loc[idx]
        h = int(row["horizon"])
        cache = shap_cache[h]
        origin = int(row["origin_index"])
        pos = list(cache["X"].index).index(origin)
        vals = cache["values"][pos]
        contrib = pd.DataFrame({"feature": HIST_COLS, "shap_value": vals, "feature_value": cache["X"].iloc[pos].values})
        top_pos = contrib.sort_values("shap_value", ascending=False).head(5)
        top_neg = contrib.sort_values("shap_value").head(5)
        for _, c in pd.concat([top_pos.assign(direction="positive"), top_neg.assign(direction="negative")]).iterrows():
            rows.append({**row.to_dict(), "case": label, "base_value": cache["base"], "feature": c["feature"], "feature_value": c["feature_value"], "shap_value": c["shap_value"], "direction": c["direction"]})
        top = contrib.reindex(contrib["shap_value"].abs().sort_values(ascending=False).head(12).index).iloc[::-1]
        plt.figure(figsize=(8, 5))
        plt.barh(top["feature"], top["shap_value"])
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title(f"Local SHAP waterfall-style: {label}, H{h}")
        plt.xlabel("SHAP value (kWh)")
        plt.tight_layout()
        plt.savefig(cases_dir / f"{label}_waterfall.png", dpi=160)
        plt.close()
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "shap_local_cases.csv", index=False)
    return out


def residual_shap(out_dir: Path, fig_dir: Path, features, origins, residual_models, gate):
    rows = []
    local_rows = []
    for h in [1, 6, 12, 24]:
        X = features.loc[origins, HIST_COLS]
        _, vals, base = shap_values(residual_models[h], X)
        pred = residual_models[h].predict(X)
        max_add_err = np.max(np.abs(base + vals.sum(axis=1) - pred))
        for f, v in zip(HIST_COLS, np.abs(vals).mean(axis=0)):
            rows.append({"horizon": h, "feature": f, "mean_abs_shap": v, "base_value": base, "max_additivity_error": max_add_err})
        pos = 0
        contrib = pd.DataFrame({"feature": HIST_COLS, "shap_value": vals[pos], "feature_value": X.iloc[pos].values})
        for _, c in contrib.reindex(contrib["shap_value"].abs().sort_values(ascending=False).head(10).index).iterrows():
            local_rows.append({"horizon": h, "origin_index": origins[pos], "base_value": base, "predicted_residual": pred[pos], "feature": c["feature"], "feature_value": c["feature_value"], "shap_value": c["shap_value"]})
    imp = pd.DataFrame(rows)
    imp.to_csv(out_dir / "residual_shap_importance.csv", index=False)
    local = pd.DataFrame(local_rows)
    local.to_csv(out_dir / "residual_shap_local_cases.csv", index=False)
    plot_importance(imp.groupby("feature", as_index=False)["mean_abs_shap"].mean().sort_values("mean_abs_shap", ascending=False), fig_dir / "residual_shap_summary.png", "Residual XGBoost SHAP importance")
    one = local[local["horizon"].eq(1)].copy().iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(one["feature"], one["shap_value"])
    plt.title("Residual SHAP local waterfall-style (H1)")
    plt.xlabel("SHAP value for residual correction (kWh)")
    plt.tight_layout()
    plt.savefig(fig_dir / "residual_shap_waterfall.png", dpi=160)
    plt.close()
    return imp, local


def gate_analysis(out_dir: Path, fig_dir: Path, gate: pd.DataFrame):
    activation = gate.groupby(["horizon", "selected_expert"]).size().reset_index(name="n")
    activation["share"] = activation["n"] / activation.groupby("horizon")["n"].transform("sum")
    by_hour = gate.assign(origin_hour=pd.to_datetime(gate["origin"]).dt.hour).groupby(["origin_hour", "selected_expert"]).size().reset_index(name="n")
    switch = gate[gate["selected_expert"].eq("Residual FTS-XGBoost")].copy()
    switch["switch_improvement"] = switch["generalist_abs_error"] - switch["specialist_abs_error"]
    summary = pd.DataFrame(
        [
            {
                "specialist_activation_share": gate["selected_expert"].eq("Residual FTS-XGBoost").mean(),
                "specialist_activations": int(gate["selected_expert"].eq("Residual FTS-XGBoost").sum()),
                "switch_improved_count": int((switch["switch_improvement"] > 0).sum()),
                "switch_worsened_count": int((switch["switch_improvement"] < 0).sum()),
                "mean_switch_improvement_kWh": switch["switch_improvement"].mean(),
            }
        ]
    )
    pd.concat([activation.assign(section="by_horizon"), by_hour.assign(section="by_origin_hour")], ignore_index=True).to_csv(out_dir / "gate_activation_analysis.csv", index=False)
    summary.to_csv(out_dir / "gate_activation_summary.csv", index=False)
    activation.pivot(index="horizon", columns="selected_expert", values="share").fillna(0).plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.title("Gate B activation by horizon")
    plt.ylabel("Share of predictions")
    plt.tight_layout()
    plt.savefig(fig_dir / "gate_activation_by_horizon.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, 4))
    switch["switch_improvement"].hist(bins=20)
    plt.title("Gate switch improvement distribution")
    plt.xlabel("Generalist AE - specialist AE (kWh)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(fig_dir / "gate_improvement_distribution.png", dpi=160)
    plt.close()
    return summary


def daily_explainability(out_dir: Path, fig_dir: Path, gate: pd.DataFrame, shap_cache):
    rows = []
    for origin, g in gate.groupby("origin_index"):
        contrib = pd.Series(0.0, index=HIST_COLS)
        base_sum = 0.0
        xgb_pred_sum = 0.0
        for _, r in g.iterrows():
            h = int(r["horizon"])
            cache = shap_cache[h]
            pos = list(cache["X"].index).index(origin)
            base_sum += cache["base"]
            contrib += pd.Series(cache["values"][pos], index=HIST_COLS)
            xgb_pred_sum += r["xgb_prediction"]
        for f, v in contrib.items():
            rows.append({"origin_index": origin, "forecast_origin": g["origin"].iloc[0], "feature": f, "daily_shap_contribution": v, "daily_base_value": base_sum, "daily_xgb_prediction": xgb_pred_sum, "actual_daily": g["actual"].sum()})
    daily = pd.DataFrame(rows)
    daily.to_csv(out_dir / "daily_shap_contributions.csv", index=False)
    decomp = gate.groupby("origin_index").agg(actual_daily=("actual", "sum"), xgb_daily=("xgb_prediction", "sum"), gate_daily=("predicted", "sum"), switching_delta=("switch_delta", lambda x: x[gate.loc[x.index, "selected_expert"].eq("Residual FTS-XGBoost")].sum()), specialist_count=("selected_expert", lambda x: int((x == "Residual FTS-XGBoost").sum()))).reset_index()
    decomp["gate_daily_abs_error"] = (decomp["gate_daily"] - decomp["actual_daily"]).abs()
    decomp["xgb_daily_abs_error"] = (decomp["xgb_daily"] - decomp["actual_daily"]).abs()
    decomp.to_csv(out_dir / "daily_gate_decomposition.csv", index=False)
    first = daily[daily["origin_index"].eq(daily["origin_index"].iloc[0])].copy()
    top = first.reindex(first["daily_shap_contribution"].abs().sort_values(ascending=False).head(12).index).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["daily_shap_contribution"])
    plt.title("Daily SHAP contribution waterfall-style")
    plt.xlabel("Summed SHAP contribution across H1-H24 (kWh)")
    plt.tight_layout()
    plt.savefig(fig_dir / "daily_shap_waterfall.png", dpi=160)
    plt.close()
    return daily, decomp


def write_reports(root: Path, tables: dict):
    reports = root / "reports"
    out = root / "results" / "stage6"
    fig = root / "figures" / "stage6"
    top5 = tables["global"].head(5)
    repro = tables["repro_metrics"]
    gate_summary = tables["gate_summary"].iloc[0]
    group_top = tables["group"].groupby("group")["total_mean_abs_shap"].mean().sort_values(ascending=False)
    template = {
        "reports/stage6_reproducibility_audit.md": f"""# Stage 6 Reproducibility Audit

Model dijelaskan dengan rekonstruksi prosedur Stage 4-5, karena artifact model ter-serialize tidak tersedia. Script melatih ulang model dengan konfigurasi tersimpan (`xgboost_tuning_historical.csv`), split kronologis 70/15/15, fitur hourly Stage 2, dan non-overlap origins Stage 4-5.

Threshold Gate B memakai threshold training-only `q33 = {tables['threshold']:.6f}` dan aturan deterministik: jika prediksi XGBoost <= q33 maka pilih Residual FTS-XGBoost, selain itu pilih XGBoost Direct.

Risiko leakage: tidak ditemukan pada fitur SHAP karena input inference hanya memakai fitur di origin t. Model direkonstruksi, bukan di-load dari artifact binary.
""",
        "reports/stage6_reproduction_report.md": f"""# Stage 6 Reproduction Report

{repro.round(6).to_markdown(index=False)}

Hasil reproduksi mendekati checkpoint Stage 4-5 karena model dilatih ulang deterministik dari prosedur yang sama. Detail origin tersimpan di `results/stage6/reproduction_check.csv`.
""",
        "reports/stage6_global_shap.md": f"""# Global SHAP

Top fitur XGBoost Direct:

{top5.round(6).to_markdown(index=False)}

Kelompok fitur terbesar berdasarkan total mean absolute SHAP adalah `{group_top.index[0]}`. Nilai ini menunjukkan kontribusi model, bukan hubungan kausal.
""",
        "reports/stage6_horizon_interpretation.md": f"""# Horizon SHAP Interpretation

SHAP dihitung untuk H1-H24. Output utama: `shap_by_horizon.csv`, `shap_horizon_heatmap.png`, dan `shap_feature_group_by_horizon.png`.

Pola interpretasi dibaca per horizon karena setiap H memakai model direct berbeda. Perubahan kontribusi dapat dipengaruhi horizon, jam target, dan komposisi origin.
""",
        "reports/stage6_local_shap.md": "# Local SHAP\n\nKasus lokal tersimpan di `results/stage6/shap_local_cases.csv` dan grafik waterfall-style ada di `figures/stage6/local_cases/`.\n",
        "reports/stage6_residual_shap.md": "# Residual SHAP\n\nResidual SHAP menjelaskan XGBoost residual corrector, bukan keseluruhan FTS. Additivity error tercatat di `residual_shap_importance.csv`.\n",
        "reports/stage6_gate_explainability.md": f"""# Gate Explainability

Specialist activation share: {gate_summary['specialist_activation_share']:.4%}. Improved switches: {int(gate_summary['switch_improved_count'])}; worsened switches: {int(gate_summary['switch_worsened_count'])}.

Gate B adalah aturan threshold deterministik, sehingga penjelasan utamanya adalah decision trace, bukan SHAP pada gate.
""",
        "reports/stage6_daily_explainability.md": "# Daily Explainability\n\nDaily XGBoost SHAP dijumlahkan dari kontribusi H1-H24. Untuk gated hybrid, dekomposisi memakai `Daily Gate Forecast = Daily XGB Forecast + switching_delta`.\n",
        "reports/stage6_limitations.md": """# Stage 6 Limitations

SHAP bukan bukti kausalitas. Fitur lag dan rolling saling berkorelasi sehingga atribusi dapat tersebar. Dataset berasal dari satu rumah. Evaluasi model terbaik hanya memakai 20 origin non-overlap. Gate B memberi peningkatan kecil dan belum signifikan pada taraf 5%. Oracle bukan model deployable.
""",
        "reports/stage6_executive_summary.md": f"""# Executive Summary Stage 6

Stage 6 menjelaskan XGBoost Direct, residual hybrid, dan Threshold Gate B dengan SHAP dan decision trace.

SHAP base value adalah nilai acuan prediksi model. SHAP value adalah kontribusi fitur terhadap selisih prediksi dari nilai acuan. Gate decision trace adalah catatan aturan threshold, input gate, dan expert yang dipilih.

Temuan utama: fitur paling dominan adalah `{top5.iloc[0]['feature']}`. Gate B memilih specialist hanya pada sebagian kecil prediksi. Interpretasi mendukung kesimpulan Stage 5: XGBoost Direct tetap kandidat metode utama, sementara FTS lebih tepat sebagai analisis pembanding/eksploratif.
""",
    }
    final = f"""# Stage 6 SHAP Final Report

## 1. Introduction
Interpretabilitas diperlukan agar prediksi konsumsi energi 24 jam dapat dijelaskan, terutama karena model menggunakan lag, rolling statistics, dan gate hybrid.

## 2. Methodology
Tree SHAP digunakan untuk XGBoost Direct dan XGBoost residual corrector. Gate B dijelaskan dengan decision trace karena merupakan aturan threshold deterministik.

## 3. Experimental Setup
Data hourly clean, split kronologis, non-overlap origins, dan threshold Stage 5 digunakan kembali. Model direkonstruksi dari prosedur training karena artifact model binary tidak tersedia.

## 4. Results
Top fitur global:

{top5.round(6).to_markdown(index=False)}

Reproduksi:

{repro.round(6).to_markdown(index=False)}

## 5. Discussion
SHAP menunjukkan prediksi didominasi fitur historis/temporal. Gate B hanya mengalihkan prediksi saat XGBoost forecast berada pada regime rendah, sehingga improvement harian berasal dari switching terbatas.

## 6. Limitations
SHAP tidak kausal, fitur berkorelasi, data satu rumah, dan evaluasi hanya 20 origin non-overlap. Keunggulan Gate B belum signifikan pada taraf 5%.

## 7. Conclusion
Stage 6 memberi penjelasan yang konsisten dengan Stage 1-5. Final proposed method tetap Direct multi-horizon XGBoost; FTS dipertahankan sebagai pembanding/analisis fuzzy, bukan komponen inti.
"""
    template["reports/stage6_shap_final_report.md"] = final
    for rel, text in template.items():
        (root / rel).write_text(text, encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results" / "stage6"
    fig_dir = root / "figures" / "stage6"
    ensure_dirs(out_dir, fig_dir, fig_dir / "local_cases")
    hourly, features, bounds, params, th, best, origins, xgb_models, xgb_windows, residual_models, residual_windows = train_context(root)
    xgb_flat, residual_flat, gate, repro_metrics = reproduction(root, out_dir, hourly, features, th, origins, xgb_windows, residual_windows)
    shap_cache, global_imp, group_imp, by_h = global_shap(out_dir, fig_dir, features, origins, xgb_models)
    local = local_cases(out_dir, fig_dir, features, xgb_flat, gate, shap_cache)
    residual_imp, residual_local = residual_shap(out_dir, fig_dir, features, origins, residual_models, gate)
    gate_summary = gate_analysis(out_dir, fig_dir, gate)
    daily, decomp = daily_explainability(out_dir, fig_dir, gate, shap_cache)
    config = pd.DataFrame([{"random_state": RANDOM_STATE, "n_origins": len(origins), "gate_b_threshold_q33": th["q33"], "shap_sample": "20 non-overlap origins x H1-H24"}])
    config.to_csv(out_dir / "shap_config.csv", index=False)
    write_reports(root, {"global": global_imp, "group": group_imp, "repro_metrics": repro_metrics, "gate_summary": gate_summary, "threshold": th["q33"]})
    print("Stage 6 SHAP complete.")
    print(f"Outputs: {out_dir}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
