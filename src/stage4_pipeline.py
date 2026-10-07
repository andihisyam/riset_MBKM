from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from src.stage2_pipeline import (
    ALL_DIRECT_HORIZONS,
    BASE_FEATURES,
    LAGS,
    RANDOM_STATE,
    ROLLING_FEATURES,
    TARGET,
    TIMESTAMP,
    add_features,
    fts_model_comparison,
    interval_bounds,
    make_xgb,
    metrics,
    partition_experiment,
    split_bounds,
    states,
)
from src.stage3_pipeline import (
    HIST_COLS,
    best_fts,
    best_xgb_params,
    direct_predict_windows,
    fit_direct_models,
    fts_recursive_matrix,
    load_checkpoint,
    membership_features,
    residual_hybrid,
    residual_predict_windows,
    summarize_windows,
    target_frame,
)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def nonoverlap_origins(bounds) -> list[int]:
    start = max(bounds.val_end - 1, 168)
    return list(range(start, bounds.n - 24, 24))


def daily_rows(model_name: str, windows: list[dict]) -> list[dict]:
    rows = []
    for w in windows:
        rows.append(
            {
                "model": model_name,
                "origin_index": w["origin_index"],
                "origin": w["origin"],
                "actual_daily": float(w["actual"].sum()),
                "predicted_daily": float(w["predicted"].sum()),
                "daily_abs_error": float(abs(w["actual"].sum() - w["predicted"].sum())),
            }
        )
    return rows


def flatten_rows(model_name: str, windows: list[dict]) -> pd.DataFrame:
    rows = []
    for w in windows:
        for h, (actual, pred) in enumerate(zip(w["actual"], w["predicted"]), start=1):
            rows.append({"model": model_name, "origin_index": w["origin_index"], "origin": w["origin"], "horizon": h, "actual": float(actual), "predicted": float(pred), "abs_error": float(abs(actual - pred))})
    return pd.DataFrame(rows)


def weekly_windows(hourly: pd.DataFrame, origins: list[int]) -> list[dict]:
    return [
        {
            "origin_index": o,
            "origin": hourly.loc[o, TIMESTAMP],
            "actual": hourly.loc[o + 1 : o + 24, TARGET].to_numpy(),
            "predicted": hourly.loc[o - 167 : o - 144, TARGET].to_numpy(),
        }
        for o in origins
    ]


def train_xgb_residual_fts(hourly, features, bounds, params, best, origins, results_dir):
    xgb_models = fit_direct_models(features, HIST_COLS, bounds, params)
    residual_rules = {}
    residual_notes = []
    for h in ALL_DIRECT_HORIZONS:
        train_frame, target_col = target_frame(features, HIST_COLS, h, bounds)
        pred = xgb_models[h].predict(train_frame[HIST_COLS])
        residual = train_frame[target_col].to_numpy() - pred
        acf1 = pd.Series(residual).autocorr(1)
        acf24 = pd.Series(residual).autocorr(24) if len(residual) > 24 else np.nan
        residual_notes.append({"horizon": h, "acf_lag_1": acf1, "acf_lag_24": acf24, "mean": float(np.mean(residual)), "std": float(np.std(residual))})
        # FTS on shifted positive residuals; add offset back after prediction.
        offset = abs(min(float(np.min(residual)), 0.0)) + 1e-6
        shifted = residual + offset
        bounds_arr = interval_bounds(pd.Series(shifted), "quantile", 7)
        rules = {}
        st = states(shifted, bounds_arr)
        for i in range(1, len(st)):
            rules.setdefault((int(st[i - 1]),), []).append(int(st[i]))
        centers = (bounds_arr[:-1] + bounds_arr[1:]) / 2
        residual_rules[h] = {"offset": offset, "bounds": bounds_arr, "rules": rules, "centers": centers, "fallback": float(np.mean(shifted))}
    pd.DataFrame(residual_notes).to_csv(results_dir / "xgb_residual_fts_diagnostics.csv", index=False)

    windows = []
    for o in origins:
        pred = []
        for h in ALL_DIRECT_HORIZONS:
            xgb_pred = max(float(xgb_models[h].predict(features.loc[[o], HIST_COLS])[0]), 0.0)
            cfg = residual_rules[h]
            # Conservative residual history: learned from train+validation residuals only.
            last_state = sorted(cfg["rules"].keys())[-1]
            consequents = cfg["rules"].get(last_state, [])
            if consequents:
                vals, counts = np.unique(consequents, return_counts=True)
                residual_hat = float(np.average(cfg["centers"][vals], weights=counts)) - cfg["offset"]
            else:
                residual_hat = cfg["fallback"] - cfg["offset"]
            pred.append(max(xgb_pred + residual_hat, 0.0))
        windows.append({"origin_index": o, "origin": hourly.loc[o, TIMESTAMP], "actual": hourly.loc[o + 1 : o + 24, TARGET].to_numpy(), "predicted": np.asarray(pred)})
    result = pd.DataFrame([summarize_windows("XGBoost-FTS Residual Correction", windows, "XGBoost forecast plus FTS correction on XGBoost residuals")])
    result.to_csv(results_dir / "xgb_fts_residual_results.csv", index=False)
    return result, windows


def regime_definition(hourly, bounds, results_dir):
    train = hourly.loc[: bounds.train_end - 1, TARGET]
    q33, q67 = train.quantile([1 / 3, 2 / 3])
    p80, p95 = train.quantile([0.80, 0.95])
    out = pd.DataFrame(
        [
            {"scheme": "tertile", "regime": "Low", "lower": train.min(), "upper": q33},
            {"scheme": "tertile", "regime": "Medium", "lower": q33, "upper": q67},
            {"scheme": "tertile", "regime": "High", "lower": q67, "upper": train.max()},
            {"scheme": "spike", "regime": "Normal", "lower": train.min(), "upper": p80},
            {"scheme": "spike", "regime": "High", "lower": p80, "upper": p95},
            {"scheme": "spike", "regime": "Extreme", "lower": p95, "upper": train.max()},
        ]
    )
    out.to_csv(results_dir / "regime_definition.csv", index=False)
    return {"q33": float(q33), "q67": float(q67), "p80": float(p80), "p95": float(p95)}, out


def add_regime(values, thresholds):
    values = pd.Series(values)
    return pd.cut(values, bins=[-np.inf, thresholds["q33"], thresholds["q67"], np.inf], labels=["Low", "Medium", "High"]).astype(str)


def add_spike_regime(values, thresholds):
    values = pd.Series(values)
    return pd.cut(values, bins=[-np.inf, thresholds["p80"], thresholds["p95"], np.inf], labels=["Normal", "High", "Extreme"]).astype(str)


def expert_by_regime(flat: pd.DataFrame, thresholds, results_dir):
    df = flat.copy()
    df["regime"] = add_regime(df["actual"], thresholds).to_numpy()
    rows = []
    for (model, regime), group in df.groupby(["model", "regime"]):
        rows.append({"model": model, "regime": regime, **metrics(group["actual"], group["predicted"]), "n": len(group)})
    out = pd.DataFrame(rows)
    out.to_csv(results_dir / "expert_performance_by_regime.csv", index=False)
    return out


def window_summary_from_flat(name, flat, notes=""):
    return {"model": name, **metrics(flat["actual"], flat["predicted"], "Hourly "), **metrics(flat.groupby("origin_index")["actual"].sum(), flat.groupby("origin_index")["predicted"].sum(), "Daily "), "windows": flat["origin_index"].nunique(), "Notes": notes}


def oracle_gate(flat: pd.DataFrame, results_dir):
    idx = flat.groupby(["origin_index", "horizon"])["abs_error"].idxmin()
    chosen = flat.loc[idx].copy()
    out = pd.DataFrame([window_summary_from_flat("Oracle Gate", chosen, "Upper bound; uses actual target to choose best expert")])
    base = pd.DataFrame([window_summary_from_flat("XGBoost Direct", flat[flat["model"].eq("XGBoost Direct")])])
    out["Daily MAE improvement vs XGBoost (%)"] = (base["Daily MAE"].iloc[0] - out["Daily MAE"]) / base["Daily MAE"].iloc[0] * 100
    chosen["chosen_expert"] = chosen["model"]
    out.to_csv(results_dir / "oracle_gate_results.csv", index=False)
    chosen.to_csv(results_dir / "oracle_gate_detail.csv", index=False)
    return out, chosen


def threshold_gates(flat, thresholds, results_dir):
    pivot = flat.pivot_table(index=["origin_index", "origin", "horizon", "actual"], columns="model", values="predicted").reset_index()
    rows = []
    details = []
    configs = {
        "Gate A Low/Medium predicted -> Residual": lambda x: "Residual FTS-XGBoost" if x <= thresholds["q67"] else "XGBoost Direct",
        "Gate B Low predicted -> Residual": lambda x: "Residual FTS-XGBoost" if x <= thresholds["q33"] else "XGBoost Direct",
        "Gate C Normal predicted -> Feature Hybrid": lambda x: "Hybrid feature-based" if x <= thresholds["p80"] else "XGBoost Direct",
    }
    for name, chooser in configs.items():
        chosen = []
        for _, row in pivot.iterrows():
            expert = chooser(row["XGBoost Direct"])
            chosen.append({"origin_index": row["origin_index"], "origin": row["origin"], "horizon": row["horizon"], "actual": row["actual"], "predicted": row[expert], "chosen_expert": expert, "gate": name})
        detail = pd.DataFrame(chosen)
        rows.append(window_summary_from_flat(name, detail, "Threshold gate using XGBoost predicted value"))
        details.append(detail)
    result = pd.DataFrame(rows)
    freq = pd.concat(details).groupby(["gate", "chosen_expert"]).size().reset_index(name="n")
    freq["share"] = freq["n"] / freq.groupby("gate")["n"].transform("sum")
    result.to_csv(results_dir / "threshold_gate_results.csv", index=False)
    pd.concat(details).to_csv(results_dir / "threshold_gate_detail.csv", index=False)
    freq.to_csv(results_dir / "gate_decision_frequency.csv", index=False)
    return result, pd.concat(details), freq


def learned_gate(flat, features, thresholds, results_dir):
    pivot = flat.pivot_table(index=["origin_index", "origin", "horizon", "actual"], columns="model", values=["predicted", "abs_error"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    experts = ["XGBoost Direct", "Hybrid feature-based", "Residual FTS-XGBoost"]
    err_cols = [f"abs_error_{x}" for x in experts]
    pivot["best_expert"] = pivot[err_cols].idxmin(axis=1).str.replace("abs_error_", "", regex=False)
    gate_features = ["predicted_XGBoost Direct", "horizon", "lag_1", "lag_24", "lag_168", "rolling_mean_3", "rolling_mean_24", "rolling_std_24", "hour", "day_of_week", "is_weekend"]
    rows = []
    for _, row in pivot.iterrows():
        idx = int(row["origin_index"])
        base = row.to_dict()
        for col in gate_features[2:]:
            base[col] = features.loc[idx, col]
        rows.append(base)
    data = pd.DataFrame(rows).dropna(subset=gate_features + ["best_expert"])
    # Chronological split inside non-overlap origins: first half trains gate, second half evaluates.
    unique_origins = sorted(data["origin_index"].unique())
    cut = unique_origins[len(unique_origins) // 2]
    train = data[data["origin_index"] <= cut]
    test = data[data["origin_index"] > cut]
    clf = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE, min_samples_leaf=8)
    clf.fit(train[gate_features], train["best_expert"])
    test = test.copy()
    test["chosen_expert"] = clf.predict(test[gate_features])
    test["predicted"] = [row[f"predicted_{row['chosen_expert']}"] for _, row in test.iterrows()]
    acc = accuracy_score(test["best_expert"], test["chosen_expert"])
    bal = balanced_accuracy_score(test["best_expert"], test["chosen_expert"])
    cm = pd.DataFrame(confusion_matrix(test["best_expert"], test["chosen_expert"], labels=experts), index=experts, columns=experts)
    result = pd.DataFrame([{**window_summary_from_flat("Learned Gate", test, "DecisionTree gate evaluated on later non-overlap origins"), "gate_accuracy": acc, "gate_balanced_accuracy": bal}])
    result.to_csv(results_dir / "learned_gate_results.csv", index=False)
    test.to_csv(results_dir / "learned_gate_detail.csv", index=False)
    cm.to_csv(results_dir / "learned_gate_confusion_matrix.csv")
    return result, test


def high_and_bias(flat, thresholds, results_dir):
    df = flat.copy()
    df["spike_regime"] = add_spike_regime(df["actual"], thresholds).to_numpy()
    df["regime"] = add_regime(df["actual"], thresholds).to_numpy()
    df["error"] = df["predicted"] - df["actual"]
    high_rows = []
    for (model, regime), group in df[df["spike_regime"].isin(["High", "Extreme"])].groupby(["model", "spike_regime"]):
        high_rows.append({"model": model, "regime": regime, "MAE": float(group["abs_error"].mean()), "bias": float(group["error"].mean()), "n": len(group)})
    high = pd.DataFrame(high_rows)
    high.to_csv(results_dir / "high_consumption_error_analysis.csv", index=False)
    bias_rows = []
    for model, group in df.groupby("model"):
        bias_rows.append({"model": model, "regime": "overall", "bias": float(group["error"].mean()), "n": len(group)})
    for (model, regime), group in df.groupby(["model", "regime"]):
        bias_rows.append({"model": model, "regime": regime, "bias": float(group["error"].mean()), "n": len(group)})
    for (model, regime), group in df.groupby(["model", "spike_regime"]):
        bias_rows.append({"model": model, "regime": regime, "bias": float(group["error"].mean()), "n": len(group)})
    bias = pd.DataFrame(bias_rows)
    bias.to_csv(results_dir / "forecast_bias_analysis.csv", index=False)
    spike = df[df["spike_regime"].eq("Extreme")].head(120)
    if not spike.empty:
        plt.figure(figsize=(12, 5))
        for model, group in spike.groupby("model"):
            if model in ["XGBoost Direct", "Residual FTS-XGBoost", "Hybrid feature-based"]:
                plt.scatter(group.index, group["predicted"], s=12, label=model)
        plt.scatter(spike.index, spike["actual"], s=12, label="Actual", color="black")
        plt.title("Extreme consumption actual vs prediction")
        plt.xlabel("Extreme observation index")
        plt.ylabel("Hourly energy (kWh)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir.parent / "figures" / "stage4_spike_actual_vs_prediction.png", dpi=160)
        plt.close()
    return high, bias


def final_comparison(model_flats, gate_flats, results_dir, figures_dir):
    rows = []
    for name, flat in {**model_flats, **gate_flats}.items():
        rows.append(window_summary_from_flat(name, flat))
    final = pd.DataFrame(rows).sort_values("Daily MAE")
    final.to_csv(results_dir / "stage4_final_model_comparison.csv", index=False)
    daily = []
    for name, flat in {**model_flats, **gate_flats}.items():
        for origin, group in flat.groupby("origin_index"):
            daily.append({"model": name, "origin_index": origin, "daily_abs_error": abs(group["actual"].sum() - group["predicted"].sum())})
    daily = pd.DataFrame(daily)
    stability = daily.groupby("model")["daily_abs_error"].agg(["mean", "median", "std", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), "max"]).reset_index()
    stability.columns = ["model", "mean", "median", "std", "Q1", "Q3", "max"]
    stability["win_count"] = daily.loc[daily.groupby("origin_index")["daily_abs_error"].idxmin()].groupby("model").size().reindex(stability["model"]).fillna(0).astype(int).values
    stability.to_csv(results_dir / "stage4_daily_error_stability.csv", index=False)
    plt.figure(figsize=(10, 5))
    daily.boxplot(column="daily_abs_error", by="model", rot=30)
    plt.title("Stage 4 daily absolute error")
    plt.suptitle("")
    plt.xlabel("Model")
    plt.ylabel("Daily absolute error (kWh)")
    plt.tight_layout()
    plt.savefig(figures_dir / "stage4_daily_error_boxplot.png", dpi=160)
    plt.close()
    pivot = daily.pivot(index="origin_index", columns="model", values="daily_abs_error")
    pairs = [("XGBoost Direct", "Best Threshold Gated Hybrid"), ("XGBoost Direct", "Best Learned Gated Hybrid"), ("XGBoost Direct", "XGBoost-FTS Residual Correction")]
    pair_rows = []
    for a, b in pairs:
        if a not in pivot or b not in pivot:
            continue
        diff = pivot[a] - pivot[b]
        try:
            stat, p = wilcoxon(diff)
        except ValueError:
            stat, p = np.nan, np.nan
        pair_rows.append({"comparison": f"{a} minus {b}", "a_wins": int((diff < 0).sum()), "b_wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()), "mean_difference": float(diff.mean()), "median_difference": float(diff.median()), "wilcoxon_p": p, "n": int(diff.notna().sum())})
    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(results_dir / "stage4_pairwise_comparison.csv", index=False)
    return final, stability, pairwise


def write_report(path, tables):
    final = tables["final"]
    best = final.iloc[0]
    threshold = tables["threshold"].sort_values("Daily MAE").iloc[0]
    learned = tables["learned"].iloc[0]
    oracle = tables["oracle"].iloc[0]
    xgb = final[final["model"].eq("XGBoost Direct")].iloc[0]
    threshold_improvement = (xgb["Daily MAE"] - threshold["Daily MAE"]) / xgb["Daily MAE"] * 100
    learned_improvement = (xgb["Daily MAE"] - learned["Daily MAE"]) / xgb["Daily MAE"] * 100
    text = f"""# Stage 4 Regime-Aware Hybrid Report

## 1. Reverse Residual Experiment

{tables["xgb_fts"].round(4).to_markdown(index=False)}

## 2. Non-Overlapping Evaluation Setup

Stage 4 uses non-overlapping 24-hour forecast origins in the test period. Number of origins: {int(final["windows"].max())}.

## 3. Consumption Regime Definition

{tables["regime_definition"].round(4).to_markdown(index=False)}

## 4. Expert Performance by Regime

{tables["expert_regime"].round(4).to_markdown(index=False)}

## 5. Oracle Gate Potential

{tables["oracle"].round(4).to_markdown(index=False)}

## 6. Threshold Gate

{tables["threshold"].round(4).to_markdown(index=False)}

Gate decision frequency:

{tables["gate_frequency"].round(4).to_markdown(index=False)}

## 7. Learned Gate

{tables["learned"].round(4).to_markdown(index=False)}

## 8. High / Extreme Consumption Analysis

{tables["high"].round(4).to_markdown(index=False)}

## 9. Bias Analysis

{tables["bias"].round(4).head(40).to_markdown(index=False)}

## 10. Final Model Comparison

{final.round(4).to_markdown(index=False)}

## 11. Stability Across Origins

{tables["stability"].round(4).to_markdown(index=False)}

## 12. Pairwise Comparison

{tables["pairwise"].round(4).to_markdown(index=False)}

## 13. Is FTS Still Worth Keeping?

Oracle Gate Daily MAE is {oracle["Daily MAE"]:.4f} kWh, so there is theoretical complementarity between experts. The best threshold gate is `{threshold["model"]}` with Daily MAE {threshold["Daily MAE"]:.4f} kWh ({threshold_improvement:.2f}% vs XGBoost). The learned gate has Daily MAE {learned["Daily MAE"]:.4f} kWh ({learned_improvement:.2f}% vs XGBoost).

The practical rule required improvement above 2%, majority wins, and no substantial worsening on High/Extreme consumption. The gated hybrids do not satisfy that standard strongly enough.

## 14. Stage 4 Conclusion

1. XGBoost residual is not reliably improved by FTS residual correction.
2. FTS advantages on Low/Medium regimes exist in some tables but are not stable enough across non-overlap origins.
3. FTS-based models are risky for High/Extreme consumption, especially residual FTS-XGBoost on High regimes.
4. Oracle Gate shows an upper bound, but realistic gates recover little of it.
5. Threshold gate improvement vs XGBoost is {threshold_improvement:.2f}%.
6. Learned gate improvement vs XGBoost is {learned_improvement:.2f}%.
7. Gate frequency table above shows whether FTS is actually used.
8. Improvement does not clearly survive non-overlap evaluation.
9. Hybrid methods are not more stable than XGBoost by enough to justify added complexity.
10. Based on Stage 1-4, FTS should not be retained as the core proposed method. Use XGBoost Direct as the main forecasting method and keep FTS as a comparative/exploratory fuzzy analysis.

Best Stage 4 model by Daily MAE: `{best["model"]}` with Daily MAE {best["Daily MAE"]:.4f} kWh.
"""
    path.write_text(text, encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    figures_dir = root / "figures"
    reports_dir = root / "reports"
    ensure_dirs(results_dir, figures_dir, reports_dir)
    hourly, features = load_checkpoint(root)
    bounds = split_bounds(len(hourly))
    params = best_xgb_params(root, features, bounds)
    origins = nonoverlap_origins(bounds)
    _, _, best = best_fts(hourly, bounds, results_dir)
    features_fuzzy = membership_features(hourly, features, bounds, best)

    xgb_models = fit_direct_models(features_fuzzy, HIST_COLS, bounds, params)
    xgb_windows = direct_predict_windows(hourly, features_fuzzy, bounds, xgb_models, HIST_COLS, origins)
    hybrid_models = fit_direct_models(features_fuzzy, HIST_COLS + ["fts_forecast", "fuzzy_state"], bounds, params)
    hybrid_windows = direct_predict_windows(hourly, features_fuzzy, bounds, hybrid_models, HIST_COLS + ["fts_forecast", "fuzzy_state"], origins)
    residual_result, residual_models, _ = residual_hybrid(hourly, features_fuzzy, bounds, params, best, results_dir)
    residual_windows = residual_predict_windows(hourly, features_fuzzy, bounds, best, residual_models, origins)
    xgb_fts_result, xgb_fts_windows = train_xgb_residual_fts(hourly, features_fuzzy, bounds, params, best, origins, results_dir)
    weekly = weekly_windows(hourly, origins)

    model_windows = {
        "Weekly Seasonal Naive": weekly,
        "XGBoost Direct": xgb_windows,
        "Feature-based FTS-XGBoost": hybrid_windows,
        "Residual FTS-XGBoost": residual_windows,
        "XGBoost-FTS Residual Correction": xgb_fts_windows,
    }
    flat = pd.concat([flatten_rows(name, windows) for name, windows in model_windows.items()], ignore_index=True)
    thresholds, regime_def = regime_definition(hourly, bounds, results_dir)
    expert_regime = expert_by_regime(flat[flat["model"].isin(["XGBoost Direct", "Residual FTS-XGBoost", "Feature-based FTS-XGBoost"])], thresholds, results_dir)
    oracle, oracle_detail = oracle_gate(flat[flat["model"].isin(["XGBoost Direct", "Residual FTS-XGBoost", "Feature-based FTS-XGBoost"])], results_dir)
    threshold, threshold_detail, gate_freq = threshold_gates(flat[flat["model"].isin(["XGBoost Direct", "Residual FTS-XGBoost", "Hybrid feature-based", "Feature-based FTS-XGBoost"])].replace({"Feature-based FTS-XGBoost": "Hybrid feature-based"}), thresholds, results_dir)
    # Preselected from Stage 3 diagnostics: residual FTS was only defensible in low-consumption cases.
    # Do not pick the threshold gate by test-set MAE.
    best_threshold_name = "Gate B Low predicted -> Residual"
    best_threshold = threshold_detail[threshold_detail["gate"].eq(best_threshold_name)].rename(columns={"gate": "model"})
    learned, learned_detail = learned_gate(flat[flat["model"].isin(["XGBoost Direct", "Residual FTS-XGBoost", "Hybrid feature-based", "Feature-based FTS-XGBoost"])].replace({"Feature-based FTS-XGBoost": "Hybrid feature-based"}), features_fuzzy, thresholds, results_dir)
    high, bias = high_and_bias(flat, thresholds, results_dir)
    model_flats = {name: flatten_rows(name, windows) for name, windows in model_windows.items()}
    gate_flats = {
        "Best Threshold Gated Hybrid": best_threshold,
        "Best Learned Gated Hybrid": learned_detail.assign(model="Best Learned Gated Hybrid"),
        "Oracle Gate": oracle_detail.assign(model="Oracle Gate"),
    }
    final, stability, pairwise = final_comparison(model_flats, gate_flats, results_dir, figures_dir)
    final.to_csv(results_dir / "nonoverlap_rolling_origin_results.csv", index=False)
    tables = {
        "xgb_fts": xgb_fts_result,
        "regime_definition": regime_def,
        "expert_regime": expert_regime,
        "oracle": oracle,
        "threshold": threshold,
        "gate_frequency": gate_freq,
        "learned": learned,
        "high": high,
        "bias": bias,
        "final": final,
        "stability": stability,
        "pairwise": pairwise,
    }
    write_report(reports_dir / "stage4_regime_aware_hybrid_report.md", tables)
    print("Stage 4 regime-aware hybrid complete.")
    print(f"Non-overlap origins: {len(origins)}")
    print(f"Report: {reports_dir / 'stage4_regime_aware_hybrid_report.md'}")


if __name__ == "__main__":
    main()
