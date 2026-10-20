"""Hourly ARIMA/SARIMA benchmark aligned with Stage 1-7 evaluation."""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "stage9_arima_sarima"
FIG = OUT / "figures"
TARGET = "Appliances_kWh"
RANDOM_STATE = 42


def metric(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    denom = np.abs(a) + np.abs(p)
    return {
        "Hourly MAE": float(np.mean(np.abs(a - p))),
        "Hourly RMSE": float(np.sqrt(np.mean((a - p) ** 2))),
        "Hourly MAPE": float(np.mean(np.abs((a[a != 0] - p[a != 0]) / a[a != 0])) * 100) if (a != 0).any() else np.nan,
        "Hourly sMAPE": float(np.mean(2 * np.abs(a[denom != 0] - p[denom != 0]) / denom[denom != 0]) * 100) if (denom != 0).any() else np.nan,
        "Bias": float(np.mean(p - a)),
    }


def fit_forecast(y: np.ndarray, origin: int, family: str, order: tuple, seasonal_order: tuple | None) -> np.ndarray:
    history = y[: origin + 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family == "ARIMA":
            model = ARIMA(history, order=order, trend=None, enforce_stationarity=False, enforce_invertibility=False).fit(method_kwargs={"maxiter": 100})
        else:
            model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend=None, simple_differencing=True, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=100)
    return np.maximum(np.asarray(model.forecast(24), dtype=float), 0.0)


def origins_from_config() -> tuple[list[int], list[int]]:
    cfg = ROOT / "configs" / "stage7" / "stage7_config.json"
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        return [int(x) for x in data["val_origins"]], [int(x) for x in data["test_origins"]]
    raise FileNotFoundError("Stage 7 origin configuration is required for aligned comparison")


def candidates() -> list[dict]:
    return [
        {"family": "ARIMA", "order": (1, 1, 1), "seasonal_order": None},
        {"family": "ARIMA", "order": (2, 1, 1), "seasonal_order": None},
        {"family": "SARIMA", "order": (1, 1, 1), "seasonal_order": (1, 0, 1, 24)},
        {"family": "SARIMA", "order": (1, 0, 1), "seasonal_order": (1, 0, 1, 24)},
    ]


def run_candidate(y: np.ndarray, timestamps: pd.Series, origins: list[int], spec: dict, split: str) -> tuple[dict, list[dict]]:
    windows, rows = [], []
    for origin in origins:
        actual = y[origin + 1 : origin + 25]
        try:
            pred = fit_forecast(y, origin, spec["family"], spec["order"], spec["seasonal_order"])
            if len(actual) != 24:
                continue
            windows.append((actual, pred))
            for h, (a, p) in enumerate(zip(actual, pred), 1):
                rows.append({"family": spec["family"], "order": str(spec["order"]), "seasonal_order": str(spec["seasonal_order"]), "split": split, "origin_index": origin, "origin": str(timestamps.iloc[origin]), "horizon": h, "actual_kWh": float(a), "predicted_kWh": float(p), "abs_error": float(abs(a - p))})
        except Exception as exc:
            rows.append({"family": spec["family"], "order": str(spec["order"]), "seasonal_order": str(spec["seasonal_order"]), "split": split, "origin_index": origin, "error": type(exc).__name__ + ": " + str(exc)[:200]})
    if not windows:
        return {"family": spec["family"], "order": str(spec["order"]), "seasonal_order": str(spec["seasonal_order"]), "split": split, "windows": 0, "Daily MAE": np.nan, "Daily RMSE": np.nan, "Daily MAPE": np.nan, "Daily sMAPE": np.nan, "error": "no successful origins"}, rows
    hourly_actual = np.concatenate([x[0] for x in windows]); hourly_pred = np.concatenate([x[1] for x in windows])
    daily_actual = np.asarray([x[0].sum() for x in windows]); daily_pred = np.asarray([x[1].sum() for x in windows])
    row = {"model": f"{spec['family']} {spec['order']} {spec['seasonal_order']}", "family": spec["family"], "order": str(spec["order"]), "seasonal_order": str(spec["seasonal_order"]), "split": split, "windows": len(windows), **metric(hourly_actual, hourly_pred)}
    row.update({f"Daily {k.split(' ', 1)[1]}": v for k, v in metric(daily_actual, daily_pred).items() if k.startswith("Hourly ")})
    return row, rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    hourly = pd.read_csv(ROOT / "data" / "hourly_energy_clean.csv", parse_dates=["date"])
    y = hourly[TARGET].to_numpy(float)
    val_origins, test_origins = origins_from_config()
    result_rows, prediction_rows, tuning_rows = [], [], []
    for spec in candidates():
        val_row, val_pred = run_candidate(y, hourly["date"], val_origins, spec, "validation")
        test_row, test_pred = run_candidate(y, hourly["date"], test_origins, spec, "test")
        result_rows.extend([val_row, test_row]); prediction_rows.extend(val_pred + test_pred)
        tuning_rows.append({"family": spec["family"], "order": str(spec["order"]), "seasonal_order": str(spec["seasonal_order"]), "validation_daily_mae": val_row.get("Daily MAE"), "validation_windows": val_row.get("windows"), "selected_by_validation": False})
    results = pd.DataFrame(result_rows)
    tuning = pd.DataFrame(tuning_rows)
    valid = tuning.dropna(subset=["validation_daily_mae"]).sort_values("validation_daily_mae")
    selected = valid.iloc[0].to_dict()
    selected["selected_by_validation"] = True
    tuning.loc[tuning.validation_daily_mae == selected["validation_daily_mae"], "selected_by_validation"] = True
    selected_key = (selected["family"], selected["order"], selected["seasonal_order"])
    results["protocol"] = "hourly_h1_h24"; results["resolution"] = "hourly"; results["target_unit"] = "kWh"; results["selected_by_validation"] = results.apply(lambda r: (r.family, r.order, r.seasonal_order) == selected_key, axis=1)
    results.to_csv(OUT / "ARIMA_SARIMA_RESULTS.csv", index=False); tuning.to_csv(OUT / "ARIMA_SARIMA_TUNING.csv", index=False); pd.DataFrame(prediction_rows).to_csv(OUT / "ARIMA_SARIMA_PREDICTIONS.csv", index=False)
    config = {"random_state": RANDOM_STATE, "target": TARGET, "forecast_horizon": 24, "validation_origins": val_origins, "test_origins": test_origins, "candidates": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()} for s in candidates()], "selected_by_validation": selected}
    (OUT / "stage9_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    stage7 = pd.read_csv(ROOT / "results" / "stage7" / "stage7_final_model_ranking.csv")
    stage7["protocol"] = "hourly_h1_h24"; stage7["resolution"] = "hourly"; stage7["target_unit"] = "kWh"; stage7["source_stage"] = "Stage 7"; stage7["split"] = "test"; stage7["selected_by_validation"] = False
    arima_test = results[results.split.eq("test")].copy(); arima_test["source_stage"] = "Stage 9"; arima_test["source_model"] = arima_test["model"]; arima_test["model"] = arima_test["model"].astype(str); arima_test["Daily MAE"] = arima_test["Daily MAE"].astype(float)
    catalog = pd.concat([stage7, arima_test], ignore_index=True, sort=False); catalog.to_csv(OUT / "ALL_HOURLY_MODELS.csv", index=False)
    ranking = catalog[catalog["split"].eq("test")].sort_values("Daily MAE").reset_index(drop=True); ranking["hourly_rank_by_daily_mae"] = np.arange(1, len(ranking) + 1); ranking.to_csv(OUT / "HOURLY_MODEL_RANKING.csv", index=False)
    plt.figure(figsize=(11, 6)); top = ranking.head(15); plt.barh(top["model"].astype(str)[::-1], top["Daily MAE"][::-1]); plt.xlabel("Daily MAE (kWh)"); plt.title("Ranking model hourly Stage 1-7 + ARIMA/SARIMA"); plt.tight_layout(); plt.savefig(FIG / "hourly_model_ranking.png", dpi=160); plt.close()
    report = f"""# Stage 9 — ARIMA dan SARIMA\n\nARIMA dan SARIMA dilatih pada data hourly dengan forecast horizon H1-H24 dan forecast origin yang sama dengan Stage 7. Parameter dipilih dari validation Daily MAE; test hanya dipakai untuk evaluasi akhir.\n\n## Konfigurasi terpilih\n\n- Model: `{selected['family']}`\n- Order: `{selected['order']}`\n- Seasonal order: `{selected['seasonal_order']}`\n- Validation Daily MAE: **{selected['validation_daily_mae']:.6f} kWh**\n\nHasil seluruh kandidat ada di `ARIMA_SARIMA_RESULTS.csv`. Katalog gabungan model hourly Stage 1-7 dan Stage 9 ada di `ALL_HOURLY_MODELS.csv`; ranking ada di `HOURLY_MODEL_RANKING.csv`. Stage 8 tidak dimasukkan ke ranking ini karena memakai resolusi 10 menit dan satuan Wh.\n"""
    (OUT / "STAGE9_REPORT.md").write_text(report, encoding="utf-8")
    shutil.make_archive(str((ROOT / "stage9_arima_sarima_package")), "zip", OUT)
    print(json.dumps({"output": str(OUT), "zip": str(ROOT / "stage9_arima_sarima_package.zip"), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
