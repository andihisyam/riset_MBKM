from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "results" / "stage9_arima_sarima" / "ALL_HOURLY_MODELS.csv"
output_dir = ROOT / "results" / "stage9_arima_sarima"

models = pd.read_csv(source)
required = {
    "model",
    "algorithm",
    "family",
    "fts_order",
    "Hourly MAPE",
    "Daily MAPE",
    "Daily MAE",
    "Hourly MAE",
    "rank",
}
missing = required.difference(models.columns)
assert not missing, f"Kolom wajib hilang: {sorted(missing)}"

table = models[
    [
        "rank",
        "model",
        "algorithm",
        "family",
        "fts_order",
        "Daily MAPE",
        "Hourly MAPE",
        "Daily MAE",
        "Hourly MAE",
    ]
].copy()
table = table.rename(
    columns={
        "rank": "Peringkat test Daily MAE",
        "model": "Model",
        "algorithm": "Algoritma",
        "family": "Pendekatan",
        "fts_order": "Orde FTS",
        "Daily MAPE": "Test Daily MAPE (%)",
        "Hourly MAPE": "Test Hourly MAPE (%)",
        "Daily MAE": "Test Daily MAE (kWh)",
        "Hourly MAE": "Test Hourly MAE (kWh)",
    }
)
table["Orde FTS"] = table["Orde FTS"].fillna("-")
table = table.sort_values("Peringkat test Daily MAE", kind="stable")

csv_path = output_dir / "ALL_MODELS_MAPE_TABLE.csv"
md_path = output_dir / "ALL_MODELS_MAPE_TABLE.md"
table.to_csv(csv_path, index=False, float_format="%.6f")

display = table.copy()
for column in [
    "Test Daily MAPE (%)",
    "Test Hourly MAPE (%)",
    "Test Daily MAE (kWh)",
    "Test Hourly MAE (kWh)",
]:
    display[column] = display[column].map(lambda value: f"{value:.4f}")

lines = [
    "# Tabel MAPE Seluruh Model Hourly",
    "",
    "Tabel ini menggunakan prediksi test asli dari `ALL_HOURLY_MODELS.csv`. "
    "Sebanyak 57 konfigurasi dibandingkan; tidak ada training atau tuning ulang.",
    "",
    "> Catatan: MAPE dapat membesar ketika nilai aktual konsumsi mendekati nol. "
    "Karena itu MAPE dibaca bersama Daily MAE dan Hourly MAE.",
    "",
    display.to_markdown(index=False),
    "",
]
md_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Menyimpan {len(table)} baris ke {csv_path}")
print(f"Menyimpan tabel Markdown ke {md_path}")
