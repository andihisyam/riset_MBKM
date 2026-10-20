# Stage 9 — ARIMA dan SARIMA

ARIMA dan SARIMA dilatih pada data hourly dengan forecast horizon H1-H24 dan forecast origin yang sama dengan Stage 7. Parameter dipilih dari validation Daily MAE; test hanya dipakai untuk evaluasi akhir.

## Konfigurasi terpilih

- Model: `SARIMA`
- Order: `(1, 0, 1)`
- Seasonal order: `(1, 0, 1, 24)`
- Validation Daily MAE: **2.431812 kWh**

Hasil seluruh kandidat ada di `ARIMA_SARIMA_RESULTS.csv`. Katalog gabungan model hourly Stage 1-7 dan Stage 9 ada di `ALL_HOURLY_MODELS.csv`; ranking ada di `HOURLY_MODEL_RANKING.csv`. Stage 8 tidak dimasukkan ke ranking ini karena memakai resolusi 10 menit dan satuan Wh.
