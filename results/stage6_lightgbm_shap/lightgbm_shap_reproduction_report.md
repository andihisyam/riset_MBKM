# Reproduksi SHAP Hybrid FTS1-LightGBM Fixed 0.31

Analisis ini mereproduksi dua cabang model pada konfigurasi Stage 7: LightGBM Direct dan Residual FTS1-LightGBM. Threshold gate dikunci pada 0,31 kWh dan tidak dituning ulang.

- Forecast origin test: 20
- Baris SHAP per cabang: 480
- Aktivasi residual: 52 dari 480 (10.83%)
- Maksimum selisih reproduksi hybrid terhadap Stage 7: 2.22044604925e-16
- Maksimum error additivity Direct: 3.99680288865e-15
- Maksimum error additivity Residual: 1.05471187339e-15

SHAP residual menjelaskan output korektor residual, bukan nilai FTS awal atau hasil clipping prediksi akhir. Gate dijelaskan terpisah melalui decision trace.
