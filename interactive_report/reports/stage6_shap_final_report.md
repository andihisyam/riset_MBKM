# Stage 6 SHAP Final Report

## 1. Introduction
Interpretabilitas diperlukan agar prediksi konsumsi energi 24 jam dapat dijelaskan, terutama karena model menggunakan lag, rolling statistics, dan gate hybrid.

## 2. Methodology
Tree SHAP digunakan untuk XGBoost Direct dan XGBoost residual corrector. Gate B dijelaskan dengan decision trace karena merupakan aturan threshold deterministik.

## 3. Experimental Setup
Data hourly clean, split kronologis, non-overlap origins, dan threshold Stage 5 digunakan kembali. Model direkonstruksi dari prosedur training karena artifact model binary tidak tersedia.

## 4. Results
Top fitur global:

| feature         |   mean_abs_shap |
|:----------------|----------------:|
| hour            |        0.071305 |
| cos_hour        |        0.060992 |
| sin_hour        |        0.051994 |
| rolling_mean_24 |        0.025619 |
| rolling_std_24  |        0.020929 |

Reproduksi:

| model            |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows | Notes   |
|:-----------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|----------:|:--------|
| XGBoost Direct   |     0.213333 |      0.355831 |       36.1312 |        30.0461 |     3.28985 |      4.15394 |      24.09   |       22.9296 |        20 |         |
| Threshold Gate B |     0.219958 |      0.357415 |       38.1847 |        32.8062 |     3.2028  |      4.10161 |      23.2123 |       22.2971 |        20 |         |

## 5. Discussion
SHAP menunjukkan prediksi didominasi fitur historis/temporal. Gate B hanya mengalihkan prediksi saat XGBoost forecast berada pada regime rendah, sehingga improvement harian berasal dari switching terbatas.

## 6. Limitations
SHAP tidak kausal, fitur berkorelasi, data satu rumah, dan evaluasi hanya 20 origin non-overlap. Keunggulan Gate B belum signifikan pada taraf 5%.

## 7. Conclusion
Stage 6 memberi penjelasan yang konsisten dengan Stage 1-5. Final proposed method tetap Direct multi-horizon XGBoost; FTS dipertahankan sebagai pembanding/analisis fuzzy, bukan komponen inti.
