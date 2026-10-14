# Stage 6 Reproduction Report

| model            |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows | Notes   |
|:-----------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|----------:|:--------|
| XGBoost Direct   |     0.213333 |      0.355831 |       36.1312 |        30.0461 |     3.28985 |      4.15394 |      24.09   |       22.9296 |        20 |         |
| Threshold Gate B |     0.219958 |      0.357415 |       38.1847 |        32.8062 |     3.2028  |      4.10161 |      23.2123 |       22.2971 |        20 |         |

Hasil reproduksi mendekati checkpoint Stage 4-5 karena model dilatih ulang deterministik dari prosedur yang sama. Detail origin tersimpan di `results/stage6/reproduction_check.csv`.
