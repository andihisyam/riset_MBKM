# Stage 2 Experiment Report

## 1. Hourly aggregation correction

Hourly aggregation now counts raw 10-minute records per hour with `raw_count`. Normal complete hours require `raw_count = 6`. Incomplete hours found: 1. Clean modeling rows: 3,289.

`Appliances` is summed across six 10-minute Wh intervals and converted to `Appliances_kWh`, meaning total energy used during the one-hour interval in kWh. Temperature, humidity, and weather variables use mean aggregation.

## 2. Baseline correction

The main benchmark now uses fixed-origin 24-hour forecasting. The previous one-step daily file remains diagnostic only.

| model           |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows_evaluated | Notes                               |
|:----------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|--------------------:|:------------------------------------|
| Last Value      |       0.26   |        0.4883 |       29.4971 |        40.6913 |      6.0295 |       7.4871 |      39.2048 |       51.5878 |                  20 | Fixed-origin 24-hour naive baseline |
| Daily Seasonal  |       0.2645 |        0.4763 |       40.883  |        34.5188 |      4.255  |       6.021  |      27.8263 |       28.2832 |                  20 | Fixed-origin 24-hour naive baseline |
| Weekly Seasonal |       0.2498 |        0.4687 |       39.3266 |        32.0723 |      3.777  |       5.7126 |      26.5959 |       24.7501 |                  20 | Fixed-origin 24-hour naive baseline |

Valid benchmark after correction: Weekly Seasonal Naive Daily MAE = 3.7770 kWh and Daily MAPE = 26.60%.

## 3. Feature engineering

Hourly features include lag 1, 2, 3, 6, 12, 24, 48, and 168; rolling means 3, 6, 12, 24; rolling std 3, 6, 24; integer calendar features; and cyclical hour/day-of-week encodings. Rolling features use `shift(1)` before rolling, so current/future target values do not enter features.

## 4. XGBoost standalone results

| model                                     |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows_evaluated | Notes                                                                                |
|:------------------------------------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|--------------------:|:-------------------------------------------------------------------------------------|
| XGBoost Historical Recursive              |       0.2311 |        0.3741 |       39.3882 |        32.3736 |      3.48   |       4.5008 |      25.7943 |       24.0929 |                  20 | Realistic historical-only recursive 24-hour forecast                                 |
| XGBoost Historical Direct                 |       0.2133 |        0.3558 |       36.1312 |        30.0461 |      3.2899 |       4.1539 |      24.09   |       22.9296 |                  20 | Realistic historical-only direct H1-H24 forecast                                     |
| XGBoost Historical + Environmental Direct |       0.492  |        0.6293 |      116.336  |        61.4928 |      9.9388 |      12.0897 |      86.5271 |       52.49   |                  20 | Oracle/exogenous-information experiment using actual future-row environmental values |

## 5. Multi-horizon XGBoost results

| model                                     |   horizon_hours |    MAE |   RMSE |     MAPE |   sMAPE |   windows_evaluated |
|:------------------------------------------|----------------:|-------:|-------:|---------:|--------:|--------------------:|
| XGBoost Historical Recursive              |               1 | 0.0193 | 0.0224 |   6.0278 |  6.0298 |                  20 |
| XGBoost Historical Recursive              |               3 | 0.0775 | 0.135  |  16.3705 | 17.361  |                  20 |
| XGBoost Historical Recursive              |               6 | 0.3226 | 0.4328 |  63.2212 | 49.611  |                  20 |
| XGBoost Historical Recursive              |              12 | 0.3092 | 0.4355 |  59.2124 | 44.2668 |                  20 |
| XGBoost Historical Recursive              |              24 | 0.0297 | 0.035  |   9.3987 |  9.4932 |                  20 |
| XGBoost Historical Direct                 |               1 | 0.0219 | 0.0392 |   6.9582 |  6.4876 |                  20 |
| XGBoost Historical Direct                 |               3 | 0.1097 | 0.1665 |  26.0104 | 24.0194 |                  20 |
| XGBoost Historical Direct                 |               6 | 0.3236 | 0.4194 |  69.1178 | 49.6266 |                  20 |
| XGBoost Historical Direct                 |              12 | 0.2695 | 0.3781 |  53.3537 | 39.385  |                  20 |
| XGBoost Historical Direct                 |              24 | 0.0365 | 0.058  |  11.7254 | 10.8942 |                  20 |
| XGBoost Historical + Environmental Direct |               1 | 0.2873 | 0.3473 |  90.5511 | 55.9551 |                  20 |
| XGBoost Historical + Environmental Direct |               3 | 0.5533 | 0.6504 | 147.846  | 75.4444 |                  20 |
| XGBoost Historical + Environmental Direct |               6 | 0.833  | 0.9746 | 220.062  | 86.6243 |                  20 |
| XGBoost Historical + Environmental Direct |              12 | 0.6623 | 0.7958 | 158.814  | 73.2913 |                  20 |
| XGBoost Historical + Environmental Direct |              24 | 0.3043 | 0.3785 |  95.3851 | 57.2186 |                  20 |

## 6. Fuzzy partition experiment

Best validation partition: `quantile` with 15 requested intervals. Validation MAE = 0.3066.

| partition   |   requested_intervals |   actual_states | boundaries                                                                                                                                            |   min_state_count |   max_state_count |   state_imbalance |   FLR |   FLRG |    MAE |   RMSE |     MAPE |   sMAPE |
|:------------|----------------------:|----------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------|------------------:|------------------:|------------------:|------:|-------:|-------:|-------:|---------:|--------:|
| quantile    |                    15 |              15 | [0.17, 0.23, 0.26, 0.28, 0.29, 0.31, 0.33, 0.35, 0.39, 0.46, 0.55, 0.65, 0.81, 1.072, 1.52, 3.65]                                                     |                97 |               180 |            1.8557 |  2301 |    204 | 0.3066 | 0.4097 |  66.4265 | 50.1379 |
| quantile    |                    11 |              11 | [0.17, 0.24, 0.27, 0.29, 0.32, 0.34, 0.4, 0.51, 0.64, 0.86, 1.36, 3.65]                                                                               |               163 |               280 |            1.7178 |  2301 |    116 | 0.3272 | 0.4247 |  72.5805 | 53.2382 |
| quantile    |                     9 |               9 | [0.17, 0.25, 0.28, 0.31, 0.34, 0.41, 0.55, 0.75, 1.213333, 3.65]                                                                                      |               212 |               305 |            1.4387 |  2301 |     78 | 0.3402 | 0.4308 |  76.7614 | 55.0074 |
| quantile    |                     7 |               7 | [0.17, 0.26, 0.3, 0.34, 0.42, 0.62, 1.02, 3.65]                                                                                                       |               280 |               355 |            1.2679 |  2301 |     49 | 0.3558 | 0.4476 |  81.322  | 56.97   |
| quantile    |                     5 |               5 | [0.17, 0.28, 0.33, 0.46, 0.81, 3.65]                                                                                                                  |               448 |               474 |            1.058  |  2301 |     25 | 0.3956 | 0.4702 |  94.3051 | 61.7811 |
| kmeans      |                    11 |              11 | [0.17, 0.289721, 0.406049, 0.565421, 0.761917, 1.004059, 1.272423, 1.54596, 1.853252, 2.260598, 2.817303, 3.65]                                       |                12 |               731 |           60.9167 |  2301 |    109 | 0.7182 | 0.7516 | 186.294  | 88.534  |
| kmeans      |                    15 |              15 | [0.17, 0.27573, 0.353037, 0.453041, 0.576101, 0.720192, 0.886468, 1.0844, 1.300717, 1.493625, 1.675927, 1.896714, 2.198036, 2.556857, 3.049417, 3.65] |                 6 |               643 |          107.167  |  2301 |    182 | 0.7433 | 0.7791 | 194.282  | 89.9625 |
| kmeans      |                     9 |               9 | [0.17, 0.317364, 0.474908, 0.708703, 1.000897, 1.341267, 1.732273, 2.214792, 2.810625, 3.65]                                                          |                12 |               810 |           67.5    |  2301 |     73 | 0.8118 | 0.8538 | 214.19   | 93.6936 |
| kmeans      |                     5 |               5 | [0.17, 0.453556, 0.806234, 1.306203, 2.055333, 3.65]                                                                                                  |                69 |              1372 |           19.8841 |  2301 |     25 | 0.8218 | 0.8589 | 217.41   | 94.5491 |
| kmeans      |                     7 |               7 | [0.17, 0.421491, 0.679887, 0.98515, 1.354935, 1.83794, 2.471213, 3.65]                                                                                |                31 |              1320 |           42.5806 |  2301 |     48 | 0.886  | 0.9242 | 234.649  | 97.895  |

## 7. FTS model comparison

Best validation FTS: `Weighted FTS`. Validation MAE = 0.2436.

| model                           |   order | weighted   |    MAE |   RMSE |    MAPE |   sMAPE |
|:--------------------------------|--------:|:-----------|-------:|-------:|--------:|--------:|
| Weighted FTS                    |       1 | True       | 0.2436 | 0.3692 | 44.4699 | 35.9148 |
| Weighted High-Order FTS order 2 |       2 | True       | 0.2523 | 0.3928 | 45.6031 | 35.7611 |
| High-Order FTS order 2          |       2 | False      | 0.2661 | 0.3911 | 50.9769 | 39.7435 |
| Weighted High-Order FTS order 3 |       3 | True       | 0.2775 | 0.4665 | 49.3583 | 38.5542 |
| High-Order FTS order 3          |       3 | False      | 0.2855 | 0.4756 | 51.8661 | 39.298  |
| High-Order FTS order 6          |       6 | False      | 0.3005 | 0.4273 | 61.0385 | 49.0543 |
| Weighted High-Order FTS order 6 |       6 | True       | 0.3005 | 0.4273 | 61.0385 | 49.0543 |
| Chen FTS                        |       1 | False      | 0.3066 | 0.4097 | 66.4265 | 50.1379 |

## 8. Hybrid FTS-XGBoost results

| model              |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows_evaluated | Notes                                                                |
|:-------------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|--------------------:|:---------------------------------------------------------------------|
| Hybrid FTS-XGBoost |       0.2138 |        0.3564 |       36.4537 |        30.1357 |      3.3414 |       4.1816 |      24.6635 |       23.2709 |                  20 | Direct H1-H24 with lag, rolling, calendar, fts_forecast, fuzzy_state |

## 9. Ablation analysis

| model               |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows_evaluated | Notes                  |   Hourly MAE improvement vs + Rolling (%) |   Daily MAE improvement vs + Rolling (%) |
|:--------------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|--------------------:|:-----------------------|------------------------------------------:|-----------------------------------------:|
| Base Lag + Calendar |       0.2132 |        0.3593 |       35.4384 |        29.7052 |      3.395  |       4.2563 |      24.3881 |       23.6486 |                  20 | Direct H1-H24 ablation |                                    0.0761 |                                  -3.1968 |
| + Rolling           |       0.2133 |        0.3558 |       36.1312 |        30.0461 |      3.2899 |       4.1539 |      24.09   |       22.9296 |                  20 | Direct H1-H24 ablation |                                    0      |                                   0      |
| + FTS               |       0.2138 |        0.3564 |       36.4537 |        30.1357 |      3.3414 |       4.1816 |      24.6635 |       23.2709 |                  20 | Direct H1-H24 ablation |                                   -0.2336 |                                  -1.567  |

## 10. Comparison against Weekly Seasonal Naive

| Model                     |   Hourly MAE |   Hourly RMSE |   Daily MAE |   Daily MAPE | Notes                                                                |
|:--------------------------|-------------:|--------------:|------------:|-------------:|:---------------------------------------------------------------------|
| XGBoost Historical Direct |       0.2133 |        0.3558 |      3.2899 |      24.09   | Realistic historical-only direct H1-H24 forecast                     |
| Hybrid FTS-XGBoost        |       0.2138 |        0.3564 |      3.3414 |      24.6635 | Direct H1-H24 with lag, rolling, calendar, fts_forecast, fuzzy_state |
| Weekly Seasonal           |       0.2498 |        0.4687 |      3.777  |      26.5959 | Fixed-origin 24-hour naive baseline                                  |
| Daily Seasonal            |       0.2645 |        0.4763 |      4.255  |      27.8263 | Fixed-origin 24-hour naive baseline                                  |
| Best FTS Standalone       |       0.3495 |        0.4403 |      5.2865 |      44.3384 | Weighted FTS                                                         |
| Last Value                |       0.26   |        0.4883 |      6.0295 |      39.2048 | Fixed-origin 24-hour naive baseline                                  |

## 11. Preliminary conclusion

Hourly remains suitable after dropping incomplete hours because it aligns with the 24-hour forecasting objective and leaves enough observations for chronological validation.

The valid fixed-origin 24-hour benchmark is Weekly Seasonal Naive with Daily MAE 3.7770 kWh and Daily MAPE 26.60%. XGBoost Historical Direct beats that benchmark with Daily MAE 3.2899 kWh, an improvement of 12.90% by Daily MAE.

The best fuzzy partition is quantile with 15 intervals, and the best validation FTS type is Weighted FTS. As a standalone fixed-origin model, FTS is weaker than XGBoost and the weekly seasonal baseline.

Hybrid FTS-XGBoost has Daily MAE 3.3414 kWh. That is -1.57% versus XGBoost Historical Direct by Daily MAE, so the current fuzzy features do not improve XGBoost. They are marginally worse on daily total accuracy.

The best temporary multi-step method is Direct H1-H24 XGBoost, improving Daily MAE by 5.46% versus recursive XGBoost. The environmental oracle/exogenous experiment is also worse here, with Daily MAE 9.9388 kWh, so environmental variables should not be added to the main proposed model yet.

The main limitation remains the short single-house dataset, so these results support preliminary method selection only. SHAP and 7-day forecasting were not run in this stage.
