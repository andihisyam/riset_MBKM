# Stage 3 Hybrid Audit Report

## 1. Multi-Horizon Audit

Full H1-H24 audit is saved in `results/h1_h24_full_audit.csv`. The non-monotonic H1-H24 pattern is valid under the checked feature-target alignment. H6/H12 remain harder, while H24 benefits from daily seasonality.

|   horizon |   Daily Seasonal Naive |   Weekly Seasonal Naive |   XGBoost Historical Direct |   XGBoost Historical Recursive |
|----------:|-----------------------:|------------------------:|----------------------------:|-------------------------------:|
|         1 |                 0.028  |                  0.016  |                      0.0219 |                         0.0439 |
|         2 |                 0.0575 |                  0.069  |                      0.0548 |                         0.0692 |
|         3 |                 0.1265 |                  0.0995 |                      0.1097 |                         0.1549 |
|         4 |                 0.235  |                  0.259  |                      0.1703 |                         0.261  |
|         5 |                 0.4285 |                  0.425  |                      0.3067 |                         0.3677 |
|         6 |                 0.362  |                  0.304  |                      0.3236 |                         0.4415 |
|         7 |                 0.4185 |                  0.3795 |                      0.3575 |                         0.3851 |
|         8 |                 0.568  |                  0.5135 |                      0.5041 |                         0.4848 |
|         9 |                 0.479  |                  0.443  |                      0.3486 |                         0.3188 |
|        10 |                 0.5445 |                  0.441  |                      0.3383 |                         0.3248 |
|        11 |                 0.4375 |                  0.2715 |                      0.2909 |                         0.3232 |
|        12 |                 0.385  |                  0.2925 |                      0.2695 |                         0.3298 |
|        13 |                 0.3985 |                  0.599  |                      0.4109 |                         0.5215 |
|        14 |                 0.547  |                  0.5645 |                      0.536  |                         0.4769 |
|        15 |                 0.4095 |                  0.4645 |                      0.3692 |                         0.304  |
|        16 |                 0.206  |                  0.2335 |                      0.1956 |                         0.1485 |
|        17 |                 0.2945 |                  0.219  |                      0.1765 |                         0.2175 |
|        18 |                 0.1665 |                  0.15   |                      0.1229 |                         0.1305 |
|        19 |                 0.058  |                  0.0595 |                      0.0445 |                         0.0589 |
|        20 |                 0.096  |                  0.073  |                      0.0555 |                         0.0781 |
|        21 |                 0.0255 |                  0.03   |                      0.0225 |                         0.0298 |
|        22 |                 0.0205 |                  0.026  |                      0.0221 |                         0.0216 |
|        23 |                 0.027  |                  0.033  |                      0.0319 |                         0.023  |
|        24 |                 0.03   |                  0.0305 |                      0.0365 |                         0.0315 |

## 2. Horizon Alignment Verification

Automated timestamp checks are saved in `results/horizon_alignment_audit.csv`. All sampled checks valid: True.

## 3. Environmental Feature Audit

Environmental schema/alignment diagnostics are saved in `results/environmental_schema_audit.csv`. The corrected comparison separates current-time environmental features from oracle future environmental features.

| variant                                  |   horizon |    MAE |   RMSE |     MAPE |   sMAPE |   windows |
|:-----------------------------------------|----------:|-------:|-------:|---------:|--------:|----------:|
| Historical only                          |         1 | 0.0219 | 0.0392 |   6.9582 |  6.4876 |        20 |
| Historical only                          |         6 | 0.3236 | 0.4194 |  69.1178 | 49.6266 |        20 |
| Historical only                          |        12 | 0.2695 | 0.3781 |  53.3537 | 39.385  |        20 |
| Historical only                          |        24 | 0.0365 | 0.058  |  11.7254 | 10.8942 |        20 |
| Historical + current environmental       |         1 | 0.1687 | 0.1989 |  53.0577 | 39.3997 |        20 |
| Historical + current environmental       |         6 | 0.5701 | 0.6881 | 147.576  | 70.1693 |        20 |
| Historical + current environmental       |        12 | 0.534  | 0.6945 | 118.024  | 61.4777 |        20 |
| Historical + current environmental       |        24 | 0.1441 | 0.1926 |  45.9115 | 33.7141 |        20 |
| Historical + oracle future environmental |         1 | 0.2873 | 0.3473 |  90.5511 | 55.9551 |        20 |
| Historical + oracle future environmental |         6 | 0.833  | 0.9746 | 220.062  | 86.6243 |        20 |
| Historical + oracle future environmental |        12 | 0.6623 | 0.7958 | 158.814  | 73.2913 |        20 |
| Historical + oracle future environmental |        24 | 0.3043 | 0.3785 |  95.3851 | 57.2186 |        20 |

## 4. Residual FTS-XGBoost

| model                |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows | Notes                                         |
|:---------------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|----------:|:----------------------------------------------|
| Residual FTS-XGBoost |       0.2255 |        0.3696 |       37.5247 |        38.5076 |      3.2989 |       4.7159 |      20.6606 |       23.3278 |        20 | FTS forecast plus XGBoost residual correction |

The reverse direction audit (`XGBoost -> FTS residual`) is saved in `results/xgb_residual_structure_audit.csv`.

|   acf_lag_1 |   acf_lag_24 |   residual_std | decision                        |
|------------:|-------------:|---------------:|:--------------------------------|
|      0.2442 |       0.0814 |         0.2717 | residual has temporal structure |

## 5. Membership Degree Experiment

| model                               |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows | Notes                      |
|:------------------------------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|----------:|:---------------------------|
| XGBoost historical                  |       0.2133 |        0.3558 |       36.1312 |        30.0461 |      3.2899 |       4.1539 |      24.09   |       22.9296 |        20 | Fixed-origin direct H1-H24 |
| XGBoost + FTS forecast              |       0.2149 |        0.3574 |       36.4247 |        30.1504 |      3.3558 |       4.175  |      24.6191 |       23.3863 |        20 | Fixed-origin direct H1-H24 |
| XGBoost + fuzzy state               |       0.2158 |        0.3579 |       36.8131 |        30.3496 |      3.3775 |       4.2099 |      24.9117 |       23.5339 |        20 | Fixed-origin direct H1-H24 |
| XGBoost + membership degree         |       0.2148 |        0.3578 |       36.2901 |        30.2062 |      3.3153 |       4.1533 |      24.2486 |       23.1224 |        20 | Fixed-origin direct H1-H24 |
| XGBoost + FTS forecast + membership |       0.2141 |        0.3567 |       36.1717 |        30.1467 |      3.3014 |       4.1301 |      24.1593 |       23.0355 |        20 | Fixed-origin direct H1-H24 |

## 6. FTS Redundancy Analysis

`fts_forecast` is strongly tied to lag features. Pearson correlation with `lag_24` is 0.3178.

| feature         |   pearson_with_fts |   spearman_with_fts |   mutual_information_with_fts |
|:----------------|-------------------:|--------------------:|------------------------------:|
| fts_forecast    |             1      |              1      |                      nan      |
| lag_1           |             0.9302 |              0.9579 |                        2.676  |
| lag_24          |             0.3178 |              0.4755 |                        0.1953 |
| lag_168         |             0.3052 |              0.4415 |                        0.1548 |
| rolling_mean_3  |             0.8213 |              0.8431 |                        0.8317 |
| rolling_mean_24 |             0.247  |              0.24   |                        0.1507 |

## 7. Feature Group Ablation

| model                                 |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows | Notes                      |
|:--------------------------------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|----------:|:---------------------------|
| Calendar only                         |       0.2275 |        0.3652 |       38.3486 |        32.1877 |      3.4838 |       4.3354 |      25.2805 |       24.1728 |        20 | Fixed-origin direct H1-H24 |
| Lag only                              |       0.2411 |        0.385  |       43.6681 |        34.9393 |      3.6601 |       4.4298 |      27.3743 |       25.6095 |        20 | Fixed-origin direct H1-H24 |
| Lag + Calendar                        |       0.2132 |        0.3593 |       35.4384 |        29.7052 |      3.395  |       4.2563 |      24.3881 |       23.6486 |        20 | Fixed-origin direct H1-H24 |
| Lag + Calendar + Rolling              |       0.2133 |        0.3558 |       36.1312 |        30.0461 |      3.2899 |       4.1539 |      24.09   |       22.9296 |        20 | Fixed-origin direct H1-H24 |
| Lag + Calendar + Rolling + FTS        |       0.2138 |        0.3564 |       36.4537 |        30.1357 |      3.3414 |       4.1816 |      24.6635 |       23.2709 |        20 | Fixed-origin direct H1-H24 |
| Lag + Calendar + Rolling + Membership |       0.2148 |        0.3578 |       36.2901 |        30.2062 |      3.3153 |       4.1533 |      24.2486 |       23.1224 |        20 | Fixed-origin direct H1-H24 |
| Residual FTS-XGBoost                  |       0.2255 |        0.3696 |       37.5247 |        38.5076 |      3.2989 |       4.7159 |      20.6606 |       23.3278 |        20 | Residual FTS-XGBoost       |

## 8. Rolling-Origin Evaluation

Rolling-origin audit uses forecast origins every 12 hours in the test period, giving more windows than Stage 2.

| model                |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows | Notes                                                    |
|:---------------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|----------:|:---------------------------------------------------------|
| Hybrid feature-based |       0.2169 |        0.3647 |       35.9373 |        30.3948 |      3.242  |       3.9933 |      23.5019 |       22.716  |        40 | Rolling origins every 12 hours with Stage 3 fixed models |
| XGBoost Direct       |       0.2178 |        0.363  |       36.0356 |        30.5215 |      3.2544 |       3.9595 |      23.6408 |       22.8423 |        40 | Rolling origins every 12 hours with Stage 3 fixed models |
| Residual FTS-XGBoost |       0.234  |        0.3822 |       38.4212 |        41.2194 |      3.3661 |       4.8654 |      20.7318 |       24.2179 |        40 | Rolling origins every 12 hours with Stage 3 fixed models |
| Weekly Seasonal      |       0.2491 |        0.4666 |       38.7907 |        32.1658 |      3.7188 |       5.2575 |      26.0014 |       25.2311 |        40 | Rolling origins every 12 hours with Stage 3 fixed models |

## 9. Stability Analysis

| model                |   mean |   median |    std |    min |     max |     Q1 |     Q3 |
|:---------------------|-------:|---------:|-------:|-------:|--------:|-------:|-------:|
| Hybrid feature-based | 3.242  |   2.6371 | 2.3613 | 0.0859 |  8.9291 | 1.6838 | 3.9447 |
| Residual FTS-XGBoost | 3.3661 |   1.9294 | 3.5579 | 0.0627 | 12.4795 | 0.9382 | 4.2966 |
| Weekly Seasonal      | 3.7188 |   2.48   | 3.7638 | 0.03   | 15.02   | 0.6825 | 5.52   |
| XGBoost Direct       | 3.2544 |   2.655  | 2.2839 | 0.1896 |  8.7335 | 1.7607 | 3.8689 |

Pairwise comparison:

| comparison                                |   model_a_wins |   model_b_wins |   ties |   mean_error_difference |   wilcoxon_stat |   wilcoxon_p |   n |
|:------------------------------------------|---------------:|---------------:|-------:|------------------------:|----------------:|-------------:|----:|
| XGBoost Direct minus Weekly Seasonal      |             21 |             19 |      0 |                 -0.4643 |             359 |       0.5014 |  40 |
| Hybrid feature-based minus XGBoost Direct |             19 |             21 |      0 |                 -0.0125 |             394 |       0.8368 |  40 |
| Residual FTS-XGBoost minus XGBoost Direct |             21 |             19 |      0 |                  0.1116 |             382 |       0.7146 |  40 |

## 10. Error Regime Analysis

| model                | consumption_regime   |   mean |   median |   count |
|:---------------------|:---------------------|-------:|---------:|--------:|
| Hybrid feature-based | Low                  | 3.2178 |   2.9383 |      14 |
| Hybrid feature-based | Medium               | 1.7975 |   1.5086 |      12 |
| Hybrid feature-based | High                 | 4.5042 |   3.3518 |      14 |
| Residual FTS-XGBoost | Low                  | 1.2264 |   1.0551 |      14 |
| Residual FTS-XGBoost | Medium               | 1.6453 |   1.3492 |      12 |
| Residual FTS-XGBoost | High                 | 6.9807 |   6.5343 |      14 |
| Weekly Seasonal      | Low                  | 2.71   |   1.21   |      14 |
| Weekly Seasonal      | Medium               | 2.33   |   1.13   |      12 |
| Weekly Seasonal      | High                 | 5.9179 |   5.63   |      14 |
| XGBoost Direct       | Low                  | 3.2535 |   3.4268 |      14 |
| XGBoost Direct       | Medium               | 1.7852 |   1.4262 |      12 |
| XGBoost Direct       | High                 | 4.5148 |   3.4782 |      14 |

## 11. Final Hybrid Decision

1. The H1-H24 pattern is not caused by a timestamp alignment bug.
2. H24 is strong because same-hour previous-day information is highly predictive; H24 top built-in XGBoost feature is `hour`.
3. Stage 2 environmental comparison needed clearer current-vs-oracle separation; Stage 3 fixes that split.
4. Environmental features do not provide reliable improvement in the corrected audit.
5. Residual FTS-XGBoost does not beat the best XGBoost standalone result consistently.
6. Membership degree features are not materially better than hard fuzzy state/FTS forecast.
7. FTS is largely redundant with lag/rolling features, especially daily lag behavior.
8. Rolling-origin results do not show stable fuzzy-hybrid improvement.
9. Consumption-regime results should be read as diagnostic because this is still one-house data.
10. Based on Stage 3, FTS is not strong enough to retain as the main proposed method for this dataset. A better research framing is XGBoost/direct multi-horizon forecasting as the main method, with FTS reported as a tested but non-improving fuzzy augmentation.

Best overall row in Stage 3 feature ablation: `Lag + Calendar + Rolling` with Daily MAE 3.2899 kWh. Best fuzzy-related row in the feature ablation is Residual FTS-XGBoost with Daily MAE 3.2989 kWh.
