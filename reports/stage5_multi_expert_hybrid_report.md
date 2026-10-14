# Stage 5 Multi-Expert Hybrid Report

## 1. Stage 4 Starting Point

Stage 4 found XGBoost Direct as the most defensible main model, while Threshold Gate B had a small Daily MAE gain. Stage 5 tests whether future regime prediction can make multi-expert FTS useful without using actual future target values.

## 2. Regime Definition

| scheme     | class   |   lower |   upper |   count_train |
|:-----------|:--------|--------:|--------:|--------------:|
| A_3_regime | Low     |    0.17 |    0.31 |           810 |
| A_3_regime | Normal  |    0.31 |    0.81 |          1034 |
| A_3_regime | High    |    0.81 |    3.65 |           458 |
| B_spike    | Normal  |    0.17 |    0.81 |          1844 |
| B_spike    | High    |    0.81 |    1.7  |           343 |
| B_spike    | Extreme |    1.7  |    3.65 |           115 |

## 3. Regime Prediction Performance

Baseline detector from XGBoost predictions:

| class   |   precision |   recall |     f1 |   support |   accuracy |   balanced_accuracy | scheme     |
|:--------|------------:|---------:|-------:|----------:|-----------:|--------------------:|:-----------|
| Low     |      0.2962 |   0.2035 | 0.2412 |      1622 |   nan      |            nan      | A_3_regime |
| Normal  |      0.7378 |   0.7361 | 0.737  |      7932 |   nan      |            nan      | A_3_regime |
| High    |      0.377  |   0.4919 | 0.4268 |      1726 |   nan      |            nan      | A_3_regime |
| overall |      0.4703 |   0.4772 | 0.4683 |     11280 |     0.6222 |              0.4772 | A_3_regime |
| Normal  |      0.9029 |   0.8532 | 0.8773 |      9554 |   nan      |            nan      | B_spike    |
| High    |      0.2492 |   0.4247 | 0.3141 |      1321 |   nan      |            nan      | B_spike    |
| Extreme |      0      |   0      | 0      |       405 |   nan      |            nan      | B_spike    |
| overall |      0.384  |   0.4259 | 0.3971 |     11280 |     0.7723 |              0.4259 | B_spike    |

Classifier comparison:

| classifier          |   accuracy |   balanced_accuracy |   macro_f1 |   high_recall |   high_precision |
|:--------------------|-----------:|--------------------:|-----------:|--------------:|-----------------:|
| Decision Tree       |     0.5852 |              0.5147 |     0.4862 |        0.5313 |           0.3659 |
| Logistic Regression |     0.5875 |              0.5028 |     0.4799 |        0.5116 |           0.3719 |
| Random Forest       |     0.5233 |              0.5454 |     0.4732 |        0.5985 |           0.3534 |

## 4. Spike Detection Performance

| target   |   accuracy |   balanced_accuracy |   spike_precision |   spike_recall |   spike_f1 |   false_negative_rate |
|:---------|-----------:|--------------------:|------------------:|---------------:|-----------:|----------------------:|
| spike80  |     0.751  |              0.708  |            0.3366 |         0.646  |     0.4425 |                0.354  |
| spike90  |     0.8129 |              0.7116 |            0.2239 |         0.5921 |     0.3249 |                0.4079 |

## 5. Expert Performance

Expert complementarity:

| actual_regime   | expert               |   win_count |   win_rate |   mean_abs_error |   mean_error_diff_vs_xgb |
|:----------------|:---------------------|------------:|-----------:|-----------------:|-------------------------:|
| High            | XGBoost Direct       |           9 |     0.1216 |           0.6044 |                   0      |
| High            | Residual FTS-XGBoost |           3 |     0.0405 |           0.7055 |                   0.1011 |
| High            | XGB-FTS Residual     |          62 |     0.8378 |           0.3727 |                  -0.2317 |
| Low             | XGBoost Direct       |          38 |     0.5507 |           0.1441 |                   0      |
| Low             | Residual FTS-XGBoost |          31 |     0.4493 |           0.1493 |                   0.0051 |
| Low             | XGB-FTS Residual     |           0 |     0      |           0.5046 |                   0.3605 |
| Normal          | XGBoost Direct       |         154 |     0.457  |           0.1416 |                   0      |
| Normal          | Residual FTS-XGBoost |         179 |     0.5312 |           0.1357 |                  -0.0059 |
| Normal          | XGB-FTS Residual     |           4 |     0.0119 |           0.4775 |                   0.3359 |

## 6. Specialist Retraining

| experiment                             | model                |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows | Notes   | note                                                                                                                                                     |
|:---------------------------------------|:---------------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|----------:|:--------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|
| Residual FTS-XGBoost global specialist | Residual FTS-XGBoost |       0.2255 |        0.3696 |       37.5247 |        38.5076 |      3.2989 |       4.7159 |      20.6606 |       23.3278 |        20 |         | Weighted/filter retraining skipped because non-overlap test has only 20 origins; global specialists retained to avoid unstable tiny-specialist training. |
| XGB-FTS Residual global specialist     | XGB-FTS Residual     |       0.4653 |        0.5217 |      108.852  |        65.3522 |      9.0155 |       9.9251 |      79.0627 |       52.6465 |        20 |         | Weighted/filter retraining skipped because non-overlap test has only 20 origins; global specialists retained to avoid unstable tiny-specialist training. |

## 7. Three-Expert Gate

Three-expert hard gate is included in the final comparison table.

## 8. Probability Gate

| model            |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows |   Notes |
|:-----------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|----------:|--------:|
| Probability Gate |       0.2526 |        0.4046 |       46.5502 |        34.1195 |      3.9975 |       4.8688 |      32.2559 |       27.0431 |        20 |     nan |

## 9. Soft Mixture-of-Experts

| model                          |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows |   Notes |
|:-------------------------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|----------:|--------:|
| Soft Regime Mixture-of-Experts |       0.2613 |        0.3871 |       49.8015 |        37.6834 |      3.8609 |       4.6143 |      31.0451 |       26.4283 |        20 |     nan |

## 10. Oracle Upper Bound

| model               |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows | Notes                           |
|:--------------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|----------:|:--------------------------------|
| Oracle Three-Expert |       0.1296 |        0.2352 |       23.3836 |        19.4573 |      2.0228 |       2.6014 |      15.3157 |        14.355 |        20 | Upper bound using actual target |

## 11. High/Extreme Performance

| model                          | regime   |    MAE |    bias |   n |
|:-------------------------------|:---------|-------:|--------:|----:|
| Daily-Optimized Regime Gate    | Extreme  | 1.2467 | -1.2467 |  17 |
| Daily-Optimized Regime Gate    | High     | 0.4128 | -0.3953 |  57 |
| Oracle Three-Expert            | Extreme  | 0.8463 | -0.8463 |  17 |
| Oracle Three-Expert            | High     | 0.1722 | -0.1075 |  57 |
| Probability Gate               | Extreme  | 1.0179 | -1.0179 |  17 |
| Probability Gate               | High     | 0.3775 | -0.2325 |  57 |
| Residual FTS-XGBoost           | Extreme  | 1.3689 | -1.3689 |  17 |
| Residual FTS-XGBoost           | High     | 0.5076 | -0.5014 |  57 |
| Soft Regime Mixture-of-Experts | Extreme  | 1.0152 | -1.0152 |  17 |
| Soft Regime Mixture-of-Experts | High     | 0.3407 | -0.2408 |  57 |
| Stage 4 Threshold Gate B       | Extreme  | 1.2467 | -1.2467 |  17 |
| Stage 4 Threshold Gate B       | High     | 0.4128 | -0.3953 |  57 |
| Three-Expert Hard Gate         | Extreme  | 0.9638 | -0.9638 |  17 |
| Three-Expert Hard Gate         | High     | 0.3775 | -0.2325 |  57 |
| Weekly Seasonal Naive          | Extreme  | 1.3765 | -1.3765 |  17 |
| Weekly Seasonal Naive          | High     | 0.5581 | -0.4107 |  57 |
| XGB-FTS Residual               | Extreme  | 0.8463 | -0.8463 |  17 |
| XGB-FTS Residual               | High     | 0.2315 | -0.0167 |  57 |
| XGBoost Direct                 | Extreme  | 1.2467 | -1.2467 |  17 |
| XGBoost Direct                 | High     | 0.4128 | -0.3953 |  57 |

## 12. Daily Total Forecast Performance

| model                          |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows | Notes   |
|:-------------------------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|----------:|:--------|
| Oracle Three-Expert            |       0.1296 |        0.2352 |       23.3836 |        19.4573 |      2.0228 |       2.6014 |      15.3157 |       14.355  |        20 |         |
| Stage 4 Threshold Gate B       |       0.22   |        0.3574 |       38.1847 |        32.8063 |      3.2028 |       4.1016 |      23.2123 |       22.2971 |        20 |         |
| Daily-Optimized Regime Gate    |       0.22   |        0.3574 |       38.1847 |        32.8063 |      3.2028 |       4.1016 |      23.2123 |       22.2971 |        20 |         |
| XGBoost Direct                 |       0.2133 |        0.3558 |       36.1312 |        30.0461 |      3.2899 |       4.1539 |      24.09   |       22.9296 |        20 |         |
| Residual FTS-XGBoost           |       0.2255 |        0.3696 |       37.5247 |        38.5076 |      3.2989 |       4.7159 |      20.6606 |       23.3278 |        20 |         |
| Weekly Seasonal Naive          |       0.2498 |        0.4687 |       39.3266 |        32.0723 |      3.777  |       5.7126 |      26.5959 |       24.7501 |        20 |         |
| Soft Regime Mixture-of-Experts |       0.2613 |        0.3871 |       49.8015 |        37.6834 |      3.8609 |       4.6143 |      31.0451 |       26.4283 |        20 |         |
| Three-Expert Hard Gate         |       0.2614 |        0.4023 |       49.6361 |        37.9343 |      3.8665 |       4.7697 |      30.9732 |       26.2079 |        20 |         |
| Probability Gate               |       0.2526 |        0.4046 |       46.5502 |        34.1195 |      3.9975 |       4.8688 |      32.2559 |       27.0431 |        20 |         |
| XGB-FTS Residual               |       0.4653 |        0.5217 |      108.852  |        65.3522 |      9.0155 |       9.9251 |      79.0627 |       52.6465 |        20 |         |

## 13. Stability

| model                          |   mean |   median |    std |     Q1 |      Q3 |     max |   wins_vs_xgb |
|:-------------------------------|-------:|---------:|-------:|-------:|--------:|--------:|--------------:|
| Daily-Optimized Regime Gate    | 3.2028 |   2.3137 | 2.6288 | 1.7357 |  3.7527 |  8.7218 |            11 |
| Oracle Three-Expert            | 2.0228 |   1.4006 | 1.6782 | 0.9419 |  3.2083 |  5.4252 |            20 |
| Probability Gate               | 3.9975 |   3.3244 | 2.8517 | 1.4639 |  6.0655 |  9.7452 |             7 |
| Residual FTS-XGBoost           | 3.2989 |   2.1566 | 3.4576 | 1.0667 |  3.9505 | 11.015  |            10 |
| Soft Regime Mixture-of-Experts | 3.8609 |   3.6135 | 2.5927 | 1.6789 |  5.6104 |  9.1099 |             6 |
| Stage 4 Threshold Gate B       | 3.2028 |   2.3137 | 2.6288 | 1.7357 |  3.7527 |  8.7218 |            11 |
| Three-Expert Hard Gate         | 3.8665 |   3.0846 | 2.8654 | 1.5584 |  5.8062 |  9.7452 |             8 |
| Weekly Seasonal Naive          | 3.777  |   1.895  | 4.3971 | 0.5025 |  5.5025 | 15.02   |             9 |
| XGB-FTS Residual               | 9.0155 |  10.2145 | 4.2586 | 7.3042 | 11.845  | 15.0474 |             3 |
| XGBoost Direct                 | 3.2899 |   2.5729 | 2.602  | 1.8279 |  3.8689 |  8.7335 |             0 |

Pairwise:

| comparison                                    |   gate_wins |   xgb_wins |   ties |   mean_difference |   median_difference |   wilcoxon_p |   n |
|:----------------------------------------------|------------:|-----------:|-------:|------------------:|--------------------:|-------------:|----:|
| XGBoost Direct minus Stage 4 Threshold Gate B |          11 |          4 |      5 |             0.087 |              0.0113 |       0.0609 |  20 |

## 14. Complexity vs Benefit

| Model                          |   Daily MAE |   Improvement vs XGB |   Number of Models | Gate Needed   | Complexity   |
|:-------------------------------|------------:|---------------------:|-------------------:|:--------------|:-------------|
| Oracle Three-Expert            |      2.0228 |              38.5138 |                  3 | Oracle        | Theoretical  |
| Stage 4 Threshold Gate B       |      3.2028 |               2.646  |                  3 | Yes           | High         |
| Daily-Optimized Regime Gate    |      3.2028 |               2.646  |                  3 | Yes           | High         |
| XGBoost Direct                 |      3.2899 |               0      |                  1 | No            | Low          |
| Residual FTS-XGBoost           |      3.2989 |              -0.2742 |                  2 | No            | Medium       |
| Weekly Seasonal Naive          |      3.777  |             -14.8077 |                  1 | No            | Low          |
| Soft Regime Mixture-of-Experts |      3.8609 |             -17.3569 |                  3 | Yes           | High         |
| Three-Expert Hard Gate         |      3.8665 |             -17.5269 |                  3 | Yes           | High         |
| Probability Gate               |      3.9975 |             -21.5095 |                  3 | Yes           | High         |
| XGB-FTS Residual               |      9.0155 |            -174.04   |                  2 | No            | Medium       |

## 15. Final Proposed Method Decision

1. Future regime prediction is only moderately reliable; high/spike recall remains the limiting factor.
2. High/Extreme consumption can be detected better than random, but not reliably enough to trust a complex gate.
3. Low regime does not show a strong enough FTS advantage to justify a dedicated core expert.
4. Normal is best handled by XGBoost Direct.
5. High/Extreme has some benefit from XGB-FTS residual in bias terms, but it destroys overall performance when used too broadly.
6. Three-expert Oracle is an upper bound and should not be treated as deployable.
7. Realistic three-expert gates do not clearly beat the simpler Stage 4 Threshold Gate B.
8. Probability gating and soft mixture do not provide a robust improvement over XGBoost.
9. Spike underestimation is reduced by the high specialist, but at a large overall error cost.
10. The best realistic Stage 5 model is `Stage 4 Threshold Gate B` with Daily MAE 3.2028 kWh, an improvement of 2.65% vs XGBoost Direct.
11. The >=2% rule is met only by selective/gated variants, but stability and complexity evidence is not strong enough.
12. FTS-based experts are used by gates, but not with enough reliable benefit.
13. Complexity is not justified as the core method.
14. Based on Stage 1-5, FTS should not be a core proposed component. Final proposed method candidate: Direct multi-horizon XGBoost, with FTS retained as comparative/exploratory fuzzy analysis. Revisi metodologi/judul diperlukan if the original title requires FTS as the main contribution.
