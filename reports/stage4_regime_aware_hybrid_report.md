# Stage 4 Regime-Aware Hybrid Report

## 1. Reverse Residual Experiment

| model                           |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows | Notes                                                     |
|:--------------------------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|----------:|:----------------------------------------------------------|
| XGBoost-FTS Residual Correction |       0.4653 |        0.5217 |       108.852 |        65.3522 |      9.0155 |       9.9251 |      79.0627 |       52.6465 |        20 | XGBoost forecast plus FTS correction on XGBoost residuals |

## 2. Non-Overlapping Evaluation Setup

Stage 4 uses non-overlapping 24-hour forecast origins in the test period. Number of origins: 20.

## 3. Consumption Regime Definition

| scheme   | regime   |   lower |   upper |
|:---------|:---------|--------:|--------:|
| tertile  | Low      |    0.17 |    0.31 |
| tertile  | Medium   |    0.31 |    0.55 |
| tertile  | High     |    0.55 |    3.65 |
| spike    | Normal   |    0.17 |    0.81 |
| spike    | High     |    0.81 |    1.7  |
| spike    | Extreme  |    1.7  |    3.65 |

## 4. Expert Performance by Regime

| model                     | regime   |    MAE |   RMSE |    MAPE |   sMAPE |   n |
|:--------------------------|:---------|-------:|-------:|--------:|--------:|----:|
| Feature-based FTS-XGBoost | High     | 0.3749 | 0.5466 | 31.4683 | 36.269  | 152 |
| Feature-based FTS-XGBoost | Low      | 0.1462 | 0.2352 | 50.61   | 32.0289 |  69 |
| Feature-based FTS-XGBoost | Medium   | 0.1374 | 0.2127 | 35.6081 | 26.0318 | 259 |
| Residual FTS-XGBoost      | High     | 0.4088 | 0.6022 | 32.6479 | 41.596  | 152 |
| Residual FTS-XGBoost      | Low      | 0.1493 | 0.1923 | 50.7532 | 43.2705 |  69 |
| Residual FTS-XGBoost      | Medium   | 0.1383 | 0.1747 | 36.8625 | 35.4263 | 259 |
| XGBoost Direct            | High     | 0.3764 | 0.5461 | 31.6517 | 36.5278 | 152 |
| XGBoost Direct            | Low      | 0.1441 | 0.2312 | 49.8578 | 31.7176 |  69 |
| XGBoost Direct            | Medium   | 0.1361 | 0.2131 | 35.1032 | 25.7969 | 259 |

## 5. Oracle Gate Potential

| model       |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows | Notes                                                 |   Daily MAE improvement vs XGBoost (%) |
|:------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|----------:|:------------------------------------------------------|---------------------------------------:|
| Oracle Gate |       0.1689 |        0.3235 |       25.8468 |        23.3488 |      2.8472 |       3.8411 |      19.3002 |       19.7077 |        20 | Upper bound; uses actual target to choose best expert |                                13.4559 |

## 6. Threshold Gate

| model                                     |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows | Notes                                        |
|:------------------------------------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|----------:|:---------------------------------------------|
| Gate A Low/Medium predicted -> Residual   |       0.2406 |        0.3646 |       43.5139 |        40.5894 |      3.243  |       4.2677 |      22.2603 |       22.7373 |        20 | Threshold gate using XGBoost predicted value |
| Gate B Low predicted -> Residual          |       0.22   |        0.3574 |       38.1847 |        32.8063 |      3.2028 |       4.1016 |      23.2123 |       22.2971 |        20 | Threshold gate using XGBoost predicted value |
| Gate C Normal predicted -> Feature Hybrid |       0.2152 |        0.3577 |       36.7138 |        30.2803 |      3.3584 |       4.199  |      24.8335 |       23.4014 |        20 | Threshold gate using XGBoost predicted value |

Gate decision frequency:

| gate                                      | chosen_expert        |   n |   share |
|:------------------------------------------|:---------------------|----:|--------:|
| Gate A Low/Medium predicted -> Residual   | Residual FTS-XGBoost | 246 |  0.5125 |
| Gate A Low/Medium predicted -> Residual   | XGBoost Direct       | 234 |  0.4875 |
| Gate B Low predicted -> Residual          | Residual FTS-XGBoost |  33 |  0.0688 |
| Gate B Low predicted -> Residual          | XGBoost Direct       | 447 |  0.9312 |
| Gate C Normal predicted -> Feature Hybrid | Hybrid feature-based | 385 |  0.8021 |
| Gate C Normal predicted -> Feature Hybrid | XGBoost Direct       |  95 |  0.1979 |

## 7. Learned Gate

| model        |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows | Notes                                                    |   gate_accuracy |   gate_balanced_accuracy |
|:-------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|----------:|:---------------------------------------------------------|----------------:|-------------------------:|
| Learned Gate |       0.2299 |        0.4274 |       31.7093 |        30.0683 |      3.8274 |       5.0527 |      25.1756 |       26.1657 |         9 | DecisionTree gate evaluated on later non-overlap origins |          0.4861 |                   0.4599 |

## 8. High / Extreme Consumption Analysis

| model                           | regime   |    MAE |    bias |   n |
|:--------------------------------|:---------|-------:|--------:|----:|
| Feature-based FTS-XGBoost       | Extreme  | 1.2476 | -1.2476 |  17 |
| Feature-based FTS-XGBoost       | High     | 0.4084 | -0.3924 |  57 |
| Residual FTS-XGBoost            | Extreme  | 1.3689 | -1.3689 |  17 |
| Residual FTS-XGBoost            | High     | 0.5076 | -0.5014 |  57 |
| Weekly Seasonal Naive           | Extreme  | 1.3765 | -1.3765 |  17 |
| Weekly Seasonal Naive           | High     | 0.5581 | -0.4107 |  57 |
| XGBoost Direct                  | Extreme  | 1.2467 | -1.2467 |  17 |
| XGBoost Direct                  | High     | 0.4128 | -0.3953 |  57 |
| XGBoost-FTS Residual Correction | Extreme  | 0.8463 | -0.8463 |  17 |
| XGBoost-FTS Residual Correction | High     | 0.2315 | -0.0167 |  57 |

## 9. Bias Analysis

| model                           | regime   |    bias |   n |
|:--------------------------------|:---------|--------:|----:|
| Feature-based FTS-XGBoost       | overall  |  0.0099 | 480 |
| Residual FTS-XGBoost            | overall  | -0.0907 | 480 |
| Weekly Seasonal Naive           | overall  | -0.0268 | 480 |
| XGBoost Direct                  | overall  |  0.0067 | 480 |
| XGBoost-FTS Residual Correction | overall  |  0.3756 | 480 |
| Feature-based FTS-XGBoost       | High     | -0.2337 | 152 |
| Feature-based FTS-XGBoost       | Low      |  0.142  |  69 |
| Feature-based FTS-XGBoost       | Medium   |  0.1176 | 259 |
| Residual FTS-XGBoost            | High     | -0.3457 | 152 |
| Residual FTS-XGBoost            | Low      |  0.0465 |  69 |
| Residual FTS-XGBoost            | Medium   |  0.0224 | 259 |
| Weekly Seasonal Naive           | High     | -0.2928 | 152 |
| Weekly Seasonal Naive           | Low      |  0.1238 |  69 |
| Weekly Seasonal Naive           | Medium   |  0.0891 | 259 |
| XGBoost Direct                  | High     | -0.2369 | 152 |
| XGBoost Direct                  | Low      |  0.1406 |  69 |
| XGBoost Direct                  | Medium   |  0.114  | 259 |
| XGBoost-FTS Residual Correction | High     |  0.1418 | 152 |
| XGBoost-FTS Residual Correction | Low      |  0.5046 |  69 |
| XGBoost-FTS Residual Correction | Medium   |  0.4785 | 259 |
| Feature-based FTS-XGBoost       | Extreme  | -1.2476 |  17 |
| Feature-based FTS-XGBoost       | High     | -0.3924 |  57 |
| Feature-based FTS-XGBoost       | Normal   |  0.119  | 406 |
| Residual FTS-XGBoost            | Extreme  | -1.3689 |  17 |
| Residual FTS-XGBoost            | High     | -0.5014 |  57 |
| Residual FTS-XGBoost            | Normal   |  0.0205 | 406 |
| Weekly Seasonal Naive           | Extreme  | -1.3765 |  17 |
| Weekly Seasonal Naive           | High     | -0.4107 |  57 |
| Weekly Seasonal Naive           | Normal   |  0.0836 | 406 |
| XGBoost Direct                  | Extreme  | -1.2467 |  17 |
| XGBoost Direct                  | High     | -0.3953 |  57 |
| XGBoost Direct                  | Normal   |  0.1156 | 406 |
| XGBoost-FTS Residual Correction | Extreme  | -0.8463 |  17 |
| XGBoost-FTS Residual Correction | High     | -0.0167 |  57 |
| XGBoost-FTS Residual Correction | Normal   |  0.4819 | 406 |

## 10. Final Model Comparison

| model                           |   Hourly MAE |   Hourly RMSE |   Hourly MAPE |   Hourly sMAPE |   Daily MAE |   Daily RMSE |   Daily MAPE |   Daily sMAPE |   windows | Notes   |
|:--------------------------------|-------------:|--------------:|--------------:|---------------:|------------:|-------------:|-------------:|--------------:|----------:|:--------|
| Oracle Gate                     |       0.1689 |        0.3235 |       25.8468 |        23.3488 |      2.8472 |       3.8411 |      19.3002 |       19.7077 |        20 |         |
| Best Threshold Gated Hybrid     |       0.22   |        0.3574 |       38.1847 |        32.8063 |      3.2028 |       4.1016 |      23.2123 |       22.2971 |        20 |         |
| XGBoost Direct                  |       0.2133 |        0.3558 |       36.1312 |        30.0461 |      3.2899 |       4.1539 |      24.09   |       22.9296 |        20 |         |
| Residual FTS-XGBoost            |       0.2255 |        0.3696 |       37.5247 |        38.5076 |      3.2989 |       4.7159 |      20.6606 |       23.3278 |        20 |         |
| Feature-based FTS-XGBoost       |       0.2138 |        0.3564 |       36.4537 |        30.1357 |      3.3414 |       4.1816 |      24.6635 |       23.2709 |        20 |         |
| Weekly Seasonal Naive           |       0.2498 |        0.4687 |       39.3266 |        32.0723 |      3.777  |       5.7126 |      26.5959 |       24.7501 |        20 |         |
| Best Learned Gated Hybrid       |       0.2299 |        0.4274 |       31.7093 |        30.0683 |      3.8274 |       5.0527 |      25.1756 |       26.1657 |         9 |         |
| XGBoost-FTS Residual Correction |       0.4653 |        0.5217 |      108.852  |        65.3522 |      9.0155 |       9.9251 |      79.0627 |       52.6465 |        20 |         |

## 11. Stability Across Origins

| model                           |   mean |   median |    std |     Q1 |      Q3 |     max |   win_count |
|:--------------------------------|-------:|---------:|-------:|-------:|--------:|--------:|------------:|
| Best Learned Gated Hybrid       | 3.8274 |   2.7135 | 3.4988 | 1.1739 |  6.5749 | 10.1856 |           1 |
| Best Threshold Gated Hybrid     | 3.2028 |   2.3137 | 2.6288 | 1.7357 |  3.7527 |  8.7218 |           1 |
| Feature-based FTS-XGBoost       | 3.3414 |   2.7556 | 2.5794 | 1.9339 |  3.9447 |  8.4831 |           2 |
| Oracle Gate                     | 2.8472 |   2.0634 | 2.6453 | 1.1004 |  2.8154 |  8.8935 |           0 |
| Residual FTS-XGBoost            | 3.2989 |   2.1566 | 3.4576 | 1.0667 |  3.9505 | 11.015  |           5 |
| Weekly Seasonal Naive           | 3.777  |   1.895  | 4.3971 | 0.5025 |  5.5025 | 15.02   |           6 |
| XGBoost Direct                  | 3.2899 |   2.5729 | 2.602  | 1.8279 |  3.8689 |  8.7335 |           2 |
| XGBoost-FTS Residual Correction | 9.0155 |  10.2145 | 4.2586 | 7.3042 | 11.845  | 15.0474 |           3 |

## 12. Pairwise Comparison

| comparison                                           |   a_wins |   b_wins |   ties |   mean_difference |   median_difference |   wilcoxon_p |   n |
|:-----------------------------------------------------|---------:|---------:|-------:|------------------:|--------------------:|-------------:|----:|
| XGBoost Direct minus Best Threshold Gated Hybrid     |        4 |       11 |      5 |            0.087  |              0.0113 |       0.0609 |  20 |
| XGBoost Direct minus Best Learned Gated Hybrid       |        6 |        3 |      0 |           -0.3387 |             -0.3818 |     nan      |   9 |
| XGBoost Direct minus XGBoost-FTS Residual Correction |       17 |        3 |      0 |           -5.7257 |             -8.8544 |       0.0009 |  20 |

## 13. Is FTS Still Worth Keeping?

Oracle Gate Daily MAE is 2.8472 kWh, so there is theoretical complementarity between experts. The best threshold gate is `Gate B Low predicted -> Residual` with Daily MAE 3.2028 kWh (2.65% vs XGBoost). The learned gate has Daily MAE 3.8274 kWh (-16.34% vs XGBoost).

The practical rule required improvement above 2%, majority wins, and no substantial worsening on High/Extreme consumption. The gated hybrids do not satisfy that standard strongly enough.

## 14. Stage 4 Conclusion

1. XGBoost residual is not reliably improved by FTS residual correction.
2. FTS advantages on Low/Medium regimes exist in some tables but are not stable enough across non-overlap origins.
3. FTS-based models are risky for High/Extreme consumption, especially residual FTS-XGBoost on High regimes.
4. Oracle Gate shows an upper bound, but realistic gates recover little of it.
5. Threshold gate improvement vs XGBoost is 2.65%.
6. Learned gate improvement vs XGBoost is -16.34%.
7. Gate frequency table above shows whether FTS is actually used.
8. Improvement does not clearly survive non-overlap evaluation.
9. Hybrid methods are not more stable than XGBoost by enough to justify added complexity.
10. Based on Stage 1-4, FTS should not be retained as the core proposed method. Use XGBoost Direct as the main forecasting method and keep FTS as a comparative/exploratory fuzzy analysis.

Best Stage 4 model by Daily MAE: `Oracle Gate` with Daily MAE 2.8472 kWh.
