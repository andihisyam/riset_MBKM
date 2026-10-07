# Preliminary Experiment Report

## 1. Dataset Characteristics

Dataset berisi 19,735 observasi dan 29 kolom. Periode data dimulai dari 2016-01-11T17:00:00 sampai 2016-05-27T18:00:00. Interval paling umum adalah 0 days 00:10:00. Duplicate row = 0, duplicate timestamp = 0.

Target `Appliances` memiliki median 60.00 Wh per 10 menit, mean 97.69 Wh, maksimum 1080.00 Wh, dan skewness 3.39. Ini mengonfirmasi distribusi right-skewed. Batas IQR memberi 2,138 kandidat spike (10.83%), tetapi spike tidak dihapus karena secara domain dapat merepresentasikan penggunaan appliance nyata.

`rv1` dan `rv2` identik: True. Korelasi Pearson `rv1` terhadap target adalah -0.0111, dan `rv2` adalah -0.0111. Keduanya dikeluarkan dari proposed model.

## 2. Forecasting Problem Definition

Target praktis penelitian adalah memprediksi total konsumsi energi 24 jam berikutnya. Pada resolusi hourly, target model adalah total energi per jam dalam Wh, dengan output ringkas harian dihitung sebagai jumlah 24 prediksi hourly dan dikonversi ke kWh.

## 3. Why Hourly / Why 10-minute

Agregasi hourly menjumlahkan enam interval `Appliances` 10 menit menjadi energi per jam. Fitur temperatur, humidity, tekanan, angin, visibility, dew point, `rv1`, dan `rv2` diagregasi dengan mean. Jumlah observasi hourly menjadi 3,290, dibanding 19,735 observasi 10 menit.

Keputusan sementara: hourly lebih selaras dengan target 24-hour dan 7-day forecasting karena horizon 24 jam hanya memerlukan 24 langkah, sedangkan data 10 menit memerlukan 144 langkah. Namun keputusan final tetap harus dibandingkan dengan model ML/FTS, bukan hanya baseline.

## 4. Temporal Analysis

Autocorrelation lag penting pada data 10 menit:

|          |        0 |
|:---------|---------:|
| lag_1    | 0.753166 |
| lag_6    | 0.323661 |
| lag_12   | 0.310939 |
| lag_144  | 0.216741 |
| lag_288  | 0.16324  |
| lag_1008 | 0.222025 |

Interpretasi awal: lag pendek masih kuat, dan lag 144 menunjukkan pola harian. Lag 1008 memberi indikasi pola mingguan, tetapi perlu hati-hati karena dataset hanya sekitar 4,5 bulan.

## 5. Stationarity

ADF p-value = 0.000000. KPSS p-value = 0.1. Hasil ini tidak otomatis berarti differencing harus diterapkan. Untuk XGBoost dan FTS, fitur lag/seasonal serta validasi kronologis lebih relevan daripada memaksa differencing di awal.

## 6. Predictor Notes

Beberapa pasangan temperatur/humidity sangat berkorelasi. Ini perlu diperhatikan saat interpretasi SHAP karena fitur lingkungan yang saling berkorelasi dapat berbagi kontribusi model.

Top Pearson correlation terhadap target:

|            |         0 |
|:-----------|----------:|
| Appliances | 1         |
| lights     | 0.197278  |
| T2         | 0.120073  |
| T6         | 0.117638  |
| T_out      | 0.0991547 |
| Windspeed  | 0.0871221 |
| RH_1       | 0.0860314 |
| T3         | 0.0850599 |
| T1         | 0.0554475 |
| T4         | 0.040281  |
| T8         | 0.039572  |
| RH_3       | 0.0362917 |

## 7. Baseline Results

Baseline dihitung dengan split kronologis 70% train, 15% validation, dan 15% test. Tidak ada random split.

| resolution   | split      | model                 |      MAE |     RMSE |    MAPE |   sMAPE |   MASE |   n_evaluated |
|:-------------|:-----------|:----------------------|---------:|---------:|--------:|--------:|-------:|--------------:|
| 10min        | train      | Last Value Naive      |  30.9028 |  74.2173 | 26.4922 | 23.4064 | 0.4544 |         13813 |
| 10min        | train      | Daily Seasonal Naive  |  68.0146 | 134.416  | 72.8805 | 47.901  | 1      |         13670 |
| 10min        | train      | Weekly Seasonal Naive |  67.7058 | 133.05   | 77.7865 | 48.7108 | 0.9955 |         12806 |
| 10min        | validation | Last Value Naive      |  26.1622 |  66.4297 | 22.7143 | 20.7241 | 0.3847 |          2960 |
| 10min        | validation | Daily Seasonal Naive  |  53.3209 | 115.384  | 54.4127 | 39.3064 | 0.784  |          2960 |
| 10min        | validation | Weekly Seasonal Naive |  50.1216 | 109.386  | 52.7172 | 36.9505 | 0.7369 |          2960 |
| 10min        | test       | Last Value Naive      |  26.7376 |  66.8369 | 21.5134 | 19.7359 | 0.3931 |          2961 |
| 10min        | test       | Daily Seasonal Naive  |  52.489  | 109.473  | 48.83   | 37.122  | 0.7717 |          2961 |
| 10min        | test       | Weekly Seasonal Naive |  52.7727 | 111.302  | 49.4405 | 37.7528 | 0.7759 |          2961 |
| hourly       | train      | Last Value Naive      | 251.368  | 467.358  | 36.6273 | 32.1099 | 0.7387 |          2302 |
| hourly       | train      | Daily Seasonal Naive  | 340.285  | 605.724  | 57.5233 | 42.1279 | 1      |          2279 |
| hourly       | train      | Weekly Seasonal Naive | 336.843  | 595.609  | 62.3529 | 43.2669 | 0.9899 |          2135 |
| hourly       | validation | Last Value Naive      | 222.982  | 415.072  | 33.0265 | 29.5953 | 0.6553 |           493 |
| hourly       | validation | Daily Seasonal Naive  | 256.653  | 486.773  | 41.7088 | 34.026  | 0.7542 |           493 |
| hourly       | validation | Weekly Seasonal Naive | 229.777  | 445.718  | 37.5636 | 29.8458 | 0.6752 |           493 |
| hourly       | test       | Last Value Naive      | 188.583  | 358.358  | 26.7549 | 24.1708 | 0.5542 |           494 |
| hourly       | test       | Daily Seasonal Naive  | 271.478  | 489.471  | 41.4113 | 34.6916 | 0.7978 |           494 |
| hourly       | test       | Weekly Seasonal Naive | 257.53   | 476.987  | 39.4508 | 32.9573 | 0.7568 |           494 |

Model baseline terbaik pada test untuk tiap resolusi:

| resolution   | split   | model            |      MAE |     RMSE |    MAPE |   sMAPE |   MASE |   n_evaluated |
|:-------------|:--------|:-----------------|---------:|---------:|--------:|--------:|-------:|--------------:|
| 10min        | test    | Last Value Naive |  26.7376 |  66.8369 | 21.5134 | 19.7359 | 0.3931 |          2961 |
| hourly       | test    | Last Value Naive | 188.583  | 358.358  | 26.7549 | 24.1708 | 0.5542 |           494 |

## 8. Daily Total Energy Accuracy

Pada resolusi hourly, hourly prediction one-step rolling dijumlahkan menjadi total harian kWh untuk test set. Ini berguna sebagai diagnosis awal, tetapi bukan klaim final untuk prediksi 24 jam penuh karena model `Last Value Naive` masih memakai aktual jam sebelumnya di dalam periode target.

| model                 | split   | unit    |    MAE |   RMSE |    MAPE |   sMAPE |   MASE |   days_evaluated |
|:----------------------|:--------|:--------|-------:|-------:|--------:|--------:|-------:|-----------------:|
| Last Value Naive      | test    | kWh/day | 0.0933 | 0.1893 |  0.5696 |  0.5688 |    nan |               21 |
| Daily Seasonal Naive  | test    | kWh/day | 4.1548 | 5.8627 | 27.2329 | 27.6398 |    nan |               21 |
| Weekly Seasonal Naive | test    | kWh/day | 3.9419 | 5.7589 | 28.0405 | 26.877  |    nan |               21 |

Evaluasi yang lebih sesuai dengan target praktis adalah rolling 24-hour total forecast. Dari setiap forecast origin pada test set, baseline memprediksi 24 jam berikutnya lalu hasilnya dijumlahkan.

| model                 | split   | unit    |    MAE |   RMSE |    MAPE |   sMAPE |   MASE |   windows_evaluated |
|:----------------------|:--------|:--------|-------:|-------:|--------:|--------:|-------:|--------------------:|
| Daily Seasonal Naive  | test    | kWh/24h | 4.272  | 6.0413 | 27.9122 | 28.3717 |    nan |                  20 |
| Last Value Naive      | test    | kWh/24h | 5.9825 | 7.4435 | 39.0041 | 50.8758 |    nan |                  20 |
| Weekly Seasonal Naive | test    | kWh/24h | 3.779  | 5.7301 | 26.5396 | 24.7213 |    nan |                  20 |

## 9. Forecast Horizon Check

Baseline hourly juga dievaluasi untuk horizon 1, 6, 12, dan 24 jam.

| resolution   | split   |   horizon_hours | model                 |     MAE |    RMSE |    MAPE |   sMAPE |   MASE |   n_evaluated |
|:-------------|:--------|----------------:|:----------------------|--------:|--------:|--------:|--------:|-------:|--------------:|
| hourly       | test    |               1 | Last Value Naive      | 188.583 | 358.358 | 26.7549 | 24.1708 | 0.5542 |           494 |
| hourly       | test    |               1 | Daily Seasonal Naive  | 271.478 | 489.471 | 41.4113 | 34.6916 | 0.7978 |           494 |
| hourly       | test    |               1 | Weekly Seasonal Naive | 257.53  | 476.987 | 39.4508 | 32.9573 | 0.7568 |           494 |
| hourly       | test    |               6 | Last Value Naive      | 357.53  | 562.742 | 63.4819 | 49.4508 | 1.0507 |           494 |
| hourly       | test    |               6 | Daily Seasonal Naive  | 271.478 | 489.471 | 41.4113 | 34.6916 | 0.7978 |           494 |
| hourly       | test    |               6 | Weekly Seasonal Naive | 257.53  | 476.987 | 39.4508 | 32.9573 | 0.7568 |           494 |
| hourly       | test    |              12 | Last Value Naive      | 411.275 | 633.671 | 74.7302 | 56.0212 | 1.2086 |           494 |
| hourly       | test    |              12 | Daily Seasonal Naive  | 271.478 | 489.471 | 41.4113 | 34.6916 | 0.7978 |           494 |
| hourly       | test    |              12 | Weekly Seasonal Naive | 257.53  | 476.987 | 39.4508 | 32.9573 | 0.7568 |           494 |
| hourly       | test    |              24 | Last Value Naive      | 271.478 | 489.471 | 41.4113 | 34.6916 | 0.7978 |           494 |
| hourly       | test    |              24 | Daily Seasonal Naive  | 271.478 | 489.471 | 41.4113 | 34.6916 | 0.7978 |           494 |
| hourly       | test    |              24 | Weekly Seasonal Naive | 257.53  | 476.987 | 39.4508 | 32.9573 | 0.7568 |           494 |

## 10. Preliminary Research Conclusion

Hasil tahap awal mendukung penggunaan hourly sebagai kandidat resolusi utama karena lebih sesuai dengan target 24 jam dan tetap mempertahankan jumlah observasi yang cukup. Baseline seasonal wajib dipertahankan sebagai pembanding utama sebelum mengklaim XGBoost, FTS, atau hybrid lebih baik.

Belum ada dasar untuk menyimpulkan bahwa FTS-XGBoost adalah model terbaik. Tahap berikutnya harus menjalankan XGBoost standalone, eksperimen partition FTS, FTS standalone, lalu hybrid secara kronologis dengan validation set.

## 11. Next Experiments

1. Feature engineering hourly: lag, rolling past-only, calendar, dan cyclical encoding.
2. XGBoost historical-only sebagai model realistis.
3. Eksperimen fuzzy partition equal-width vs quantile untuk 5, 7, 9, 11, dan 15 interval.
4. Chen FTS, Weighted FTS, dan jika stabil High-Order FTS.
5. Hybrid FTS-XGBoost dan ablation study.
