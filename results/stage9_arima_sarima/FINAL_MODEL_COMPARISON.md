# Tabel Lengkap Perbandingan Semua Model Hourly

Semua 57 model disusun berurutan berdasarkan Test Daily MAE. Metrik utama penelitian tetap Daily MAE. Status Terpilih validation menunjukkan model yang dipilih dari validation, sedangkan Terbaik test (deskriptif) hanya menunjukkan nilai test terendah.

|   rank_test | model                                         | algorithm    | family   | fts_order   |   validation_daily_mae_kWh |   daily_mae_kWh |   hourly_mae_kWh | status                    |
|------------:|:----------------------------------------------|:-------------|:---------|:------------|---------------------------:|----------------:|-----------------:|:--------------------------|
|           1 | Residual FTS3-RandomForest                    | RandomForest | Residual | 3.0         |                    5.78795 |         3.0977  |         0.261135 | Terbaik test (deskriptif) |
|           2 | Hybrid FTS1-LightGBM Fixed 0.31               | LightGBM     | Hybrid   | 1.0         |                    2.74291 |         3.13545 |         0.211324 | Kandidat                  |
|           3 | Residual FTS3-LightGBM                        | LightGBM     | Residual | 3.0         |                    6.11739 |         3.13804 |         0.261427 | Kandidat                  |
|           4 | LightGBM Direct                               | LightGBM     | Direct   | -           |                    2.54532 |         3.13973 |         0.212029 | Kandidat                  |
|           5 | Hybrid FTS1-LightGBM Validation threshold     | LightGBM     | Hybrid   | 1.0         |                    2.54532 |         3.13973 |         0.212029 | Kandidat                  |
|           6 | Hybrid FTS2-LightGBM Validation threshold     | LightGBM     | Hybrid   | 2.0         |                    2.54532 |         3.13973 |         0.212029 | Kandidat                  |
|           7 | Hybrid FTS3-LightGBM Validation threshold     | LightGBM     | Hybrid   | 3.0         |                    2.54532 |         3.13973 |         0.212029 | Kandidat                  |
|           8 | Hybrid FTS3-LightGBM Fixed 0.31               | LightGBM     | Hybrid   | 3.0         |                    3.11767 |         3.14218 |         0.221575 | Kandidat                  |
|           9 | Hybrid FTS3-RandomForest Validation threshold | RandomForest | Hybrid   | 3.0         |                    2.48264 |         3.1435  |         0.206358 | Kandidat                  |
|          10 | RandomForest Direct                           | RandomForest | Direct   | -           |                    2.48264 |         3.1435  |         0.206358 | Kandidat                  |
|          11 | Hybrid FTS1-RandomForest Validation threshold | RandomForest | Hybrid   | 1.0         |                    2.47783 |         3.14823 |         0.206255 | Kandidat                  |
|          12 | Hybrid FTS3-CatBoost Validation threshold     | CatBoost     | Hybrid   | 3.0         |                    2.63881 |         3.14926 |         0.211888 | Kandidat                  |
|          13 | CatBoost Direct                               | CatBoost     | Direct   | -           |                    2.63881 |         3.14926 |         0.211888 | Kandidat                  |
|          14 | Hybrid FTS3-RandomForest Fixed 0.31           | RandomForest | Hybrid   | 3.0         |                    2.96643 |         3.15908 |         0.209684 | Kandidat                  |
|          15 | Hybrid FTS1-RandomForest Fixed 0.31           | RandomForest | Hybrid   | 1.0         |                    2.52621 |         3.17081 |         0.206717 | Kandidat                  |
|          16 | Hybrid FTS2-CatBoost Validation threshold     | CatBoost     | Hybrid   | 2.0         |                    2.29391 |         3.17381 |         0.271915 | Kandidat                  |
|          17 | Hybrid FTS1-CatBoost Validation threshold     | CatBoost     | Hybrid   | 1.0         |                    2.50549 |         3.18077 |         0.213063 | Kandidat                  |
|          18 | Hybrid FTS3-CatBoost Fixed 0.31               | CatBoost     | Hybrid   | 3.0         |                    2.83406 |         3.18258 |         0.218577 | Kandidat                  |
|          19 | SARIMA (1, 0, 1) (1, 0, 1, 24)                | SARIMA       | SARIMA   | -           |                    2.43181 |         3.18405 |         0.20529  | Terpilih validation       |
|          20 | Hybrid FTS1-CatBoost Fixed 0.31               | CatBoost     | Hybrid   | 1.0         |                    2.6428  |         3.18549 |         0.212021 | Kandidat                  |
|          21 | Hybrid FTS2-CatBoost Fixed 0.31               | CatBoost     | Hybrid   | 2.0         |                    2.58735 |         3.20144 |         0.226702 | Kandidat                  |
|          22 | Residual FTS1-LightGBM                        | LightGBM     | Residual | 1.0         |                    3.69457 |         3.23116 |         0.215199 | Kandidat                  |
|          23 | Hybrid FTS2-LightGBM Fixed 0.31               | LightGBM     | Hybrid   | 2.0         |                    2.58568 |         3.24015 |         0.233722 | Kandidat                  |
|          24 | ARIMA (2, 1, 1) None                          | ARIMA        | ARIMA    | -           |                    2.76866 |         3.27811 |         0.269554 | Kandidat                  |
|          25 | Hybrid FTS2-RandomForest Validation threshold | RandomForest | Hybrid   | 2.0         |                    2.37407 |         3.27869 |         0.265262 | Kandidat                  |
|          26 | Residual FTS1-RandomForest                    | RandomForest | Residual | 1.0         |                    3.32534 |         3.29515 |         0.219807 | Kandidat                  |
|          27 | ARIMA (1, 1, 1) None                          | ARIMA        | ARIMA    | -           |                    2.94483 |         3.31112 |         0.27539  | Kandidat                  |
|          28 | Hybrid FTS3-SVR Validation threshold          | SVR          | Hybrid   | 3.0         |                    2.24387 |         3.35138 |         0.21234  | Kandidat                  |
|          29 | Residual FTS1-CatBoost                        | CatBoost     | Residual | 1.0         |                    3.68552 |         3.35636 |         0.228815 | Kandidat                  |
|          30 | Hybrid FTS2-RandomForest Fixed 0.31           | RandomForest | Hybrid   | 2.0         |                    2.43193 |         3.36241 |         0.221775 | Kandidat                  |
|          31 | Hybrid FTS3-XGBoost Fixed 0.31                | XGBoost      | Hybrid   | 3.0         |                    3.17209 |         3.37078 |         0.224858 | Kandidat                  |
|          32 | SVR Direct                                    | SVR          | Direct   | -           |                    2.53269 |         3.38207 |         0.208808 | Kandidat                  |
|          33 | Hybrid FTS3-XGBoost Validation threshold      | XGBoost      | Hybrid   | 3.0         |                    2.92173 |         3.38851 |         0.219302 | Kandidat                  |
|          34 | XGBoost Direct                                | XGBoost      | Direct   | -           |                    2.92173 |         3.38851 |         0.219302 | Kandidat                  |
|          35 | Residual FTS3-CatBoost                        | CatBoost     | Residual | 3.0         |                    5.90755 |         3.39123 |         0.264558 | Kandidat                  |
|          36 | Hybrid FTS1-XGBoost Fixed 0.31                | XGBoost      | Hybrid   | 1.0         |                    2.96838 |         3.40138 |         0.218778 | Kandidat                  |
|          37 | Hybrid FTS1-XGBoost Validation threshold      | XGBoost      | Hybrid   | 1.0         |                    2.86887 |         3.4052  |         0.218827 | Kandidat                  |
|          38 | Residual FTS1-XGBoost                         | XGBoost      | Residual | 1.0         |                    3.85423 |         3.40816 |         0.229434 | Kandidat                  |
|          39 | Hybrid FTS2-XGBoost Validation threshold      | XGBoost      | Hybrid   | 2.0         |                    2.62133 |         3.42939 |         0.254825 | Kandidat                  |
|          40 | Residual FTS3-XGBoost                         | XGBoost      | Residual | 3.0         |                    6.29462 |         3.43009 |         0.272081 | Kandidat                  |
|          41 | Hybrid FTS3-SVR Fixed 0.31                    | SVR          | Hybrid   | 3.0         |                    2.58132 |         3.43304 |         0.225124 | Kandidat                  |
|          42 | Hybrid FTS2-XGBoost Fixed 0.31                | XGBoost      | Hybrid   | 2.0         |                    2.79879 |         3.45722 |         0.231731 | Kandidat                  |
|          43 | Weighted FTS order 3                          | nan          | nan      | -           |                    6.07635 |         3.60969 |         0.293295 | Kandidat                  |
|          44 | Hybrid FTS1-SVR Validation threshold          | SVR          | Hybrid   | 1.0         |                    2.06195 |         3.66873 |         0.222503 | Kandidat                  |
|          45 | Hybrid FTS1-SVR Fixed 0.31                    | SVR          | Hybrid   | 1.0         |                    2.06195 |         3.66873 |         0.222503 | Kandidat                  |
|          46 | Residual FTS3-SVR                             | SVR          | Residual | 3.0         |                    5.01766 |         3.86512 |         0.273865 | Kandidat                  |
|          47 | Hybrid FTS2-SVR Fixed 0.31                    | SVR          | Hybrid   | 2.0         |                    2.02633 |         3.89794 |         0.232508 | Kandidat                  |
|          48 | Weighted FTS order 2                          | nan          | nan      | -           |                    5.8637  |         4.36677 |         0.317013 | Kandidat                  |
|          49 | Hybrid FTS2-SVR Validation threshold          | SVR          | Hybrid   | 2.0         |                    1.96035 |         4.46885 |         0.25896  | Kandidat                  |
|          50 | Residual FTS1-SVR                             | SVR          | Residual | 1.0         |                    3.06898 |         4.50655 |         0.248768 | Kandidat                  |
|          51 | Residual FTS2-CatBoost                        | CatBoost     | Residual | 2.0         |                    2.83847 |         5.26404 |         0.297102 | Kandidat                  |
|          52 | Residual FTS2-XGBoost                         | XGBoost      | Residual | 2.0         |                    3.07515 |         5.57911 |         0.307472 | Kandidat                  |
|          53 | Residual FTS2-RandomForest                    | RandomForest | Residual | 2.0         |                    2.7875  |         5.79433 |         0.307802 | Kandidat                  |
|          54 | Weighted FTS order 1                          | nan          | nan      | -           |                    8.48301 |         5.83027 |         0.369668 | Kandidat                  |
|          55 | Residual FTS2-LightGBM                        | LightGBM     | Residual | 2.0         |                    3.08194 |         5.90176 |         0.320494 | Kandidat                  |
|          56 | Residual FTS2-SVR                             | SVR          | Residual | 2.0         |                    2.65111 |         6.50842 |         0.334663 | Kandidat                  |
|          57 | SARIMA (1, 1, 1) (1, 0, 1, 24)                | SARIMA       | SARIMA   | -           |                   12.5103  |        12.855   |         0.535627 | Kandidat                  |
