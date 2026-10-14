# STAGE 6 vs STAGE 7 RECONCILIATION — Phase C

## Angka yang diverifikasi

- Stage 6 XGBoost Direct Daily MAE dari `results/stage6/reproduction_check.csv`: **3.289850 kWh**
- Stage 7 XGBoost Direct Daily MAE dari `results/stage7/stage7_direct_comparison.csv`: **3.388513 kWh**
- Selisih Stage 7 - Stage 6: **0.098663 kWh**

## Eksperimen kontrol

Saya menjalankan kontrol XGBoost memakai hyperparameter terbaik historis Stage 6/Stage 2 dalam protokol Stage 7, tanpa tuning baru.

- Control Daily MAE: **3.289850 kWh**
- Control params: `{"colsample_bytree": 0.8, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 5, "n_estimators": 250, "subsample": 0.8}`

Hasil kontrol mendekati angka Stage 6 dan menunjukkan penyebab utama selisih adalah **perbedaan hyperparameter/model-selection protocol**, bukan dataset, target, origin, atau definisi Daily MAE.

## Rekonsiliasi faktor

| factor                          | stage6                                                                                                                               | stage7                                                                                                                         | status              | evidence                                                     |
|:--------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------|:--------------------|:-------------------------------------------------------------|
| dataset_target                  | data/hourly_energy_clean.csv Appliances_kWh                                                                                          | same                                                                                                                           | Confirmed same      | target alignment max diff=0.0                                |
| forecast_origins                | 20 origins                                                                                                                           | 20 origins                                                                                                                     | Confirmed same      | key mismatches=0                                             |
| feature_columns                 | HIST_COLS from Stage 3                                                                                                               | BASE_FEATURES + ROLLING_FEATURES                                                                                               | Confirmed same      | 22 feature columns                                           |
| hyperparameter_selection_metric | best xgboost_tuning_historical by one-step validation MAE                                                                            | Stage7 direct tuning by validation Daily MAE H1-H24                                                                            | Confirmed different | Stage6 best candidate differs from Stage7 selected candidate |
| xgboost_params                  | {"colsample_bytree": 0.8, "learning_rate": 0.03, "max_depth": 4.0, "min_child_weight": 5.0, "n_estimators": 250.0, "subsample": 0.8} | {"colsample_bytree": 0.9, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 3, "n_estimators": 350, "subsample": 0.9} | Confirmed different | control Daily MAE=3.289850                                   |
| daily_mae_definition            | sum 24 hourly predictions per origin, mean absolute daily error                                                                      | same                                                                                                                           | Confirmed same      | Stage7 metric recalc max direct diff=4.441e-16               |

## Kesimpulan Phase C

Penyebab perbedaan XGBoost Stage 6 vs Stage 7 diklasifikasikan **Confirmed**: Stage 6 memakai best candidate dari `xgboost_tuning_historical.csv` berbasis MAE validasi historis satu-langkah, sedangkan Stage 7 memilih XGBoost berdasarkan validation Daily MAE H1-H24 dari dua kandidat. Dua hasil tidak boleh disebut setara karena protokol pemilihan model berbeda.
