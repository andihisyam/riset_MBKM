# Stage 7 Preflight Audit

## Artefak yang digunakan

- Dataset hourly: `data/hourly_energy_clean.csv`
- Fitur: `data/hourly_features.csv`
- Target: `Appliances_kWh` dalam kWh
- Jumlah observasi hourly lengkap: 3,289
- Feature columns: lag_1, lag_2, lag_3, lag_6, lag_12, lag_24, lag_48, lag_168, hour, day_of_week, is_weekend, sin_hour, cos_hour, sin_dow, cos_dow, rolling_mean_3, rolling_mean_6, rolling_mean_12, rolling_mean_24, rolling_std_3, rolling_std_6, rolling_std_24

## Split kronologis

- Train: index 0 sampai 2301; tanggal 2016-01-11 17:00:00 sampai 2016-04-16 14:00:00
- Validation: index 2302 sampai 2794; tanggal 2016-04-16 15:00:00 sampai 2016-05-07 03:00:00
- Test: index 2795 sampai 3288; tanggal 2016-05-07 04:00:00 sampai 2016-05-27 17:00:00

## Audit horizon dan fitur

Seluruh model Stage 7 memakai origin yang sama untuk H1-H24. Fitur lag dan rolling sudah tersedia pada forecast origin karena dihitung dari nilai historis yang digeser satu jam. Tidak ada temperatur/kelembapan masa depan yang dimasukkan ke Direct ML Stage 7.

## FTS dan threshold historis

Stage 2 memilih quantile 15 interval dan Weighted FTS. Stage 7 membandingkan order 1, 2, dan 3 dengan jumlah interval yang sama. Threshold tetap 0.31 kWh dipakai sebagai pembanding Gate B historis; threshold validasi dipilih hanya dari validation Daily MAE.

## Catatan validitas

Periode test Stage 7 sudah pernah dipakai dalam Stage 1-6, sehingga Stage 7 adalah evaluasi lanjutan eksploratif, bukan holdout independen sepenuhnya. Residual FTS-ML memakai prediksi FTS rolling untuk membentuk residual training agar tidak memakai prediksi FTS in-sample yang terlalu optimistis.

## Reproduksi order FTS validasi

|   order |   validation_daily_mae |   Daily MAE |   n_patterns |   rare_patterns |   fallback_steps_validation |   fallback_steps_test |
|--------:|-----------------------:|------------:|-------------:|----------------:|----------------------------:|----------------------:|
|       1 |                8.48301 |     5.83027 |           15 |               0 |                           0 |                     0 |
|       2 |                5.8637  |     4.36677 |          204 |              51 |                           1 |                     0 |
|       3 |                6.07635 |     3.60969 |         1038 |             754 |                          34 |                   116 |
