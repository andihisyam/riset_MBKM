# GATING AUDIT

## Fokus Hybrid FTS1-LightGBM threshold 0.31

| model                           | split      | algorithm   |   fts_order |   threshold |   gate_rule_mismatch |   active_count |   active_share |   improved_hourly_count |   worsened_hourly_count |   unchanged_hourly_count |   daily_mae_hybrid |   daily_mae_direct |   daily_mae_change |   hourly_mae_hybrid |   hourly_mae_direct |   hourly_mae_change |   mean_error_cancellation_hybrid |   mean_error_cancellation_direct |
|:--------------------------------|:-----------|:------------|------------:|------------:|---------------------:|---------------:|---------------:|------------------------:|------------------------:|-------------------------:|-------------------:|-------------------:|-------------------:|--------------------:|--------------------:|--------------------:|---------------------------------:|---------------------------------:|
| Hybrid FTS1-LightGBM Fixed 0.31 | test       | LightGBM    |           1 |        0.31 |                    0 |             52 |       0.108333 |                      31 |                      21 |                      428 |            3.13545 |            3.13973 |        -0.00428294 |            0.211324 |            0.212029 |         -0.00070592 |                          1.93632 |                          1.94898 |
| Hybrid FTS1-LightGBM Fixed 0.31 | validation | LightGBM    |           1 |        0.31 |                    0 |             88 |       0.183333 |                      25 |                      63 |                      392 |            2.74291 |            2.54532 |         0.197592   |            0.252227 |            0.241024 |          0.0112032  |                          3.31053 |                          3.23924 |

Gate menggunakan prediksi Direct sebagai input keputusan (`direct_prediction <= threshold`), bukan nilai aktual. Threshold 0.31 diperlakukan sebagai fixed baseline. Threshold validasi pada kombinasi lain dipilih dari validation Daily MAE. Banyak gate validasi memilih threshold rendah sehingga residual tidak aktif; model tersebut secara praktis kembali menjadi Direct.

Kesimpulan: hybrid terbaik hanya memberi perbaikan kecil dan tidak cukup kuat untuk klaim metode utama.
