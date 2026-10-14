# FTS ORDER AUDIT

## Ringkasan

|   order |   n_patterns |   rare_patterns |   validation_daily_mae |   test_daily_mae |   fallback_origins_test |   fallback_steps_test |   test_hourly_mae_recalc |   test_daily_mae_recalc |
|--------:|-------------:|----------------:|-----------------------:|-----------------:|------------------------:|----------------------:|-------------------------:|------------------------:|
|       1 |           15 |               0 |                8.48301 |          5.83027 |                       0 |                     0 |                 0.369668 |                 5.83027 |
|       2 |          204 |              51 |                5.8637  |          4.36677 |                       0 |                     0 |                 0.317013 |                 4.36677 |
|       3 |         1038 |             754 |                6.07635 |          3.60969 |                      13 |                   116 |                 0.293295 |                 3.60969 |

Order dipilih berdasarkan validation, bukan test. Order 2 memiliki validation Daily MAE terendah. Order 3 memiliki test Daily MAE lebih baik, tetapi memakai banyak pola langka dan fallback pada test, sehingga tidak boleh dipilih hanya karena test.

## Catatan fallback

Fallback test terbesar terjadi pada order 3. Fallback memperlihatkan sparsity pola ketika order meningkat.
