# STATISTICAL VALIDATION

| comparison                                            |   n_origins |   mean_error_diff |   median_error_diff |   wilcoxon_p |   bootstrap_ci_low |   bootstrap_ci_high |   wins_for_second |   wins_for_first |   ties | note                                                            |
|:------------------------------------------------------|------------:|------------------:|--------------------:|-------------:|-------------------:|--------------------:|------------------:|-----------------:|-------:|:----------------------------------------------------------------|
| LightGBM Direct minus Hybrid FTS1-LightGBM Fixed 0.31 |          20 |        0.00428294 |          -0.0056995 |     0.600871 |         -0.0428325 |           0.0639079 |                 8 |               11 |      1 | Only 20 non-overlap origins; multiple comparisons not adjusted. |
| RandomForest Direct minus Residual FTS3-RandomForest  |          20 |        0.0458026  |           0.443728  |     0.812355 |         -0.918417  |           0.964757  |                12 |                8 |      0 | Only 20 non-overlap origins; multiple comparisons not adjusted. |
| LightGBM Direct minus RandomForest Direct             |          20 |       -0.00377158 |           0.0378206 |     0.840822 |         -0.408574  |           0.377619  |                11 |                9 |      0 | Only 20 non-overlap origins; multiple comparisons not adjusted. |

Hanya ada 20 forecast origin non-overlap. Wilcoxon signed-rank dipakai pada selisih error harian berpasangan. Confidence interval bootstrap origin-level dilaporkan sebagai indikasi sensitivitas, tetapi power statistik rendah dan ada risiko multiple comparisons.

Tidak ada dasar untuk menyatakan perbedaan hybrid terbaik signifikan pada taraf 5%.
