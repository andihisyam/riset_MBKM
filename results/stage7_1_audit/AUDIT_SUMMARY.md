# AUDIT SUMMARY — Stage 7.1 Final

## A. Executive Summary

Stage 7 dapat diaudit dari artefak prediksi individual, konfigurasi, dan source code. Integritas data, origin, horizon, dan target aktual konsisten. Pengujian numerik fitur tidak menemukan penggunaan nilai aktual masa depan pada fitur lag/rolling di origin evaluasi.

Namun, hasil Stage 7 harus dibaca sebagai **benchmark eksploratif dengan keterbatasan**. Model terbaik pada test bukan model yang layak dipilih otomatis karena beberapa hasil test-best buruk pada validation.

## B. Audit Verdict

**PASS WITH LIMITATIONS**

Tidak ada bug atau leakage terkonfirmasi yang memerlukan koreksi hasil Stage 7, tetapi ada keterbatasan metodologis yang memengaruhi cara klaim harus ditulis.

## C. Critical/High Findings

| id       | phase   | severity   | status               | title                                                       | evidence                                              | impact                                                            | recommendation                                                                        |
|:---------|:--------|:-----------|:---------------------|:------------------------------------------------------------|:------------------------------------------------------|:------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
| S7_1_004 | F       | High       | Confirmed limitation | Residual FTS3-RandomForest is test-best but validation-poor | Validation Daily MAE 5.7880 vs test Daily MAE 3.0977. | Do not select this model as primary method despite best test MAE. | Use validation-selected model for recommendation and report test-best as exploratory. |

## D. Verified Results

- Fitur lag/rolling direkonstruksi numerik dari histori origin; max diff: 9.416e-14.
- Metric CSV konsisten dengan prediksi individual; max Daily MAE diff: 1.332e-15.
- Hybrid FTS1-LightGBM Fixed 0.31 memberi perbaikan sangat kecil dan tidak signifikan secara statistik.

## E. Stage 6 vs Stage 7

Rekonsiliasi Phase A-C tetap berlaku: XGBoost Stage 6 vs Stage 7 berbeda karena protocol pemilihan hyperparameter, bukan dataset/target/origin.

## F. Model Selection

Model dipilih berdasarkan validation:

| class    | model                                |   validation_daily_mae |   test_daily_mae |   hourly_mae_test |
|:---------|:-------------------------------------|-----------------------:|-----------------:|------------------:|
| Hybrid   | Hybrid FTS2-SVR Validation threshold |                1.96035 |          4.46885 |          0.25896  |
| Direct   | RandomForest Direct                  |                2.48264 |          3.1435  |          0.206358 |
| Residual | Residual FTS2-SVR                    |                2.65111 |          6.50842 |          0.334663 |

Model validation-selected terbaik adalah **Hybrid FTS2-SVR Validation threshold**. Hasil test model tersebut adalah Daily MAE **4.4689 kWh**.

Model test-best eksploratif adalah **Residual FTS3-RandomForest** dengan Daily MAE **3.0977 kWh**, tetapi tidak direkomendasikan sebagai model utama karena validasi buruk.

## G. Research Implications

- XGBoost masih layak sebagai baseline historis, tetapi tidak lagi satu-satunya kandidat kuat.
- LightGBM/RandomForest Direct lebih defensible sebagai benchmark Direct ML pada Stage 7.
- Pendekatan hybrid belum terbukti membantu secara stabil.
- Orde FTS memengaruhi sparsity pola; order lebih tinggi dapat membaik pada test tetapi berisiko fallback/sparsity.
- Penelitian memerlukan eksperimen tambahan jika ingin klaim hybrid sebagai metode utama.

## H. Recommended Next Action

Untuk laporan ke dosen, gunakan narasi: **Direct ML adalah benchmark utama; hybrid FTS-ML adalah eksperimen tambahan yang belum menunjukkan peningkatan stabil/signifikan.** Jangan memilih model hanya karena test Daily MAE terendah.
