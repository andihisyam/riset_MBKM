# Stage 7 Final Report

## Ringkasan utama

Stage 7 membandingkan lima algoritma ML, tiga orde Weighted FTS, residual FTS-ML, dan hybrid threshold tanpa mengubah hasil Stage 1-6. Evaluasi memakai forecast origin H1-H24 yang sama dan split kronologis 70/15/15.

## Jawaban pertanyaan penelitian

1. **Algoritma Direct terbaik pada test:** LightGBM Direct dengan Daily MAE 3.1397 kWh dan Hourly MAE 0.2120 kWh.
2. **Algoritma Direct terbaik pada validation:** RandomForest Direct dengan validation Daily MAE 2.4826 kWh.
3. **Orde FTS terbaik berdasarkan validation:** Weighted FTS order 2, validation Daily MAE 5.8637 kWh. Pada test, FTS standalone terbaik adalah order 3 dengan Daily MAE 3.6097 kWh, tetapi ini bukan dasar pemilihan konfigurasi.
4. **Residual FTS-ML terbaik pada test:** Residual FTS3-RandomForest dengan Daily MAE 3.0977 kWh, improvement 1.46% terhadap Direct algoritma terkait. Namun validation Daily MAE-nya 5.7880 kWh, sehingga hasil test ini berisiko tidak stabil.
5. **Residual FTS-ML terbaik berdasarkan validation:** Residual FTS2-SVR dengan validation Daily MAE 2.6511 kWh dan test Daily MAE 6.5084 kWh.
6. **Hybrid threshold terbaik pada test:** Hybrid FTS1-LightGBM Fixed 0.31 dengan Daily MAE 3.1354 kWh, improvement 0.14% terhadap Direct algoritma terkait, dan residual dipilih 10.83% dari prediksi hourly test.
7. **Hybrid threshold terbaik berdasarkan validation:** Hybrid FTS2-SVR Validation threshold dengan validation Daily MAE 1.9604 kWh dan test Daily MAE 4.4689 kWh.
8. **Perbandingan dengan XGBoost Direct Stage sebelumnya:** XGBoost Direct Stage 5/6 = 3.2899 kWh. Direct terbaik Stage 7 (LightGBM Direct) lebih rendah dari angka tersebut, tetapi protokol dan algoritma Stage 7 adalah evaluasi eksploratif lanjutan pada periode test yang sudah pernah dianalisis.

## Konsistensi hasil

Hybrid test terbaik menang terhadap Direct algoritma terkait pada 8 origin, kalah pada 11 origin, dan seri pada 1 origin dari 20 origin test. Karena jumlah origin kecil dan periode test sudah dipakai pada Stage 1-6, perbedaan MAE tidak boleh dibaca sebagai bukti signifikan tanpa uji berpasangan dan pembahasan keterbatasan.

## Biaya komputasi

Waktu training dan inference dicatat di `stage7_tuning_log.csv`. Random Forest dan CatBoost lebih mahal dibanding LightGBM/SVR pada beberapa bagian. Residual dan hybrid menambah biaya karena melatih model per horizon untuk setiap orde FTS.

## Kesimpulan hati-hati

Hasil Stage 7 menunjukkan bahwa menambahkan FTS tidak otomatis meningkatkan performa semua algoritma. Beberapa hybrid memilih residual sangat jarang atau tidak sama sekali ketika threshold validasi rendah, sehingga hasilnya sama dengan Direct. Klaim yang aman adalah: **LightGBM/Random Forest Direct menjadi pembanding kuat; hybrid terbaik hanya memberi peningkatan kecil dan perlu dibaca bersama validasi, stabilitas origin, serta biaya komputasi.**

## File pendukung

- `stage7_direct_comparison.csv`
- `stage7_fts_order_comparison.csv`
- `stage7_residual_comparison.csv`
- `stage7_hybrid_comparison.csv`
- `stage7_final_model_ranking.csv`
- `stage7_hourly_predictions.csv`
- `stage7_daily_predictions.csv`
- `stage7_threshold_analysis.csv`
- `stage7_tuning_log.csv`
