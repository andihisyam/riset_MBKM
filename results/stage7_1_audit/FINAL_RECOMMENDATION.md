# FINAL RECOMMENDATION

## Model berdasarkan validation

Model dengan validation Daily MAE terbaik di antara kelas Direct/Residual/Hybrid adalah:

**Hybrid FTS2-SVR Validation threshold**

- Validation Daily MAE: 1.9604 kWh
- Test Daily MAE: 4.4689 kWh
- Test Hourly MAE: 0.2590 kWh

## Model test-best eksploratif

Model dengan test Daily MAE terendah adalah **Residual FTS3-RandomForest** dengan Daily MAE 3.0977 kWh. Namun model ini tidak dipilih sebagai metode utama karena performa validation buruk dan berisiko overfit/distribution-specific.

## Rekomendasi metode penelitian

Metode paling dapat dipertanggungjawabkan untuk narasi penelitian adalah **Direct ML sebagai baseline utama, dengan LightGBM/RandomForest sebagai pembanding kuat, dan Hybrid FTS-ML sebagai analisis eksploratif**. Jika penelitian tetap ingin mempertahankan judul hybrid, klaim harus dibatasi: hybrid berbasis threshold belum terbukti memberi peningkatan yang stabil dan signifikan.
