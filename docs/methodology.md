# Methodology

Dokumen ini menjelaskan metodologi aktual yang digunakan pada repository.

## 1. Dataset

Penelitian memakai UCI Appliances Energy Prediction Dataset. Target mentah adalah `Appliances` dalam Wh pada interval 10 menit. Data mentah berada di `energydata_complete.csv`.

## 2. Agregasi hourly

Untuk setiap jam, enam nilai 10 menit dijumlahkan:

```text
Appliances_kWh(t) = sum(Appliances_10min dalam 1 jam) / 1000
```

Jam yang tidak memiliki enam observasi lengkap dikeluarkan. Hasil akhir adalah 3.289 observasi hourly lengkap di `data/hourly_energy_clean.csv`.

## 3. Split kronologis

Data dibagi tanpa shuffle:

- Train 70%.
- Validation 15%.
- Test 15%.

Forecast H1-H24 dievaluasi pada origin non-overlap agar total harian tidak saling tumpang tindih.

## 4. Feature engineering

Fitur Direct ML terdiri dari:

- Lag konsumsi historis: `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`, `lag_24`, `lag_48`, `lag_168`.
- Rolling mean historis: 3, 6, 12, 24 jam.
- Rolling standard deviation historis: 3, 6, 24 jam.
- Kalender: hour, day of week, weekend, sin/cos hour, sin/cos day of week.

Rolling dihitung dari `Appliances_kWh.shift(1)`, sehingga hanya memakai nilai sebelum forecast origin.

## 5. Quantile 15 interval

Weighted FTS memakai 15 interval quantile yang dipelajari dari data training. Quantile dipakai karena konsumsi rendah jauh lebih padat daripada konsumsi tinggi, sehingga interval rendah menjadi lebih sempit.

## 6. Fuzzifikasi

Setiap nilai konsumsi hourly dipetakan ke fuzzy state berdasarkan interval quantile. State diberi label A1 sampai A15.

## 7. FLR dan FLRG

Fuzzy Logical Relationship (FLR) membentuk pola transisi state. Untuk order 3, pola memakai tiga state historis sebagai antecedent dan state berikutnya sebagai consequent.

## 8. Weighted FTS recursive H1-H24

Prediksi FTS dilakukan recursive. Setelah H1 diprediksi, nilai prediksi tersebut dimasukkan ke history untuk memprediksi H2, dan seterusnya sampai H24. Jika pola tidak ditemukan, fallback memakai rata-rata training yang diizinkan.

## 9. Residual learning dengan Random Forest

Residual didefinisikan sebagai:

```text
residual = aktual - prediksi_FTS
```

Random Forest Regressor dilatih untuk memprediksi residual menggunakan fitur historis pada forecast origin. Prediksi akhir:

```text
prediksi_final = prediksi_FTS + prediksi_residual_RF
```

Audit Stage 7.1 menemukan bahwa FTS3-RandomForest memiliki test Daily MAE paling kecil, tetapi validation Daily MAE buruk, sehingga tidak dipilih sebagai bukti stabilitas utama.

## 10. Pencegahan data leakage

- Split dilakukan secara kronologis.
- Quantile FTS dipelajari dari training.
- Fitur lag dan rolling hanya memakai histori.
- SVR menggunakan `StandardScaler` di dalam sklearn Pipeline, sehingga scaler fit hanya pada data training tiap model.
- Threshold hybrid dipilih dari validation, bukan test.
- Audit numerik Stage 7.1 merekonstruksi fitur lag/rolling dan tidak menemukan penggunaan aktual masa depan.

## 11. Evaluasi

Metrik utama adalah Daily MAE:

```text
Daily MAE = mean(|sum_24jam aktual - sum_24jam prediksi|)
```

Metrik pendukung: Hourly MAE, RMSE, MAPE, sMAPE, bias, horizon-wise MAE, dan statistik berpasangan antar-origin.

## 12. SHAP

SHAP tersedia pada Stage 6 untuk XGBoost Direct. SHAP Random Forest residual belum tersedia dan tidak diklaim dalam repository ini.
