# Stage 8 Final Report

## Verdict

Stage 8 berhasil dijalankan pada data 10-menit asli. Track A merupakan approximate reproduction karena split random CARET dan seed paper tidak tersedia. Track B menjadi evaluasi utama karena menjaga urutan waktu dan memisahkan train, validation, serta test.

## Konfigurasi terpilih Track B

- Variant: `B1_history_time`
- FTS order: `2`
- Seleksi: validation MAE hybrid; test tidak dipakai memilih konfigurasi.
- Test Hybrid MAE: **27.6817 Wh**
- Test FTS-only MAE: **39.1872 Wh**
- Test RF direct MAE: **27.4393 Wh**
- Perbaikan residual terhadap FTS: **29.36%**

## Interpretasi

Hasil Stage 8 tidak boleh dipakai untuk menyatakan hybrid mengungguli paper secara mutlak karena unit, split, fitur, dan protokol paper tidak identik. Fitur lingkungan pada Track B hanya memakai nilai forecast origin; nilai lingkungan target tidak digunakan sebagai forecasting utama. Fallback FTS dilaporkan pada CSV prediksi dan ringkasan model.
