# REPRODUCIBILITY AUDIT

## Checks

{
  "py_compile_returncode": 0,
  "py_compile_stderr": "",
  "required_artifacts_present": true,
  "python": "3.10.0",
  "pandas_version": "2.2.3",
  "numpy_version": "2.2.6",
  "sklearn_version": "1.7.2",
  "xgboost_version": "3.2.0",
  "lightgbm_version": "4.6.0",
  "catboost_version": "1.2.10",
  "scipy_version": "1.15.3"
}

Clean-run penuh Stage 7 tidak diulang karena runtime panjang dan Stage 7 sudah tersedia. Verifikasi yang dijalankan pada Stage 7.1:

- compile source Stage 7 dan audit scripts,
- audit keberadaan artefak wajib,
- hitung ulang metrik dari prediksi individual,
- rekonstruksi fitur lag/rolling secara numerik,
- kontrol XGBoost Stage 6 params pada Phase A-C.

Kesimpulan: artefak cukup untuk reproduksi audit dan evaluasi ulang metrik. Reproduksi penuh model training tetap mahal tetapi dapat dijalankan melalui `python run_stage7.py`.
