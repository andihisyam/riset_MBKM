# Stage 6 Reproducibility Audit

Model dijelaskan dengan rekonstruksi prosedur Stage 4-5, karena artifact model ter-serialize tidak tersedia. Script melatih ulang model dengan konfigurasi tersimpan (`xgboost_tuning_historical.csv`), split kronologis 70/15/15, fitur hourly Stage 2, dan non-overlap origins Stage 4-5.

Threshold Gate B memakai threshold training-only `q33 = 0.310000` dan aturan deterministik: jika prediksi XGBoost <= q33 maka pilih Residual FTS-XGBoost, selain itu pilih XGBoost Direct.

Risiko leakage: tidak ditemukan pada fitur SHAP karena input inference hanya memakai fitur di origin t. Model direkonstruksi, bukan di-load dari artifact binary.
