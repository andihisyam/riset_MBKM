# LEAKAGE AUDIT — Stage 7.1 Phase B

## Ringkasan awal

Audit Phase B belum menemukan bukti langsung bahwa Stage 7 memakai target test untuk training, threshold selection, atau scaling. Namun statusnya **belum boleh disebut leakage-free** sampai Phase D-H selesai.

## Lag dan rolling

- File: `src/stage2_pipeline.py`
- Bukti: fitur lag memakai shift historis; rolling mean/std dibuat dari `TARGET.shift(1)`.
- Dampak: fitur lag dan rolling tampak kausal karena tidak mengambil konsumsi aktual H1-H24.
- Status: Low risk, perlu verifikasi kolom lengkap pada audit lanjutan.

## FTS

- File: `src/stage7_pipeline.py`
- Quantile bounds utama dibuat dari training: `hourly.loc[: bounds.train_end - 1, TARGET]`.
- Test FTS matrix memakai rule_until `bounds.val_end`, sehingga train+validation boleh dipakai sebelum test.
- Forecast H1-H24 recursive menambahkan prediksi FTS ke history, bukan aktual future.
- Status: no direct leakage evidence in Phase B.

## Residual learner

- Residual training memakai rolling FTS per origin: `build_rules(values[: o + 1])`.
- Residual target tetap aktual masa depan untuk supervised training, tetapi target_index dibatasi `< fit_until`.
- Untuk final test, residual learner fit sampai validation (`bounds.val_end`), bukan test.
- Status: no direct test leakage evidence, tetapi perlu audit distribusi besar validation-test pada Phase F.

## Scaling dan tuning

- SVR memakai `make_pipeline(StandardScaler(), SVR(...))`; scaler fit terjadi di `.fit()` pada frame training masing-masing horizon.
- Tuning Direct dan Residual memakai validation Daily MAE.
- Threshold validasi dipilih dari `d_val` dan `r_val`; test hanya dievaluasi setelah threshold dikunci.
- Status: no direct test-selection leakage in code path audited.

## Temuan

Lihat `ISSUE_REGISTER.csv`. Temuan terbesar sampai Phase B bukan leakage langsung, melainkan keterbatasan bahwa test Stage 7 sudah pernah dianalisis pada Stage 1-6.
