# Presentation Redesign Audit

## 1. Permasalahan desain versi lama

Versi lama lebih mirip dashboard teknis: banyak tabel dan grafik tampil sekaligus, sehingga alur penjelasan untuk dosen kurang bertahap.

## 2. Perubahan layout

Dashboard diubah menjadi presentasi storytelling 9 halaman dengan pola judul, pengantar, satu visual utama, penjelasan, pesan utama, dan transisi. Sidebar memakai nomor 01-09, tombol sebelumnya/berikutnya, progress indicator, keyboard navigation, fullscreen, dan toggle sidebar.

## 3. Perubahan storytelling

Urutan baru: masalah prediksi, dataset, FTS, XGBoost, hybrid, hasil, SHAP global, SHAP local, kesimpulan. Detail teknis dipindahkan ke Lampiran Teknis.

## 4. Sumber data per halaman

- Halaman 1: `results/stage6/gate_decision_trace.csv`, `data/hourly_energy_clean.csv`.
- Halaman 2: `energydata_complete.csv`, `data/hourly_energy_clean.csv`.
- Halaman 3: `results/fuzzy_partition_results.csv`, `data/hourly_energy_clean.csv`.
- Halaman 4: `data/hourly_features.csv`, `results/stage6/gate_decision_trace.csv`.
- Halaman 5: `results/stage6/gate_decision_trace.csv`, `results/stage6/stage6_complete_audit_summary.csv`.
- Halaman 6: `results/baseline_24h_fixed_origin.csv`, `results/stage5_final_model_comparison.csv`, `results/stage5_complexity_vs_benefit.csv`, `results/stage5_pairwise.csv`.
- Halaman 7: `results/stage6/shap_global_importance.csv`, `figures/stage6/shap_importance.png`.
- Halaman 8: `results/stage6/shap_local_cases.csv`, `figures/stage6/local_cases/low_error_waterfall.png`.
- Halaman 9: ringkasan dari Stage 5 dan Stage 6 audit.

## 5. Grafik yang digunakan

Grafik asli: SHAP importance, SHAP heatmap, daily SHAP waterfall, local SHAP waterfall, Stage 4/5 error boxplot. Grafik tambahan dibuat dari data asli: contoh forecast origin, distribusi hourly, distribusi quantile FTS, dan bar chart Daily MAE.

## 6. Pengecekan responsivitas

CSS memakai layout responsif, tabel scroll horizontal, sidebar scroll, aturan khusus untuk tinggi layar laptop sekitar 1366x768, dan media query mobile. Navigasi keyboard dan tombol fullscreen tersedia.

## 7. Masalah yang belum terselesaikan

Browser console dan screenshot lintas resolusi belum diautomasi di environment ini. Validasi yang dilakukan adalah build statis, keberadaan asset, dan konsistensi angka terhadap CSV sumber.
