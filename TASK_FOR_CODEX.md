# TASK UNTUK CODEX — Sinkronisasi dan Menjalankan Notebook Riset MBKM

Saya sudah pernah clone repository ini sebelumnya di laptop ini.

Repository:

```text
https://github.com/andihisyam/riset_MBKM
```

Branch utama:

```text
main
```

Commit terbaru yang diharapkan dari GitHub:

```text
6557d38 Add Stage 8 and Stage 9 benchmark artifacts
```

## Tujuan

Bersihkan repository lokal agar sama dengan versi GitHub terbaru, lalu siapkan environment Python dan jalankan notebook penelitian.

Tidak masalah menghapus file lokal yang tidak perlu seperti cache, ZIP lama, hasil eksperimen lokal lama, virtual environment lama, dan file generated yang tidak tracked Git.

## Aturan penting

1. Boleh menghapus file lokal yang tidak tracked Git.
2. Boleh reset repository lokal ke `origin/main`.
3. Jangan push perubahan ke GitHub.
4. Jangan menjalankan ulang eksperimen besar Stage 1–7 kecuali saya minta.
5. Jangan mengubah hasil CSV, grafik, model, threshold, atau konfigurasi eksperimen.
6. Jika artifact Stage 6/7 tidak tersedia di repo setelah pull, laporkan sebagai file hilang. Jangan mengarang data.
7. Fokus menjalankan notebook dari artifact yang tersedia.
8. Jika menemukan konflik Git atau error environment, laporkan dulu sebelum memperbaiki secara destruktif.

## Langkah 1 — Cek repository

Masuk ke folder repo lokal, lalu jalankan:

```bash
git remote -v
git branch
git status
git log --oneline --decorate --max-count=5
```

Pastikan remote mengarah ke:

```text
https://github.com/andihisyam/riset_MBKM.git
```

Jika remote berbeda, laporkan dulu.

## Langkah 2 — Sinkronkan ke GitHub terbaru

Karena file lokal boleh dibersihkan, lakukan:

```bash
git fetch origin
git checkout main
git reset --hard origin/main
git clean -fdx
```

Catatan: `git clean -fdx` akan menghapus semua file lokal yang tidak tracked Git, termasuk `.venv`, cache, ZIP, dan artifact lokal lama. Ini memang diizinkan untuk task ini.

Setelah itu cek:

```bash
git status
git log --oneline --decorate --max-count=5
```

Pastikan commit terbaru adalah:

```text
6557d38 Add Stage 8 and Stage 9 benchmark artifacts
```

Jika commit berbeda, laporkan commit aktualnya.

## Langkah 3 — Cek struktur file penting

Pastikan file inti berikut ada:

```text
README.md
requirements.txt
docs/methodology.md
docs/repository_audit.md
docs/github_preparation_report.md
notebooks/00_complete_research_walkthrough.ipynb
notebooks/01_data_exploration.ipynb
notebooks/02_data_preprocessing.ipynb
notebooks/03_weighted_fuzzy_time_series.ipynb
notebooks/04_random_forest_residual.ipynb
notebooks/05_hybrid_forecasting.ipynb
notebooks/06_model_evaluation.ipynb
notebooks/07_shap_explainability.ipynb
tests/test_readme_links.py
tests/test_repository_artifacts.py
```

Cek juga folder dasar:

```text
data/
figures/
results/
src/
tests/
notebooks/
docs/
```

Jika file atau folder di atas hilang, buat daftar file hilang.

## Langkah 4 — Cek artifact Stage 6 dan Stage 7

Cek apakah artifact final ini tersedia:

```text
results/stage6/
results/stage7/
results/stage7_1_audit/
figures/stage6/
figures/stage7/
reports/stage7/
```

Jika folder tersebut tidak ada setelah pull/reset, laporkan bahwa artifact final Stage 6/7 belum tersedia dari GitHub current tree.

Jangan membuat CSV, grafik, atau laporan palsu untuk menggantikan artifact yang hilang.

## Langkah 4B — Cek artifact Stage 8 dan Stage 9

Pastikan file berikut tersedia setelah sinkronisasi:

```text
run_stage8.py
src/stage8_paper_benchmark.py
results/stage8_paper_benchmark/
run_stage9.py
src/stage9_arima_sarima_benchmark.py
results/stage9_arima_sarima/
```

Stage 8 memakai data 10 menit dan satuan Wh, sehingga tidak digabungkan ke ranking hourly Stage 1–7. Stage 9 menambahkan ARIMA dan SARIMA pada evaluasi hourly H1–H24. Jangan menjalankan ulang Stage 8 atau Stage 9 jika artifact hasil sudah tersedia.

## Langkah 5 — Buat environment Python baru

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jika install dependency gagal, catat error lengkap dan lanjutkan audit file yang bisa dilakukan.

## Langkah 6 — Jalankan test ringan

Jalankan:

```bash
pytest -q tests
```

Jika test gagal, baca penyebabnya.

Jika test gagal karena artifact Stage 6/7 tidak ada, laporkan file yang hilang. Jangan langsung menjalankan training ulang.

## Langkah 7 — Jalankan notebook utama

Prioritas notebook:

```text
notebooks/00_complete_research_walkthrough.ipynb
```

Jalankan lewat Jupyter UI:

```bash
jupyter notebook
```

Buka dan run notebook:

```text
notebooks/00_complete_research_walkthrough.ipynb
```

Jika ingin menjalankan dari command line:

```bash
jupyter nbconvert --to notebook --execute notebooks/00_complete_research_walkthrough.ipynb --output 00_complete_research_walkthrough.executed.ipynb --output-dir notebooks
```

Jika gagal karena kernel, permission Jupyter, atau environment Windows, coba jalankan manual lewat VS Code atau Jupyter Notebook UI. Laporkan error lengkap.

## Langkah 8 — Notebook pendukung

Jika notebook utama berhasil, cek notebook pendukung sesuai kebutuhan:

```text
notebooks/01_data_exploration.ipynb
notebooks/02_data_preprocessing.ipynb
notebooks/03_weighted_fuzzy_time_series.ipynb
notebooks/04_random_forest_residual.ipynb
notebooks/05_hybrid_forecasting.ipynb
notebooks/06_model_evaluation.ipynb
notebooks/07_shap_explainability.ipynb
```

Jangan menjalankan ulang eksperimen berat dari pipeline Python kecuali diminta.

## Langkah 9 — Interpretasi hasil yang harus dijaga

Saat menjelaskan penelitian, gunakan batasan berikut:

1. Penelitian membahas prediksi konsumsi energi H1–H24.
2. Metode yang dianalisis mencakup Direct ML, Weighted FTS, Residual FTS–ML, dan Hybrid berbasis threshold.
3. Residual FTS3–RandomForest pernah menjadi hasil test terbaik pada Stage 7, tetapi hasil validation-nya buruk.
4. Karena itu, hasil tersebut harus disebut eksploratif, bukan klaim final mutlak.
5. Verdict audit Stage 7.1 adalah `PASS WITH LIMITATIONS`.
6. SHAP Stage 6 menjelaskan XGBoost Direct dan residual corrector, bukan otomatis menjelaskan Random Forest final.
7. Jangan menyatakan hybrid selalu lebih baik jika validasi dan statistik tidak mendukung.
8. Jangan menyembunyikan keterbatasan bahwa sebagian evaluasi menggunakan periode yang sudah pernah dianalisis pada tahap sebelumnya.

## Output yang saya minta dari Codex

Setelah selesai, berikan laporan singkat dalam bahasa Indonesia berisi:

1. Commit Git terbaru setelah pull/reset.
2. Apakah pembersihan lokal berhasil.
3. Apakah environment berhasil dibuat.
4. Apakah dependency berhasil di-install.
5. Hasil `pytest -q tests`.
6. Apakah notebook utama berhasil dijalankan.
7. Daftar artifact/file yang hilang.
8. Error yang muncul, jika ada.
9. Rekomendasi langkah berikutnya.

Jangan push perubahan apa pun ke GitHub.

## Jika artifact Stage 6/7 tidak ada

Jika artifact final Stage 6/7 tidak ikut tersedia dari GitHub, berhenti pada audit file dan beri laporan. Jangan mengulang training besar atau membuat hasil baru.

Contoh laporan yang diharapkan:

```text
Repo berhasil sinkron ke origin/main commit <hash>.
Environment berhasil/gagal dibuat.
Dependency berhasil/gagal di-install.
pytest berhasil/gagal.
Notebook utama berhasil/gagal dijalankan.
Artifact yang hilang: ...
Error utama: ...
Rekomendasi: minta pemilik repo mengirim/push artifact Stage 6/7 atau ZIP pendukung.
```
