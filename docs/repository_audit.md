# Repository Audit

## Git

- Branch aktif saat audit: `main`.
- Remote: `origin` mengarah ke repository GitHub existing.
- Repository baru tidak dibuat.

## Artefak penting

- Dataset mentah: `energydata_complete.csv`.
- Dataset hourly: `data/hourly_energy_clean.csv`.
- Fitur hourly: `data/hourly_features.csv`.
- Stage 7 results: `results/stage7/`.
- Stage 7.1 audit: `results/stage7_1_audit/`.

## File yang sebaiknya tidak dipush jika belum tracked

- ZIP paket sementara: `stage*_*.zip`, `interactive_report_stage6.zip`.
- Cache Python: `__pycache__/`.
- Log CatBoost: `catboost_info/`.
- Environment lokal dan secret: `.env`, `.venv/`, `*.key`, `*.pem`.

Beberapa file cache dan ZIP lama sudah tracked sebelum `.gitignore` dibuat. Jika ingin mengeluarkan dari index tanpa menghapus lokal, gunakan:

```bash
git rm --cached -r __pycache__ src/__pycache__
git rm --cached figures/image_result.zip results/result.zip results/result2.zip results/result3.zip results/result4.zip results/result5.zip
```

Jalankan hanya setelah review manual.

## Secret scan

Pencarian pola umum secret (`API_KEY`, `SECRET`, `PASSWORD`, `TOKEN`, private key, `sk-`) tidak menemukan temuan pada file non-zip/non-cache saat audit ini.
