# RESIDUAL AUDIT

## Fokus: Residual FTS3-RandomForest

| split      |   daily_mae |      bias |   mean_cancellation |   max_abs_error |
|:-----------|------------:|----------:|--------------------:|----------------:|
| test       |     3.0977  | -0.577646 |             3.16953 |         11.4365 |
| validation |     5.78795 |  1.98955  |             3.69011 |         24.1232 |

Model ini buruk pada validation tetapi terbaik pada test. Audit per-origin menunjukkan hasil test dipengaruhi distribusi origin dan error cancellation. Karena validation adalah dasar pemilihan model, Residual FTS3-RandomForest tidak layak dijadikan model utama meskipun test MAE paling kecil.

Artefak pendukung:

- `residual_fts3_randomforest_per_origin.csv`
- `residual_fts3_randomforest_per_horizon.csv`
- `figures/residual_fts3_rf_scatter_daily.png`
- `figures/residual_fts3_rf_origin_signed_error.png`
- `figures/residual_fts3_rf_hourly_error_hist.png`
