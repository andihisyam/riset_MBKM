# Household Energy Forecasting Preliminary Experiments

This project uses the Appliances Energy Prediction dataset (`energydata_complete.csv`) to prepare a forecasting design for household energy consumption.

The current pipeline covers:

- dataset audit and EDA validation;
- target distribution, spike identification, and temporal analysis;
- stationarity checks with ADF and KPSS;
- predictor correlation checks, including `rv1` and `rv2`;
- physically correct hourly aggregation;
- chronological 70/15/15 split;
- naive, daily seasonal naive, and weekly seasonal naive baselines;
- daily total energy evaluation from hourly predictions.

Run:

```bash
python run_preliminary.py
```

Main outputs:

- `results/baseline_results.csv`
- `results/daily_baseline_results.csv`
- `results/hourly_energy.csv`
- `figures/*.png`
- `reports/preliminary_experiment_report.md`

The next stage should add hourly feature engineering, XGBoost standalone, FTS partition experiments, FTS models, hybrid FTS-XGBoost, SHAP, and error analysis.
