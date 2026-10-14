# Stage 6 Complete Audit

## Completeness

Required Stage 6 result CSV files and Stage 6 Markdown reports are present. Missing/unreproducible files: 0.

## Reproduction

- XGBoost Direct Daily MAE: 3.289850
- Threshold Gate B Daily MAE: 3.202802
- Threshold Gate B Daily MAPE: 23.212342%
- Specialist share: 6.875000%
- Non-overlap origins: 20

These match the Stage 4-5 checkpoint within floating-point tolerance.

## Additivity

- XGBoost Direct max additivity error: 0.000001549721
- Residual corrector max additivity error: 0.000000536442
- Daily XGBoost SHAP reconstruction max error: 0.000002923975
- Daily gate decomposition max error: 0.000000000000

## Dominant Time Features

Top SHAP features:

| feature         |   mean_abs_shap |
|:----------------|----------------:|
| hour            |        0.071305 |
| cos_hour        |        0.060992 |
| sin_hour        |        0.051994 |
| rolling_mean_24 |        0.025619 |
| rolling_std_24  |        0.020929 |
| rolling_std_6   |        0.020067 |
| sin_dow         |        0.019218 |
| day_of_week     |        0.016909 |
| rolling_mean_12 |        0.016654 |
| lag_1           |        0.015578 |

`hour`, `cos_hour`, and `sin_hour` are dominant because the trained direct horizon models learn strong time-of-day structure from the full chronological train/validation data. The Stage 6 interpretation sample uses non-overlap origins with origin hour 3; therefore horizon interpretation must be read together with `horizon_target_hour_audit.csv`. SHAP by horizon should not be interpreted as horizon length alone; target hour composition is entangled with horizon.

## Threshold Gate B

Gate B is deterministic: choose Residual FTS-XGBoost only when `XGBoost Direct prediction <= 0.310000`. This is the training q33 low-regime threshold used by Gate B, not the Stage 5 high/spike thresholds such as p80 or p95.

- Gate condition audit passed: True
- Specialist activations: 33
- Specialist improved hourly AE: 4
- Specialist worsened hourly AE: 29

Full specialist decision trace is in `results/stage6/specialist_activation_trace.csv`.

## Files Not Reproduced

None.
