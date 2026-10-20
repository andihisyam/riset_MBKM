# Stage 8 Paper-Aligned Benchmark

Script: `src/stage8_paper_benchmark.py`. Track A uses 10-minute data, 75/25 chronological split, and paper-style environmental/time predictors. It is approximate because the original CARET seed/split was not recovered. Track B uses chronological rolling one-step prediction with validation-selected FTS order and Random Forest parameters.

The paper reference values are stored in `PAPER_REFERENCE_RESULTS.csv`; they are not reproduced results. Main forecasting conclusions must use Track B.
