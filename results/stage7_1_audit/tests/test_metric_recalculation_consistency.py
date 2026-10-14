from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
df = pd.read_csv(ROOT / 'results' / 'stage7_1_audit' / 'metric_csv_consistency_audit.csv')
assert df['daily_mae_abs_diff'].max() < 1e-9
assert df['hourly_mae_abs_diff'].max() < 1e-9
