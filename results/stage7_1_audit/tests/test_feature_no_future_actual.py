from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
summary = pd.read_csv(ROOT / 'results' / 'stage7_1_audit' / 'feature_future_value_audit_summary.csv')
assert summary['max_abs_diff'].max() < 1e-9
assert summary['future_flags'].sum() == 0
