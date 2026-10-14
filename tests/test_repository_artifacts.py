from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_core_artifacts_exist():
    required = [
        'energydata_complete.csv',
        'data/hourly_energy_clean.csv',
        'data/hourly_features.csv',
        'results/stage7/stage7_residual_comparison.csv',
        'results/stage7_1_audit/AUDIT_SUMMARY.md',
        'docs/methodology.md',
        'notebooks/00_complete_research_walkthrough.ipynb',
    ]
    for path in required:
        p = ROOT / path
        assert p.exists(), path
        assert p.stat().st_size > 0, path

def test_verified_hourly_count():
    hourly = pd.read_csv(ROOT / 'data/hourly_energy_clean.csv')
    assert len(hourly) == 3289

def test_residual_rf_result_available():
    df = pd.read_csv(ROOT / 'results/stage7/stage7_residual_comparison.csv')
    row = df[df['model'].eq('Residual FTS3-RandomForest')]
    assert not row.empty
    assert abs(float(row.iloc[0]['Daily MAE']) - 3.097700) < 1e-4
