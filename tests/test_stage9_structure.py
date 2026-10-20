from pathlib import Path


def test_stage9_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src" / "stage9_arima_sarima_benchmark.py").exists()
    assert (root / "run_stage9.py").exists()
    assert (root / "results" / "stage7" / "stage7_final_model_ranking.csv").exists()
