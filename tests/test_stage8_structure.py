from pathlib import Path


def test_stage8_script_and_reference_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src" / "stage8_paper_benchmark.py").exists()
    assert (root / "run_stage8.py").exists()
    assert (root / "candanedo2017.pdf").exists()
