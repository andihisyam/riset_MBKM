"""Plot actual versus Residual FTS3-RandomForest predictions on test origins."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "stage7" / "stage7_hourly_predictions.csv"
OUT = ROOT / "figures" / "stage7" / "residual_fts3_rf_actual_vs_predicted_24h.png"


def main() -> None:
    data = pd.read_csv(DATA, parse_dates=["origin"])
    data = data[(data["model"] == "Residual FTS3-RandomForest") & (data["split"] == "test")].copy()
    if len(data) != 20 * 24:
        raise ValueError(f"Expected 480 test rows, found {len(data)}")
    data["target_time"] = data["origin"] + pd.to_timedelta(data["horizon"], unit="h")
    data = data.sort_values("target_time")

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(data["target_time"], data["actual"], color="#244062", linewidth=1.7, label="Aktual")
    ax.plot(data["target_time"], data["predicted"], color="#d95f02", linewidth=1.5, label="Residual FTS3-RandomForest")
    ax.set_title("Aktual vs Prediksi Residual FTS3-RandomForest pada Data Test", fontsize=15, pad=14)
    ax.set_xlabel("Waktu target")
    ax.set_ylabel("Konsumsi energi (kWh)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
