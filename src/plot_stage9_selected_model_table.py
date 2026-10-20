"""Create the report chart for the selected model comparison table."""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "stage9_arima_sarima" / "FINAL_MODEL_COMPARISON.csv"
OUT = ROOT / "results" / "stage9_arima_sarima" / "figures" / "selected_models_daily_mae.png"


def main() -> None:
    models = [
        "Residual FTS3-RandomForest",
        "Hybrid FTS1-LightGBM Fixed 0.31",
        "LightGBM Direct",
        "RandomForest Direct",
        "CatBoost Direct",
        "SARIMA (1, 0, 1) (1, 0, 1, 24)",
        "ARIMA (2, 1, 1) None",
        "SVR Direct",
        "XGBoost Direct",
    ]
    data = pd.read_csv(DATA).set_index("model").loc[models].reset_index()
    labels = ["\n".join(textwrap.wrap(str(x), 18)) for x in data["model"]]
    values = data["daily_mae_kWh"]
    lowest = values.idxmin()

    fig, ax = plt.subplots(figsize=(15, 8))
    colors = ["#d95f02" if i == lowest else "#a6a6a6" for i in range(len(data))]
    bars = ax.bar(range(len(data)), values, color=colors, width=0.72)
    ax.set_title("Perbandingan Test Daily MAE Model Terpilih", fontsize=16, pad=18)
    ax.set_ylabel("Test Daily MAE (kWh)")
    ax.set_xticks(range(len(data)), labels, rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(values) * 1.18)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.018,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.text(
        0.99,
        0.98,
        "Warna oranye = nilai test terendah / terbaik secara deskriptif",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#7f2704",
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
