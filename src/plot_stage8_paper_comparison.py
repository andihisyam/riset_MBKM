"""Create a transparent MAE comparison between paper references and Stage 8."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "stage8_paper_benchmark"
OUT = OUT_DIR / "figures" / "paper_vs_stage8_mae.png"


def main() -> None:
    paper = pd.read_csv(OUT_DIR / "PAPER_REFERENCE_RESULTS.csv")
    track_b = pd.read_csv(OUT_DIR / "TRACK_B_RESULTS.csv")
    rows = [
        {"label": "Paper\nRandom Forest", "mae": paper.loc[paper.model.eq("Random Forest"), "MAE_Wh"].iloc[0], "source": "Paper"},
        {"label": "Penelitian ini\nHybrid Weighted FTS-RF", "mae": track_b.loc[(track_b.variant.eq("B1_history_time")) & track_b.fts_order.eq(2) & track_b.split.eq("test") & track_b.model.eq("Hybrid FTS-RF residual"), "MAE_Wh"].iloc[0], "source": "Penelitian ini"},
    ]
    data = pd.DataFrame(rows)
    data.to_csv(OUT_DIR / "PAPER_RF_VS_STAGE8_COMPARISON.csv", index=False)
    best = data.mae.idxmin()
    colors = ["#d95f02" if i == best else "#a6a6a6" for i in range(len(data))]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(range(len(data)), data.mae, color=colors, width=0.68)
    ax.set_title("Perbandingan MAE Random Forest Paper dan Metode Penelitian", fontsize=16, pad=18)
    ax.set_ylabel("Test MAE (Wh)")
    ax.set_xticks(range(len(data)), data.label)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, data.mae.max() * 1.2)
    for bar, value in zip(bars, data.mae):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    ax.text(0.99, 0.98, "Oranye = MAE terendah", transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#7f2704")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
