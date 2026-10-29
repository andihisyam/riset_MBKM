"""Create a PDF table for all hourly model MAPE results."""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "stage9_arima_sarima" / "ALL_MODELS_MAPE_TABLE.csv"
OUT = ROOT / "output" / "pdf" / "tabel_mape_semua_model_hourly.pdf"


def fmt(value):
    if pd.isna(value) or value == "-":
        return "-"
    return f"{float(value):.4f}"


def text(value):
    return "-" if pd.isna(value) else str(value)


def build_table(data):
    header = [
        "Rank",
        "Model",
        "Algoritma",
        "Pendekatan",
        "FTS",
        "Daily MAPE\n(%)",
        "Hourly MAPE\n(%)",
        "Daily MAE\n(kWh)",
        "Hourly MAE\n(kWh)",
    ]
    rows = [header]
    for _, row in data.iterrows():
        rows.append(
            [
                str(int(row["Peringkat test Daily MAE"])),
                Paragraph(text(row["Model"]), ParagraphStyle("Cell", fontName="Helvetica", fontSize=6.8, leading=8)),
                text(row["Algoritma"]),
                text(row["Pendekatan"]),
                "-" if pd.isna(row["Orde FTS"]) else str(row["Orde FTS"]).replace(".0", ""),
                fmt(row["Test Daily MAPE (%)"]),
                fmt(row["Test Hourly MAPE (%)"]),
                fmt(row["Test Daily MAE (kWh)"]),
                fmt(row["Test Hourly MAE (kWh)"]),
            ]
        )
    table = Table(
        rows,
        colWidths=[1.0 * cm, 6.0 * cm, 2.6 * cm, 2.3 * cm, 1.0 * cm, 2.0 * cm, 2.0 * cm, 2.2 * cm, 2.2 * cm],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244062")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b7c3d0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 1), (0, -1), "CENTER"),
                ("ALIGN", (4, 1), (-1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def main():
    data = pd.read_csv(DATA)
    assert len(data) == 57, f"Ekspektasi 57 model, ditemukan {len(data)}"
    data = data.sort_values("Test Daily MAE (kWh)", kind="stable").reset_index(drop=True)
    data["Peringkat test Daily MAE"] = range(1, len(data) + 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22, textColor=colors.HexColor("#244062")))
    styles.add(ParagraphStyle(name="SubCenter", parent=styles["Normal"], alignment=TA_CENTER, fontSize=9, leading=12, textColor=colors.HexColor("#555555")))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=11))

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=landscape(A4),
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.1 * cm,
        bottomMargin=1.1 * cm,
        title="Tabel MAPE Semua Model Hourly",
    )
    daily_best = data.loc[data["Test Daily MAPE (%)"].idxmin()]
    hourly_best = data.loc[data["Test Hourly MAPE (%)"].idxmin()]
    story = [
        Paragraph("Tabel MAPE Semua Model Prediksi Konsumsi Energi Hourly", styles["TitleCenter"]),
        Spacer(1, 0.2 * cm),
        Paragraph("57 konfigurasi | H1-H24 | hasil test asli | tidak ada training ulang", styles["SubCenter"]),
        Spacer(1, 0.35 * cm),
        Paragraph(
            "Tabel diurutkan berdasarkan Test Daily MAE. MAPE ditampilkan sebagai metrik pendukung dan perlu dibaca bersama MAE karena nilai aktual yang mendekati nol dapat memperbesar MAPE.",
            styles["Small"],
        ),
        Spacer(1, 0.25 * cm),
        build_table(data),
        Spacer(1, 0.25 * cm),
        Paragraph(
            f"MAPE harian terendah: {daily_best['Model']} ({daily_best['Test Daily MAPE (%)']:.4f}%). "
            f"MAPE hourly terendah: {hourly_best['Model']} ({hourly_best['Test Hourly MAPE (%)']:.4f}%). "
            "Pemilihan model penelitian tetap harus mengacu pada validation, bukan hanya MAPE test.",
            styles["Small"],
        ),
    ]
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
