"""Create a compact PDF for the five models shown in the 24-hour chart."""

from pathlib import Path

import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "stage9_arima_sarima" / "ALL_HOURLY_MODELS.csv"
HOURLY_PREDICTIONS = ROOT / "results" / "stage7" / "stage7_hourly_predictions.csv"
DAILY_PREDICTIONS = ROOT / "results" / "stage7" / "stage7_daily_predictions.csv"
FIGURE = ROOT / "figures" / "stage7" / "stage7_06_actual_vs_prediction_24h.png"
OUT = ROOT / "output" / "pdf" / "perbandingan_5_model_terbaik_24_jam.pdf"

MODELS = [
    "Residual FTS3-RandomForest",
    "Hybrid FTS1-LightGBM Fixed 0.31",
    "Residual FTS3-LightGBM",
    "LightGBM Direct",
    "Hybrid FTS1-LightGBM Validation threshold",
]


def fmt(value):
    return "-" if pd.isna(value) else f"{float(value):.4f}"


def r2_score(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.sum((actual - actual.mean()) ** 2)
    return np.nan if denominator == 0 else 1 - np.sum((actual - predicted) ** 2) / denominator


def main():
    data = pd.read_csv(DATA).set_index("model")
    selected = data.loc[MODELS].reset_index()
    hourly = pd.read_csv(HOURLY_PREDICTIONS)
    daily = pd.read_csv(DAILY_PREDICTIONS)
    hourly = hourly[(hourly["split"] == "test") & hourly["model"].isin(MODELS)]
    daily = daily[(daily["split"] == "test") & daily["model"].isin(MODELS)]
    hourly_r2 = {name: r2_score(g["actual"], g["predicted"]) for name, g in hourly.groupby("model")}
    daily_r2 = {name: r2_score(g["actual_daily"], g["predicted_daily"]) for name, g in daily.groupby("model")}
    assert len(hourly) == 5 * 20 * 24, f"Prediksi hourly tidak lengkap: {len(hourly)}"
    assert len(daily) == 5 * 20, f"Prediksi daily tidak lengkap: {len(daily)}"
    OUT.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22, textColor=colors.HexColor("#244062")))
    styles.add(ParagraphStyle(name="SubCenter", parent=styles["Normal"], alignment=TA_CENTER, fontSize=9, leading=12, textColor=colors.HexColor("#555555")))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=12))
    styles.add(ParagraphStyle(name="Cell", parent=styles["BodyText"], fontSize=7.3, leading=9))
    styles.add(ParagraphStyle(name="HeadingBlue", parent=styles["Heading2"], textColor=colors.HexColor("#244062"), spaceBefore=8, spaceAfter=6))

    table_rows = [[
        "Model", "Pendekatan", "Validation\nDaily MAE", "Daily\nMAPE", "Hourly\nMAPE", "Daily\nMAE", "Hourly\nMAE", "Daily\nRMSE", "Hourly\nRMSE", "Daily\nR2", "Hourly\nR2",
    ]]
    for _, row in selected.iterrows():
        table_rows.append([
            Paragraph(str(row["model"]), styles["Cell"]),
            str(row["family"]),
            fmt(row["validation_daily_mae"]),
            fmt(row["Daily MAPE"]),
            fmt(row["Hourly MAPE"]),
            fmt(row["Daily MAE"]),
            fmt(row["Hourly MAE"]),
            fmt(row["Daily RMSE"]),
            fmt(row["Hourly RMSE"]),
            fmt(daily_r2[row["model"]]),
            fmt(hourly_r2[row["model"]]),
        ])

    table = Table(table_rows, colWidths=[6.0 * cm, 1.7 * cm, 2.0 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.8 * cm, 1.8 * cm, 1.5 * cm, 1.5 * cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244062")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b7c3d0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    doc = SimpleDocTemplate(str(OUT), pagesize=landscape(A4), rightMargin=1.0 * cm, leftMargin=1.0 * cm, topMargin=0.9 * cm, bottomMargin=0.9 * cm, title="Perbandingan Lima Model Terbaik 24 Jam")
    story = [
        Paragraph("Perbandingan Lima Model Terbaik pada Prediksi 24 Jam", styles["TitleCenter"]),
        Spacer(1, 0.15 * cm),
        Paragraph("Aktual vs prediksi H1-H24 dan metrik test dari hasil eksperimen asli", styles["SubCenter"]),
        Spacer(1, 0.3 * cm),
    ]
    if FIGURE.exists():
        story.append(Image(str(FIGURE), width=25.5 * cm, height=8.0 * cm))
    story += [
        Spacer(1, 0.2 * cm),
        Paragraph("Model yang dibandingkan", styles["HeadingBlue"]),
        table,
        Spacer(1, 0.25 * cm),
        Paragraph(
            "Grafik menggunakan forecast origin pertama pada data test. Tabel menggunakan agregat 20 forecast origin test. RMSE dan R2 dihitung dari prediksi test individual. Residual FTS3-RandomForest memiliki Test Daily MAE terendah pada lima model ini, tetapi validation Daily MAE-nya lebih buruk; karena itu hasilnya tetap perlu disebut sebagai hasil test deskriptif, bukan otomatis model utama.",
            styles["Small"],
        ),
    ]
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
