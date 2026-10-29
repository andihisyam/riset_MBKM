"""Create the combined five-model comparison and LightGBM SHAP PDF."""

from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
MODEL_DATA = ROOT / "results" / "stage9_arima_sarima" / "ALL_HOURLY_MODELS.csv"
HOURLY = ROOT / "results" / "stage7" / "stage7_hourly_predictions.csv"
DAILY = ROOT / "results" / "stage7" / "stage7_daily_predictions.csv"
CHART = ROOT / "figures" / "stage7" / "stage7_06_actual_vs_prediction_24h.png"
SHAP_DIR = ROOT / "figures" / "stage6_lightgbm_shap"
SHAP_RESULTS = ROOT / "results" / "stage6_lightgbm_shap"
OUT = ROOT / "output" / "pdf" / "perbandingan_5_model_dan_shap_lightgbm_031.pdf"

MODELS = [
    "Residual FTS3-RandomForest",
    "Hybrid FTS1-LightGBM Fixed 0.31",
    "Residual FTS3-LightGBM",
    "LightGBM Direct",
    "Hybrid FTS1-LightGBM Validation threshold",
]


def fmt(value):
    return "-" if pd.isna(value) else f"{float(value):.4f}"


def r2(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2)


def make_table(rows, widths, font_size=7.5, highlight_row=None):
    table = Table(rows, colWidths=widths, repeatRows=1)
    styles = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244062")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b7c3d0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight_row is not None:
        styles.extend([
            ("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#ffe0a3")),
            ("FONTNAME", (0, highlight_row), (-1, highlight_row), "Helvetica-Bold"),
            ("LINEABOVE", (0, highlight_row), (-1, highlight_row), 1.2, colors.HexColor("#d97706")),
            ("LINEBELOW", (0, highlight_row), (-1, highlight_row), 1.2, colors.HexColor("#d97706")),
        ])
    table.setStyle(TableStyle(styles))
    return table


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(MODEL_DATA).set_index("model").loc[MODELS].reset_index()
    hourly = pd.read_csv(HOURLY)
    daily = pd.read_csv(DAILY)

    rows = [["Model", "Pendekatan", "Val Daily\nMAE", "Daily\nMAPE", "Hourly\nMAPE", "Daily\nMAE", "Hourly\nMAE", "Daily\nRMSE", "Hourly\nRMSE", "Daily\nR2", "Hourly\nR2"]]
    for _, row in metrics.iterrows():
        hp = hourly[(hourly["model"] == row["model"]) & (hourly["split"] == "test")]
        dp = daily[(daily["model"] == row["model"]) & (daily["split"] == "test")]
        rows.append([
            row["model"], row["family"], fmt(row["validation_daily_mae"]), fmt(row["Daily MAPE"]), fmt(row["Hourly MAPE"]),
            fmt(row["Daily MAE"]), fmt(row["Hourly MAE"]), fmt(row["Daily RMSE"]), fmt(row["Hourly RMSE"]),
            fmt(r2(dp["actual_daily"], dp["predicted_daily"])), fmt(r2(hp["actual"], hp["predicted"])),
        ])

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22, textColor=colors.HexColor("#244062")))
    styles.add(ParagraphStyle(name="SubCenter", parent=styles["Normal"], alignment=TA_CENTER, fontSize=9, leading=12, textColor=colors.HexColor("#555555")))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=12))
    styles.add(ParagraphStyle(name="HeadingBlue", parent=styles["Heading2"], textColor=colors.HexColor("#244062"), spaceBefore=8, spaceAfter=6))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=1.0 * cm, leftMargin=1.0 * cm, topMargin=1.0 * cm, bottomMargin=1.0 * cm, title="Perbandingan Lima Model dan SHAP LightGBM")
    story = [
        Paragraph("Perbandingan Lima Model Terbaik dan SHAP LightGBM", styles["TitleCenter"]),
        Spacer(1, 0.15 * cm),
        Paragraph("Prediksi konsumsi energi H1-H24 berdasarkan hasil eksperimen asli", styles["SubCenter"]),
        Spacer(1, 0.25 * cm),
        Image(str(CHART), width=17.0 * cm, height=8.5 * cm),
        Paragraph("Model yang dibandingkan", styles["HeadingBlue"]),
        make_table(rows, [5.0 * cm, 1.5 * cm, 1.5 * cm, 1.25 * cm, 1.25 * cm, 1.25 * cm, 1.25 * cm, 1.35 * cm, 1.35 * cm, 1.1 * cm, 1.1 * cm], 6.5, highlight_row=2),
        Paragraph("Warna kuning menandai konfigurasi yang dipilih untuk analisis SHAP: Hybrid FTS1-LightGBM Fixed 0.31.", styles["Small"]),
        Spacer(1, 0.18 * cm),
        Paragraph("Catatan threshold 0,31", styles["HeadingBlue"]),
        Paragraph("Angka 0,31 adalah threshold atau ambang keputusan dalam Hybrid FTS1-LightGBM Fixed 0.31, bukan orde FTS. Jika prediksi LightGBM Direct <= 0,31 kWh, sistem menggunakan Residual FTS1-LightGBM. Jika prediksi Direct > 0,31 kWh, sistem menggunakan LightGBM Direct. Threshold ini dikunci sebagai pembanding tetap dan tidak ditentukan dari nilai aktual test.", styles["Small"]),
        PageBreak(),
        Paragraph("SHAP untuk LightGBM Direct", styles["HeadingBlue"]),
        Image(str(SHAP_DIR / "lightgbm_direct_shap_summary.png"), width=17.0 * cm, height=11.3 * cm),
        Paragraph("Fitur yang paling dominan adalah hour, cos_hour, sin_hour, rolling_mean_24, dan rolling_std_24. SHAP menunjukkan kontribusi fitur terhadap output model, bukan hubungan sebab-akibat.", styles["Small"]),
        Spacer(1, 0.3 * cm),
        Paragraph("SHAP untuk Residual FTS1-LightGBM", styles["HeadingBlue"]),
        Image(str(SHAP_DIR / "residual_fts1_lightgbm_shap_summary.png"), width=17.0 * cm, height=11.3 * cm),
        Paragraph("SHAP residual menjelaskan output korektor residual. Interpretasinya harus dibaca bersama prediksi awal FTS dan aturan threshold gate.", styles["Small"]),
        PageBreak(),
        Paragraph("Interpretasi akhir", styles["HeadingBlue"]),
        Paragraph("Pada konfigurasi Hybrid FTS1-LightGBM Fixed 0.31, Residual FTS1-LightGBM aktif pada 52 dari 480 prediksi test atau 10,83%. Sebagian besar prediksi tetap menggunakan LightGBM Direct. Karena model hybrid memiliki dua cabang, SHAP dibuat terpisah untuk Direct dan Residual; keputusan pemilihan cabang dijelaskan oleh threshold gate.", styles["BodyText"]),
        Spacer(1, 0.25 * cm),
        Paragraph("Pemeriksaan reproduksi menunjukkan selisih prediksi maksimum terhadap artifact Stage 7 sebesar 2,22 x 10^-16 dan error additivity SHAP maksimum 3,997 x 10^-15.", styles["BodyText"]),
        Spacer(1, 0.3 * cm),
        Paragraph("Lampiran grafik global importance", styles["HeadingBlue"]),
        Image(str(SHAP_DIR / "lightgbm_direct_shap_importance.png"), width=8.2 * cm, height=6.4 * cm),
        Image(str(SHAP_DIR / "residual_fts1_lightgbm_shap_importance.png"), width=8.2 * cm, height=6.4 * cm),
        Spacer(1, 0.1 * cm),
        Paragraph("Sumber: artifact Stage 7 dan reproduksi SHAP LightGBM pada results/stage6_lightgbm_shap.", styles["Small"]),
    ]
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
