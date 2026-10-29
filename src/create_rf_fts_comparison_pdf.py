"""Create a lecturer-ready PDF comparing Random Forest, FTS, and other models."""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "stage9_arima_sarima" / "ALL_HOURLY_MODELS.csv"
OUT = ROOT / "output" / "pdf" / "laporan_perbandingan_random_forest_fts.pdf"
FIG = ROOT / "results" / "stage9_arima_sarima" / "figures" / "selected_models_daily_mae.png"


def fmt(value):
    return "-" if pd.isna(value) else f"{float(value):.4f}"


def make_table(rows, widths):
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244062")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c3d0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    data = data[data["split"].fillna("test").eq("test")]
    direct_names = ["LightGBM Direct", "RandomForest Direct", "CatBoost Direct", "SVR Direct", "XGBoost Direct"]
    direct = data[data.model.isin(direct_names)].set_index("model").loc[direct_names].reset_index()
    rf_names = [
        "RandomForest Direct",
        "Residual FTS1-RandomForest",
        "Residual FTS2-RandomForest",
        "Residual FTS3-RandomForest",
        "Hybrid FTS1-RandomForest Fixed 0.31",
        "Hybrid FTS1-RandomForest Validation threshold",
        "Hybrid FTS2-RandomForest Fixed 0.31",
        "Hybrid FTS2-RandomForest Validation threshold",
        "Hybrid FTS3-RandomForest Fixed 0.31",
        "Hybrid FTS3-RandomForest Validation threshold",
    ]
    rf = data[data.model.isin(rf_names)].set_index("model").loc[rf_names].reset_index()

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=20, leading=25, textColor=colors.HexColor("#244062")))
    styles.add(ParagraphStyle(name="SubCenter", parent=styles["Normal"], alignment=TA_CENTER, fontSize=11, leading=15, textColor=colors.HexColor("#555555")))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=12))
    styles.add(ParagraphStyle(name="HeadingBlue", parent=styles["Heading2"], textColor=colors.HexColor("#244062"), spaceBefore=10, spaceAfter=7))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=1.6 * cm, leftMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="Perbandingan Random Forest dan Weighted FTS")
    story = [
        Spacer(1, 2.0 * cm),
        Paragraph("Perbandingan Random Forest, Weighted FTS, dan Model Machine Learning", styles["TitleCenter"]),
        Spacer(1, 0.5 * cm),
        Paragraph("Prediksi konsumsi energi hourly H1-H24", styles["SubCenter"]),
        Spacer(1, 0.8 * cm),
        Paragraph("Laporan ringkas berbasis hasil eksperimen Stage 1-7 dan benchmark ARIMA/SARIMA Stage 9", styles["SubCenter"]),
        Spacer(1, 4.0 * cm),
        Paragraph("Metrik utama: Daily MAE (kWh)", styles["SubCenter"]),
        PageBreak(),
        Paragraph("1. Tujuan laporan", styles["HeadingBlue"]),
        Paragraph("Laporan ini merangkum posisi Random Forest terhadap model Direct lain serta pengaruh Weighted Fuzzy Time Series (FTS) melalui pendekatan residual dan hybrid threshold. Semua angka diambil dari artifact eksperimen yang sudah tersedia; tidak ada pelatihan ulang pada saat pembuatan PDF.", styles["BodyText"]),
        Spacer(1, 0.25 * cm),
        Paragraph("Evaluasi menggunakan forecast horizon H1-H24 dan forecast origin kronologis yang sama pada Stage 7. Daily MAE menjadi metrik utama karena mengukur kesalahan total energi harian, sedangkan Hourly MAE digunakan sebagai metrik pendukung.", styles["BodyText"]),
        Paragraph("2. Perbandingan model Direct", styles["HeadingBlue"]),
    ]
    direct_rows = [["Model", "Validation\nDaily MAE", "Test\nDaily MAE", "Test\nHourly MAE"]]
    for _, row in direct.iterrows():
        direct_rows.append([row["model"], fmt(row["validation_daily_mae"]), fmt(row["Daily MAE"]), fmt(row["Hourly MAE"])])
    story.append(make_table(direct_rows, [6.8 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm]))
    story += [Spacer(1, 0.25 * cm), Paragraph("Random Forest Direct memiliki performa yang sangat dekat dengan LightGBM Direct dan lebih baik daripada SVR serta XGBoost pada Test Daily MAE. Namun, LightGBM Direct masih sedikit lebih rendah daripada Random Forest Direct.", styles["Small"]), PageBreak()]
    story += [Paragraph("3. Random Forest dengan FTS", styles["HeadingBlue"])]
    rf_rows = [["Konfigurasi", "Validation\nDaily MAE", "Test\nDaily MAE", "Test\nHourly MAE"]]
    for _, row in rf.iterrows():
        rf_rows.append([row["model"], fmt(row["validation_daily_mae"]), fmt(row["Daily MAE"]), fmt(row["Hourly MAE"])])
    story.append(make_table(rf_rows, [7.5 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm]))
    story += [Spacer(1, 0.3 * cm), Paragraph("Residual FTS3-RandomForest menghasilkan Test Daily MAE terendah, yaitu 3.0977 kWh. Akan tetapi, Validation Daily MAE-nya 5.7880 kWh. Karena pemilihan model harus berdasarkan validation, hasil test tersebut harus dilaporkan sebagai hasil eksploratif, bukan sebagai pemenang final yang stabil.", styles["Small"]), PageBreak()]
    story += [Paragraph("4. Visualisasi hasil", styles["HeadingBlue"])]
    if FIG.exists():
        story.append(Image(str(FIG), width=17.6 * cm, height=9.3 * cm))
    story += [Spacer(1, 0.25 * cm), Paragraph("Grafik menampilkan beberapa model utama berdasarkan Test Daily MAE. Nilai lebih rendah menunjukkan kesalahan total harian yang lebih kecil.", styles["Small"]), Paragraph("5. Kesimpulan", styles["HeadingBlue"]), Paragraph("Random Forest layak dipertahankan sebagai baseline kuat. Penambahan FTS tidak otomatis meningkatkan performa Random Forest pada semua orde dan aturan gate. Konfigurasi Residual FTS3-RandomForest terlihat paling baik pada test, tetapi tidak didukung oleh validation. Kesimpulan yang paling dapat dipertanggungjawabkan adalah menjadikan Random Forest Direct atau model Direct terbaik berdasarkan validation sebagai pembanding utama, sementara FTS-RF dilaporkan sebagai eksperimen hybrid dan analisis eksploratif.", styles["BodyText"]), Spacer(1, 0.25 * cm), Paragraph("Sumber artifact: results/stage7, results/stage9_arima_sarima, reports/stage7/stage7_final_report.md. Stage 8 memakai resolusi 10 menit dan satuan Wh sehingga tidak dicampur ke tabel hourly ini.", styles["Small"])]
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
