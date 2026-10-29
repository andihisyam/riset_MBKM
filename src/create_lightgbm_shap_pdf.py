"""Create a lecturer-ready PDF for LightGBM hybrid SHAP explainability."""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "stage6_lightgbm_shap"
FIG = ROOT / "figures" / "stage6_lightgbm_shap"
OUT = ROOT / "output" / "pdf" / "shap_hybrid_fts1_lightgbm.pdf"


def fmt(value):
    return f"{float(value):.6f}"


def make_table(rows, widths):
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244062")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b7c3d0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    config = pd.read_json(DATA / "lightgbm_shap_config.json", typ="series")
    importance = pd.read_csv(DATA / "lightgbm_shap_global_importance.csv")
    checks = pd.read_csv(DATA / "lightgbm_shap_additivity.csv")
    comparison = pd.read_csv(DATA / "lightgbm_reproduction_comparison.csv")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=19, leading=23, textColor=colors.HexColor("#244062")))
    styles.add(ParagraphStyle(name="SubCenter", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10, leading=13, textColor=colors.HexColor("#555555")))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=12))
    styles.add(ParagraphStyle(name="HeadingBlue", parent=styles["Heading2"], textColor=colors.HexColor("#244062"), spaceBefore=8, spaceAfter=7))

    story = [
        Spacer(1, 0.3 * cm),
        Paragraph("SHAP Hybrid FTS1-LightGBM Fixed 0,31", styles["TitleCenter"]),
        Spacer(1, 0.2 * cm),
        Paragraph("Reproduksi khusus untuk interpretasi model prediksi konsumsi energi H1-H24", styles["SubCenter"]),
        Spacer(1, 0.5 * cm),
        Paragraph("1. Konfigurasi dan validasi reproduksi", styles["HeadingBlue"]),
        Paragraph(
            "SHAP dihitung untuk dua cabang model: LightGBM Direct dan Residual FTS1-LightGBM. Threshold gate dikunci pada 0,31 kWh. Model dilatih ulang menggunakan parameter Stage 7 yang sama dan disimpan sebagai artifact terpisah; hasil Stage 7 sebelumnya tidak ditimpa.",
            styles["BodyText"],
        ),
        Spacer(1, 0.2 * cm),
    ]
    summary_rows = [["Pemeriksaan", "Hasil"]]
    summary_rows += [
        ["Jumlah forecast origin test", str(len(config["test_origins"]))],
        ["Baris SHAP per cabang", "480 (20 origin x 24 horizon)"],
        ["Aktivasi Residual FTS1", f"{int(config['specialist_activation_count'])} / 480 ({float(config['specialist_activation_share']) * 100:.2f}%)"],
        ["Selisih maksimum reproduksi hybrid", f"{comparison['max_abs_difference'].max():.3e}"],
        ["Error additivity SHAP maksimum", f"{checks['max_additivity_error'].max():.3e}"],
    ]
    story.append(make_table(summary_rows, [8.2 * cm, 8.2 * cm]))
    story += [
        Spacer(1, 0.25 * cm),
        Paragraph("2. SHAP LightGBM Direct", styles["HeadingBlue"]),
        Image(str(FIG / "lightgbm_direct_shap_summary.png"), width=16.8 * cm, height=11.2 * cm),
        Paragraph("Fitur paling dominan pada LightGBM Direct adalah hour, cos_hour, sin_hour, rolling_mean_24, dan rolling_std_24. SHAP menjelaskan kontribusi terhadap output model, bukan hubungan sebab-akibat.", styles["Small"]),
        PageBreak(),
        Paragraph("3. SHAP Residual FTS1-LightGBM", styles["HeadingBlue"]),
        Image(str(FIG / "residual_fts1_lightgbm_shap_summary.png"), width=16.8 * cm, height=11.2 * cm),
        Paragraph("SHAP pada cabang residual menjelaskan output korektor residual. Nilai SHAP ini tidak boleh dibaca sebagai kontribusi langsung terhadap konsumsi akhir tanpa mempertimbangkan prediksi awal FTS dan aturan gate.", styles["Small"]),
        Spacer(1, 0.25 * cm),
        Paragraph("4. Global importance dan interpretasi", styles["HeadingBlue"]),
    ]
    rows = [["Cabang", "Fitur", "Mean absolute SHAP"]]
    for branch, group in importance.groupby("branch"):
        for _, row in group.sort_values("mean_abs_shap", ascending=False).head(8).iterrows():
            rows.append([branch, row["feature"], fmt(row["mean_abs_shap"])])
    story.append(make_table(rows, [6.0 * cm, 6.0 * cm, 4.5 * cm]))
    story += [
        Spacer(1, 0.25 * cm),
        Paragraph("Kesimpulan untuk pelaporan", styles["HeadingBlue"]),
        Paragraph("Interpretasi SHAP LightGBM berhasil direproduksi dengan error additivity mendekati nol dan prediksi hybrid cocok dengan artifact Stage 7 pada toleransi numerik. Gate 0,31 memilih Residual FTS1-LightGBM pada 10,83% prediksi test; sisanya menggunakan LightGBM Direct. Karena SHAP residual menjelaskan cabang korektor dan bukan gate, decision trace tetap diperlukan untuk menjelaskan kapan specialist diaktifkan.", styles["BodyText"]),
        Spacer(1, 0.2 * cm),
        Paragraph("Sumber: data/hourly_features.csv, results/stage7, dan artifact reproduksi results/stage6_lightgbm_shap.", styles["Small"]),
    ]
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=1.4 * cm, leftMargin=1.4 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm, title="SHAP Hybrid FTS1 LightGBM")
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
