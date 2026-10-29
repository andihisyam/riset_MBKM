from __future__ import annotations

import csv
import html
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "interactive_report"
ZIP_PATH = ROOT / "interactive_report_stage6.zip"
TARGET = "Appliances_kWh"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def pct(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def int_id(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def data_table(rows: list[dict[str, object]], columns: list[str], limit: int | None = None) -> str:
    shown = rows if limit is None else rows[:limit]
    if not columns:
        return ""
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = []
    for row in shown:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns) + "</tr>")
    return f"<div class='table-scroll'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def make_generated_figures(hourly: pd.DataFrame, gate: pd.DataFrame, fts_bounds: list[float], model_rows: list[dict[str, str]]) -> dict[str, str]:
    gen = OUT / "assets" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    first_origin = int(gate["origin_index"].iloc[0])
    origin_ts = pd.to_datetime(gate["origin"].iloc[0])
    window = hourly.loc[first_origin - 48 : first_origin + 24].copy()
    pred = gate[gate["origin_index"] == first_origin].sort_values("horizon")
    pred_times = pd.date_range(origin_ts + pd.Timedelta(hours=1), periods=24, freq="h")

    plt.figure(figsize=(10, 4.8))
    plt.plot(window["date"], window[TARGET], color="#6b7890", linewidth=2, label="Aktual historis dan aktual berikutnya")
    plt.plot(pred_times, pred["predicted"], color="#1b65a7", linewidth=2.5, marker="o", markersize=3, label="Prediksi Gate B H1-H24")
    plt.axvline(origin_ts, color="#d84c3f", linestyle="--", linewidth=2, label="Forecast origin")
    plt.fill_between(pred_times, pred["predicted"], alpha=0.12, color="#1b65a7")
    plt.title("Contoh prediksi 24 jam dari origin nyata Stage 6")
    plt.ylabel("kWh")
    plt.xlabel("Waktu")
    plt.legend(loc="upper left", fontsize=8)
    plt.xticks(rotation=25, ha="right")
    save_fig(gen / "story_forecast_origin.png")

    train_end = int(len(hourly) * 0.70)
    train = hourly.loc[: train_end - 1, TARGET]
    plt.figure(figsize=(10, 4.8))
    plt.hist(train, bins=45, color="#c9d8ec", edgecolor="#ffffff")
    for i, b in enumerate(fts_bounds):
        plt.axvline(b, color="#174f86", linewidth=0.8, alpha=0.75)
        if i in [0, len(fts_bounds) - 1]:
            plt.text(b, plt.ylim()[1] * 0.92, f"{b:.2f}", rotation=90, fontsize=8, color="#174f86")
    plt.title("Distribusi konsumsi hourly dengan batas 15 interval quantile")
    plt.xlabel("Konsumsi per jam (kWh)")
    plt.ylabel("Jumlah observasi training")
    save_fig(gen / "fts_quantile_distribution.png")

    plt.figure(figsize=(8.5, 4.7))
    plt.hist(hourly[TARGET], bins=50, color="#d7e8dd", edgecolor="#ffffff")
    plt.title("Distribusi konsumsi listrik per jam setelah preprocessing")
    plt.xlabel("kWh per jam")
    plt.ylabel("Jumlah observasi")
    save_fig(gen / "hourly_distribution.png")

    baseline = read_rows(ROOT / "results" / "baseline_24h_fixed_origin.csv")
    selected = [
        ("Weekly Seasonal", "Weekly Seasonal Naive", baseline),
        ("XGBoost Direct", "XGBoost Direct", model_rows),
        ("Stage 4 Threshold Gate B", "Threshold Gate B", model_rows),
    ]
    labels, values = [], []
    for key, label, source in selected:
        row = next((r for r in source if r.get("model") == key), None)
        if row:
            labels.append(label)
            values.append(float(row["Daily MAE"]))
    plt.figure(figsize=(8.5, 4.7))
    bars = plt.bar(labels, values, color=["#9aa8b8", "#1b65a7", "#3d9b72"])
    plt.title("Perbandingan Daily MAE")
    plt.ylabel("Daily MAE (kWh), lebih kecil lebih baik")
    plt.ylim(0, max(values) * 1.25)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06, f"{value:.3f}", ha="center", fontweight="bold")
    save_fig(gen / "daily_mae_story.png")

    return {
        "forecast": "assets/generated/story_forecast_origin.png",
        "fts_distribution": "assets/generated/fts_quantile_distribution.png",
        "hourly_distribution": "assets/generated/hourly_distribution.png",
        "mae_bar": "assets/generated/daily_mae_story.png",
    }


def term_button(term: str) -> str:
    return f"<button class='term' data-term='{html.escape(term)}'>{html.escape(term)}</button>"


def narrative(text: str) -> str:
    return f"<details class='narrative'><summary>Kalimat untuk menjelaskan kepada dosen</summary><p>{html.escape(text)}</p></details>"


def tech(text: str) -> str:
    return f"<details class='technical'><summary>Lihat penjelasan teknis</summary><p>{html.escape(text)}</p></details>"


def slide(slide_id: str, number: str, title: str, lead: str, visual: str, body: str, takeaway: str, transition: str) -> str:
    return f"""
      <section class="slide" id="{slide_id}" data-title="{html.escape(title)}">
        <p class="eyebrow">Halaman {number}</p>
        <h1>{html.escape(title)}</h1>
        <p class="lead">{lead}</p>
        <div class="visual">{visual}</div>
        <div class="explain">{body}</div>
        <p class="takeaway"><b>Pesan utama:</b> {takeaway}</p>
        <p class="transition">{transition}</p>
      </section>
    """


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    stage6_results = ROOT / "results" / "stage6"
    stage6_figures = ROOT / "figures" / "stage6"
    reports = ROOT / "reports"

    hourly = pd.read_csv(ROOT / "data" / "hourly_energy_clean.csv", parse_dates=["date"])
    features = pd.read_csv(ROOT / "data" / "hourly_features.csv", parse_dates=["date"])
    raw = pd.read_csv(ROOT / "energydata_complete.csv", parse_dates=["date"])
    gate = pd.read_csv(stage6_results / "gate_decision_trace.csv")
    shap_cases = pd.read_csv(stage6_results / "shap_local_cases.csv")
    audit = (read_rows(stage6_results / "stage6_complete_audit_summary.csv") or [{}])[0]
    model_rows = read_rows(ROOT / "results" / "stage5_final_model_comparison.csv")
    complexity_rows = read_rows(ROOT / "results" / "stage5_complexity_vs_benefit.csv")
    baseline_rows = read_rows(ROOT / "results" / "baseline_24h_fixed_origin.csv")
    stage2_rows = read_rows(ROOT / "results" / "stage2_final_model_comparison.csv")
    stage4_rows = read_rows(ROOT / "results" / "stage4_final_model_comparison.csv")
    oracle_rows = read_rows(ROOT / "results" / "stage5_oracle_three_expert.csv")
    threshold_rows = read_rows(ROOT / "results" / "threshold_gate_results.csv")
    soft_rows = read_rows(ROOT / "results" / "soft_gate_results.csv")
    probability_rows = read_rows(ROOT / "results" / "probability_gate_results.csv")
    learned_rows = read_rows(ROOT / "results" / "learned_gate_results.csv")
    pairwise_rows = read_rows(ROOT / "results" / "stage5_pairwise.csv")
    top_features = read_rows(stage6_results / "shap_global_importance.csv")
    reproduction_rows = read_rows(stage6_results / "reproduction_check.csv")
    horizon_rows = read_rows(stage6_results / "horizon_target_hour_audit.csv")
    unreproduced_rows = read_rows(stage6_results / "stage6_unreproduced_files.csv")
    partition_rows = read_rows(ROOT / "results" / "fuzzy_partition_results.csv")

    best_quantile = next(r for r in partition_rows if r["partition"] == "quantile" and r["requested_intervals"] == "15")
    fts_bounds = json.loads(best_quantile["boundaries"])
    figs = make_generated_figures(hourly, gate, fts_bounds, model_rows)

    copied: list[str] = []
    missing: list[str] = []
    for src in stage6_results.glob("*.csv"):
        dst = OUT / "data" / "stage6" / src.name
        copied.append(dst.relative_to(OUT).as_posix()) if copy_if_exists(src, dst) else missing.append(str(src))
    extra_csv = [
        ROOT / "results" / "baseline_24h_fixed_origin.csv",
        ROOT / "results" / "fuzzy_partition_results.csv",
        ROOT / "results" / "fts_model_comparison.csv",
        ROOT / "results" / "stage2_final_model_comparison.csv",
        ROOT / "results" / "stage4_final_model_comparison.csv",
        ROOT / "results" / "stage5_final_model_comparison.csv",
        ROOT / "results" / "stage5_complexity_vs_benefit.csv",
        ROOT / "results" / "stage5_pairwise.csv",
        ROOT / "results" / "stage5_oracle_three_expert.csv",
        ROOT / "results" / "threshold_gate_results.csv",
        ROOT / "results" / "soft_gate_results.csv",
        ROOT / "results" / "probability_gate_results.csv",
        ROOT / "results" / "learned_gate_results.csv",
    ]
    for src in extra_csv:
        dst = OUT / "data" / src.name
        copied.append(dst.relative_to(OUT).as_posix()) if copy_if_exists(src, dst) else missing.append(str(src))
    for src in reports.glob("stage6*.md"):
        dst = OUT / "reports" / src.name
        copied.append(dst.relative_to(OUT).as_posix()) if copy_if_exists(src, dst) else missing.append(str(src))
    for src in list(stage6_figures.glob("*.png")) + list((stage6_figures / "local_cases").glob("*.png")):
        subdir = "local_cases" if src.parent.name == "local_cases" else "overview"
        dst = OUT / "assets" / subdir / src.name
        copied.append(dst.relative_to(OUT).as_posix()) if copy_if_exists(src, dst) else missing.append(str(src))
    for src in [ROOT / "figures" / "stage4_daily_error_boxplot.png", ROOT / "figures" / "stage5_daily_error_boxplot.png", ROOT / "figures" / "horizon_vs_mae.png", ROOT / "figures" / "horizon_vs_rmse.png", ROOT / "stage6_complete_audit.zip"]:
        dst = OUT / ("packages" if src.suffix == ".zip" else "assets/overview") / src.name
        copied.append(dst.relative_to(OUT).as_posix()) if copy_if_exists(src, dst) else missing.append(str(src))

    raw_hour = raw.groupby(raw["date"].dt.floor("h")).filter(lambda g: len(g) == 6).head(6).copy()
    raw_rows = [{"waktu": row["date"], "Appliances Wh": int(row["Appliances"])} for _, row in raw_hour.iterrows()]
    raw_sum_wh = int(raw_hour["Appliances"].sum())
    raw_sum_kwh = raw_sum_wh / 1000

    first_origin = int(gate["origin_index"].iloc[0])
    first_origin_rows = gate[gate["origin_index"] == first_origin].sort_values("horizon")
    first_daily_pred = float(first_origin_rows["predicted"].sum())
    first_daily_actual = float(first_origin_rows["actual"].sum())
    origin_text = str(first_origin_rows["origin"].iloc[0])

    feature_cols = ["lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_std_24", "hour", "sin_hour", "cos_hour"]
    feature_table = [{"fitur": k, "nilai": fmt(v, 4)} for k, v in features.loc[first_origin, feature_cols].to_dict().items()]
    h1 = first_origin_rows.iloc[0]

    specialist = gate[gate["gate_condition"] == True].iloc[0]
    generalist = gate[gate["gate_condition"] == False].iloc[0]
    gate_examples = [
        {
            "contoh": "Specialist dipilih",
            "origin": specialist["origin"],
            "horizon": int(specialist["horizon"]),
            "xgb_prediction": fmt(specialist["xgb_prediction"]),
            "residual_prediction": fmt(specialist["residual_prediction"]),
            "threshold": fmt(specialist["threshold"]),
            "kondisi": str(bool(specialist["gate_condition"])),
            "prediksi_final": fmt(specialist["predicted"]),
        },
        {
            "contoh": "XGBoost tetap dipakai",
            "origin": generalist["origin"],
            "horizon": int(generalist["horizon"]),
            "xgb_prediction": fmt(generalist["xgb_prediction"]),
            "residual_prediction": fmt(generalist["residual_prediction"]),
            "threshold": fmt(generalist["threshold"]),
            "kondisi": str(bool(generalist["gate_condition"])),
            "prediksi_final": fmt(generalist["predicted"]),
        },
    ]
    interval_rows = [{"state": f"A{i}", "batas bawah": fmt(lo, 3), "batas atas": fmt(hi, 3), "lebar": fmt(hi - lo, 3)} for i, (lo, hi) in enumerate(zip(fts_bounds[:-1], fts_bounds[1:]), 1)]

    local_case = shap_cases[shap_cases["case"] == "low_error"].copy()
    if local_case.empty:
        local_case = shap_cases.copy()
    first_case = local_case.iloc[0]
    positive = local_case[local_case["shap_value"] > 0].sort_values("shap_value", ascending=False).head(5)
    negative = local_case[local_case["shap_value"] < 0].sort_values("shap_value").head(5)
    posneg_rows = []
    for _, row in positive.iterrows():
        posneg_rows.append({"arah": "menaikkan", "fitur": row["feature"], "nilai fitur": fmt(row["feature_value"], 4), "SHAP": fmt(row["shap_value"], 4)})
    for _, row in negative.iterrows():
        posneg_rows.append({"arah": "menurunkan", "fitur": row["feature"], "nilai fitur": fmt(row["feature_value"], 4), "SHAP": fmt(row["shap_value"], 4)})

    xgb_row = next(r for r in model_rows if r.get("model") == "XGBoost Direct")
    gate_row = next(r for r in model_rows if r.get("model") == "Stage 4 Threshold Gate B")
    weekly_row = next(r for r in baseline_rows if r.get("model") == "Weekly Seasonal")
    improvement = float(next(r for r in complexity_rows if r.get("Model") == "Stage 4 Threshold Gate B")["Improvement vs XGB"])
    p_value = float(pairwise_rows[0]["wilcoxon_p"]) if pairwise_rows else np.nan

    nav = [
        ("01", "masalah", "Apa yang diprediksi?"),
        ("02", "dataset", "Dataset"),
        ("03", "fts", "FTS"),
        ("04", "xgboost", "XGBoost"),
        ("05", "hybrid", "Hybrid"),
        ("06", "hasil", "Hasil prediksi"),
        ("07", "shap-global", "SHAP global"),
        ("08", "shap-local", "SHAP lokal"),
        ("09", "kesimpulan", "Kesimpulan"),
    ]
    term_defs = {
        "Quantile": "Pembagian data berdasarkan persentil. Pada penelitian ini dipakai agar area konsumsi rendah yang padat mendapat interval lebih sempit.",
        "Interval": "Rentang nilai konsumsi kWh, misalnya state fuzzy A4 memiliki batas dari data training, bukan angka buatan.",
        "Fuzzy state": "Label kelompok nilai konsumsi, seperti A1 sampai A15, yang mewakili interval tertentu.",
        "Membership degree": "Derajat kedekatan suatu nilai dengan state fuzzy. Pada implementasi ini nilai dipetakan ke state interval untuk membentuk relasi historis.",
        "Weighted FTS": "FTS yang memberi bobot pada konsekuen berdasarkan frekuensi kemunculan relasi, sehingga relasi yang lebih sering muncul lebih berpengaruh.",
        "Lag": "Nilai konsumsi dari jam sebelumnya. Contoh lag_24 berarti konsumsi pada jam yang sama sehari sebelumnya.",
        "Rolling mean": "Rata-rata bergerak dari beberapa jam terakhir, misalnya rolling_mean_24 adalah rata-rata 24 jam terakhir.",
        "Residual": "Sisa kesalahan dari prediksi awal. Residual FTS-XGBoost berarti XGBoost membantu mempelajari kesalahan FTS.",
        "Threshold": "Batas keputusan. Gate B memilih residual FTS-XGBoost ketika prediksi XGBoost tidak lebih dari 0,31 kWh.",
        "MAE": "Mean Absolute Error, rata-rata besar kesalahan prediksi. Semakin kecil, semakin baik.",
        "SHAP": "Metode interpretasi model yang menunjukkan kontribusi setiap fitur terhadap output prediksi, bukan bukti sebab-akibat.",
    }

    slides = [
        slide("masalah", "01", "Bisakah kita memperkirakan penggunaan listrik rumah untuk besok?", f"Model memprediksi konsumsi energi setiap jam untuk 24 jam berikutnya. Seluruh hasilnya dijumlahkan untuk mengetahui perkiraan total energi selama satu hari. Contoh origin nyata: <b>{origin_text}</b>.", f"<figure><img src='{figs['forecast']}' alt='Contoh prediksi 24 jam'><figcaption>Prediksi Gate B H1-H24 = <b>{fmt(first_daily_pred)} kWh</b>; aktual 24 jam = <b>{fmt(first_daily_actual)} kWh</b>.</figcaption></figure>", narrative("Penelitian ini mencoba menjawab apakah data konsumsi listrik sebelumnya cukup untuk memperkirakan pemakaian 24 jam ke depan. Model menghasilkan 24 prediksi per jam, lalu totalnya dijumlahkan sebagai prediksi konsumsi harian. Angka pada grafik berasal dari forecast origin Stage 6, bukan contoh simulasi."), "Target penelitian adalah prediksi H1-H24 dan total energi harian dari satu rumah.", "Untuk melakukan prediksi tersebut, kita perlu memahami terlebih dahulu seperti apa data yang digunakan."),
        slide("dataset", "02", "Seperti apa data konsumsi listrik yang kita gunakan?", f"Dataset berasal dari UCI Appliances Energy Prediction, satu rumah, interval awal 10 menit, target <code>Appliances</code> dalam Wh. Setelah preprocessing, enam pencatatan 10 menit dijumlahkan menjadi hourly kWh dengan <b>{int_id(len(hourly))}</b> observasi lengkap.", f"<div class='split'><div>{data_table(raw_rows, ['waktu', 'Appliances Wh'])}<p class='formula'>Enam pencatatan = {raw_sum_wh} Wh = <b>{fmt(raw_sum_kwh, 3)} kWh</b></p></div><figure><img src='{figs['hourly_distribution']}' alt='Distribusi hourly'><figcaption>Distribusi konsumsi hourly dari data asli.</figcaption></figure></div>", tech("Agregasi target memakai penjumlahan, bukan rata-rata, karena Appliances adalah energi per interval 10 menit. Variabel suhu, kelembapan, dan cuaca memakai rata-rata per jam."), "Data modeling memakai konsumsi per jam dalam kWh, bukan baris 10 menit mentah.", "Setelah data menjadi per jam, kita perlu mencari cara agar model dapat mempelajari pola perubahan konsumsi listrik."),
        slide("fts", "03", "Bagaimana FTS mempelajari pola konsumsi listrik?", "FTS mengubah angka konsumsi menjadi kelompok fuzzy, lalu membaca perubahan kelompok dari waktu ke waktu. Konfigurasi terbaik Stage 2 adalah <b>Weighted FTS order 1, quantile partition, 15 interval</b>.", f"<figure><img src='{figs['fts_distribution']}' alt='Distribusi quantile FTS'><figcaption>Contoh batas: A4 = {fmt(fts_bounds[3], 3)}–{fmt(fts_bounds[4], 3)} kWh, A15 = {fmt(fts_bounds[14], 3)}–{fmt(fts_bounds[15], 3)} kWh.</figcaption></figure>{data_table(interval_rows, ['state','batas bawah','batas atas','lebar'])}", f"<p>{term_button('Quantile')} {term_button('Interval')} {term_button('Fuzzy state')} {term_button('Membership degree')} {term_button('Weighted FTS')}</p>" + narrative("FTS tidak langsung bekerja pada angka mentah saja. Angka konsumsi dikelompokkan ke interval fuzzy, misalnya A1 sampai A15. Quantile dipakai karena sebagian besar konsumsi berada pada nilai kecil, sehingga daerah yang padat dibuat lebih rinci."), "Quantile membuat interval rendah lebih sempit dan interval tinggi lebih lebar sesuai kepadatan data training.", "FTS mempelajari perubahan kelompok konsumsi. Namun, kita juga mencoba algoritma machine learning yang mempelajari langsung angka konsumsi listrik."),
        slide("xgboost", "04", "Bagaimana XGBoost memprediksi penggunaan listrik?", "XGBoost mempelajari hubungan antara fitur historis, fitur waktu target, dan konsumsi yang akan diprediksi. Contoh di bawah memakai forecast origin yang sama dengan halaman pertama.", f"<div class='split'><div>{data_table(feature_table, ['fitur','nilai'])}</div><div class='diagram'>Data historis<br>→ pembentukan fitur<br>→ XGBoost Direct<br>→ prediksi H1-H24</div></div>", f"<p>Untuk H1 pada origin ini, prediksi XGBoost = <b>{fmt(h1['xgb_prediction'])} kWh</b>, aktual = <b>{fmt(h1['actual'])} kWh</b>, jam target = <b>{int((pd.to_datetime(h1['origin']) + pd.Timedelta(hours=int(h1['horizon']))).hour)}</b>.</p><p>{term_button('Lag')} {term_button('Rolling mean')}</p>" + tech("Fitur lag dan rolling berasal dari histori sebelum origin. Fitur hour/sin_hour/cos_hour mewakili jam target, sehingga tidak boleh diperlakukan sebagai data konsumsi historis."), "XGBoost membaca pola numerik dari konsumsi sebelumnya dan informasi waktu target.", "Setelah mencoba kedua metode, pertanyaan berikutnya adalah bagaimana FTS dan XGBoost dapat digunakan bersama."),
        slide("hybrid", "05", "Bagaimana kedua metode digabungkan menjadi hybrid?", f"Threshold-Gated Hybrid memilih expert berdasarkan prediksi XGBoost. Aturannya: gunakan residual FTS-XGBoost ketika prediksi XGBoost ≤ <b>{fmt(audit.get('gate_threshold'))} kWh</b>; selain itu gunakan XGBoost Direct.", f"<div class='flow'><span>XGBoost prediksi awal</span><span>FTS + XGBoost residual</span><span>Threshold Gate B</span><span>Expert terpilih</span></div>{data_table(gate_examples, ['contoh','origin','horizon','xgb_prediction','residual_prediction','threshold','kondisi','prediksi_final'])}", f"<p>{term_button('Residual')} {term_button('Threshold')}</p><p>Sebagian besar prediksi tetap memakai XGBoost Direct. Specialist aktif <b>{audit.get('specialist_activation_count')}</b> dari 480 prediksi jam atau {pct(audit.get('specialist_activation_share'))}.</p>", "Gate B adalah aturan threshold rendah 0,31 kWh, berbeda dari threshold high/spike regime.", "Setelah kedua metode digabungkan, kita perlu melihat apakah hasil prediksinya menjadi lebih baik."),
        slide("hasil", "06", "Apakah model hybrid menghasilkan prediksi yang lebih baik?", f"Pada evaluasi 20 origin non-overlap, hybrid menurunkan kesalahan prediksi total konsumsi harian sekitar <b>{fmt(improvement, 2)}%</b> dibandingkan XGBoost Direct. Namun XGBoost Direct tetap lebih baik pada Hourly MAE.", f"<figure><img src='{figs['mae_bar']}' alt='Daily MAE bar chart'><figcaption>Weekly Seasonal Naive = {fmt(weekly_row['Daily MAE'])} kWh; XGBoost Direct = {fmt(xgb_row['Daily MAE'])} kWh; Threshold Gate B = {fmt(gate_row['Daily MAE'])} kWh.</figcaption></figure>", f"<p>{term_button('MAE')}</p><details class='technical'><summary>Mengapa hasil per jam dan per hari bisa berbeda?</summary><p>Hourly MAE menghitung kesalahan per jam. Daily MAE menjumlahkan 24 prediksi dahulu, lalu menghitung kesalahan total harian. Gate B sedikit lebih baik pada Daily MAE, tetapi XGBoost Direct lebih baik pada Hourly MAE. Uji Wilcoxon Stage 5 memberi p-value {fmt(p_value, 4)}, sehingga peningkatan belum signifikan pada taraf 5%.</p></details>", "Hybrid memberi perbaikan Daily MAE kecil, tetapi belum cukup kuat untuk diklaim unggul di semua metrik.", "Setelah mengetahui model mana yang memberikan hasil lebih baik, kita juga ingin mengetahui informasi apa yang digunakan model untuk membuat prediksi."),
        slide("shap-global", "07", "Informasi apa yang paling digunakan XGBoost?", "SHAP global menunjukkan fitur yang paling banyak berkontribusi terhadap output XGBoost Direct pada audit Stage 6.", "<figure><img src='assets/overview/shap_importance.png' alt='SHAP global importance'><figcaption>Top fitur: " + ", ".join(html.escape(r["feature"]) for r in top_features[:5]) + ".</figcaption></figure>", f"<p>Fitur <code>hour</code>, <code>cos_hour</code>, dan <code>sin_hour</code> memberi tahu model posisi jam target dalam siklus 24 jam. <code>rolling_mean_24</code> dan <code>rolling_std_24</code> memberi ringkasan konsumsi sebelumnya.</p><p>{term_button('SHAP')}</p><p class='note'>SHAP menjelaskan kontribusi fitur pada output model, bukan bukti sebab-akibat. Fitur waktu juga saling berkaitan.</p>", "Model banyak memakai informasi waktu target dan ringkasan konsumsi historis.", "SHAP tidak hanya dapat menjelaskan model secara keseluruhan, tetapi juga dapat menjelaskan alasan di balik satu prediksi tertentu."),
        slide("shap-local", "08", "Mengapa model menghasilkan prediksi sebesar itu?", f"Contoh local explanation dari Stage 6: origin <b>{first_case['origin']}</b>, horizon <b>{int(first_case['horizon'])}</b>, prediksi XGBoost <b>{fmt(first_case['xgb_prediction'])} kWh</b>, aktual <b>{fmt(first_case['actual'])} kWh</b>, base value SHAP <b>{fmt(first_case['base_value'])}</b>.", "<figure><img src='assets/local_cases/low_error_waterfall.png' alt='Waterfall SHAP local'><figcaption>Waterfall plot asli dari Stage 6 untuk kasus low_error.</figcaption></figure>", data_table(posneg_rows, ["arah", "fitur", "nilai fitur", "SHAP"]) + narrative("SHAP local membantu menjelaskan satu prediksi tertentu. Nilai dasar model bergerak naik atau turun mengikuti kontribusi fitur, sampai menjadi prediksi akhir."), "SHAP local membuat alasan satu prediksi lebih mudah dijelaskan tanpa mengubah model.", "Dari seluruh pengujian dan interpretasi tersebut, kita dapat merangkum hasil utama penelitian."),
        slide("kesimpulan", "09", "Apa yang kita pelajari dari penelitian ini?", "Kesimpulan penelitian perlu disampaikan dengan hati-hati: ada perbaikan harian dari hybrid, tetapi ada juga keterbatasan yang harus tetap ditulis.", "<div class='conclusion'><ol><li>XGBoost mampu mempelajari pola konsumsi energi dengan cukup baik.</li><li>Penggabungan FTS dan XGBoost melalui threshold menurunkan Daily MAE sekitar 2,65% pada pengujian yang digunakan.</li><li>SHAP menunjukkan bahwa informasi waktu dan konsumsi historis memiliki kontribusi penting dalam prediksi.</li></ol><h3>Keterbatasan</h3><ul><li>Peningkatan hybrid masih relatif kecil.</li><li>Hybrid tidak unggul untuk Hourly MAE.</li><li>Hasil belum menunjukkan keunggulan statistik yang konklusif.</li><li>Dataset berasal dari satu rumah.</li></ul></div>", "<p class='proposed'>Proposed method: <b>Threshold-Gated Hybrid Weighted FTS–XGBoost with SHAP Explainability</b>.</p>", "Penelitian siap dijelaskan sebagai proses bertahap dari data, model, hybrid, hasil, sampai interpretasi SHAP.", "Jika dosen ingin detail angka, buka Lampiran Teknis dari tombol kanan atas."),
    ]

    appendix = f"""
      <section class="slide appendix" id="lampiran" data-title="Lampiran Teknis">
        <p class="eyebrow">Lampiran</p>
        <h1>Lampiran Teknis</h1>
        <p class="lead">Bagian ini menyimpan detail teknis lama agar alur presentasi utama tetap ringan.</p>
        <h2>Perbandingan semua model Stage 5</h2>
        {data_table(model_rows, ['model','Hourly MAE','Hourly RMSE','Daily MAE','Daily RMSE','Daily MAPE','windows'])}
        <h2>Model Stage 1-2 dan Stage 4</h2>
        {data_table(baseline_rows, ['model','Hourly MAE','Hourly RMSE','Daily MAE','Daily RMSE','Daily MAPE','windows_evaluated'])}
        {data_table(stage2_rows, ['model','Hourly MAE','Hourly RMSE','Daily MAE','Daily RMSE','Daily MAPE','windows_evaluated'])}
        {data_table(stage4_rows, ['model','Hourly MAE','Hourly RMSE','Daily MAE','Daily RMSE','Daily MAPE','windows'])}
        <h2>Oracle, threshold, soft/probability/learned gate</h2>
        {data_table(oracle_rows, list(oracle_rows[0].keys()) if oracle_rows else [])}
        {data_table(threshold_rows, list(threshold_rows[0].keys()) if threshold_rows else [])}
        {data_table(soft_rows, list(soft_rows[0].keys()) if soft_rows else [])}
        {data_table(probability_rows, list(probability_rows[0].keys()) if probability_rows else [])}
        {data_table(learned_rows, list(learned_rows[0].keys()) if learned_rows else [])}
        <h2>Kompleksitas vs manfaat</h2>
        {data_table(complexity_rows, ['Model','Daily MAE','Improvement vs XGB','Number of Models','Gate Needed','Complexity'])}
        <h2>Reproduction check per origin</h2>
        {data_table(reproduction_rows, ['forecast_origin','actual_daily_kWh','xgb_daily_kWh','gate_daily_kWh','xgb_daily_abs_error','gate_daily_abs_error','specialist_count'])}
        <h2>Horizon target-hour audit</h2>
        {data_table(horizon_rows, list(horizon_rows[0].keys()) if horizon_rows else [])}
        <h2>SHAP heatmap dan dekomposisi</h2>
        <figure><img src="assets/overview/shap_horizon_heatmap.png" alt="SHAP heatmap"><figcaption>SHAP per horizon H1-H24.</figcaption></figure>
        <figure><img src="assets/overview/daily_shap_waterfall.png" alt="Daily SHAP waterfall"><figcaption>Daily SHAP decomposition.</figcaption></figure>
        <h2>Audit reproducibility dan additivity</h2>
        <div class="compact-grid">
          <p>XGB max additivity error: <b>{fmt(audit.get('xgb_direct_max_additivity_error'), 8)}</b></p>
          <p>Residual max additivity error: <b>{fmt(audit.get('residual_corrector_max_additivity_error'), 8)}</b></p>
          <p>Daily SHAP max reconstruction error: <b>{fmt(audit.get('daily_xgb_shap_max_reconstruction_error'), 8)}</b></p>
          <p>Gate decomposition max error: <b>{fmt(audit.get('daily_gate_max_reconstruction_error'), 8)}</b></p>
        </div>
        <h2>File tidak dapat direproduksi</h2>
        {('<p class="ok">Audit Stage 6 mencatat tidak ada file yang tidak dapat direproduksi.</p>' if not unreproduced_rows or not any(unreproduced_rows[0].values()) else data_table(unreproduced_rows, list(unreproduced_rows[0].keys())))}
        <p class="buttons"><a href="packages/stage6_complete_audit.zip">Unduh ZIP audit Stage 6</a><a href="reports/stage6_complete_audit.md">Buka laporan audit</a><a href="reports/stage6_shap_final_report.md">Buka laporan SHAP</a></p>
      </section>
    """

    css = """
      :root { --blue:#124a7c; --blue2:#eaf3fb; --green:#2e8b63; --ink:#132236; --muted:#64748b; --line:#d9e2ec; --bg:#f6f8fb; }
      * { box-sizing:border-box; } html, body { margin:0; height:100%; }
      body { font-family:"Segoe UI", Arial, sans-serif; background:var(--bg); color:var(--ink); overflow:auto; }
      .app { display:grid; grid-template-columns:270px 1fr; min-height:100vh; }
      aside { position:sticky; top:0; max-height:100vh; background:#fff; border-right:1px solid var(--line); padding:18px; overflow:auto; }
      .brand { font-weight:800; color:var(--blue); margin:0 0 14px; line-height:1.3; }
      nav button { width:100%; display:grid; grid-template-columns:42px 1fr; gap:8px; align-items:center; border:0; background:transparent; color:var(--ink); text-align:left; padding:10px; border-radius:12px; cursor:pointer; font:inherit; }
      nav button:hover, nav button.active { background:var(--blue2); color:var(--blue); } nav .no { font-weight:800; color:var(--blue); }
      .stage { display:grid; grid-template-rows:auto 1fr; min-width:0; }
      .topbar { display:flex; align-items:center; gap:12px; padding:12px 22px; background:#fff; border-bottom:1px solid var(--line); }
      .progress { flex:1; height:8px; background:#e8edf4; border-radius:99px; overflow:hidden; } .progress span { display:block; height:100%; width:11%; background:var(--green); transition:width .2s ease; }
      .topbar button, .buttons a { border:1px solid var(--line); background:#fff; padding:9px 12px; border-radius:10px; cursor:pointer; color:var(--blue); text-decoration:none; font-weight:650; }
      .topbar button:hover, .buttons a:hover { background:var(--blue2); }
      main { position:relative; overflow:visible; padding:24px 34px; }
      .slide { display:none; max-width:1120px; min-height:calc(100vh - 140px); margin:0 auto; background:#fff; border:1px solid var(--line); border-radius:20px; padding:32px 42px; animation:fade .18s ease; }
      .slide.active { display:block; } @keyframes fade { from { opacity:.3; transform:translateY(4px); } to { opacity:1; transform:none; } }
      .eyebrow { color:var(--green); font-weight:800; letter-spacing:.08em; text-transform:uppercase; margin:0 0 8px; }
      h1 { margin:0 0 12px; font-size:34px; color:var(--blue); line-height:1.15; } h2 { color:var(--blue); margin-top:24px; } h3 { color:#29445f; }
      .lead { font-size:18px; line-height:1.55; color:#29384a; max-width:980px; } .visual { margin:24px 0; } figure { margin:0; }
      img { max-width:100%; border:1px solid var(--line); border-radius:14px; background:#fff; } figcaption { color:var(--muted); margin-top:8px; font-size:14px; }
      .explain, li, p { line-height:1.6; } .takeaway { border-left:5px solid var(--green); background:#eefaf3; padding:12px 16px; border-radius:10px; } .transition { color:var(--muted); font-style:italic; }
      .split { display:grid; grid-template-columns:minmax(280px,.9fr) minmax(300px,1.1fr); gap:22px; align-items:start; }
      .table-scroll { overflow:auto; border:1px solid var(--line); border-radius:12px; background:#fff; max-width:100%; } table { width:100%; border-collapse:collapse; min-width:560px; }
      th, td { padding:9px 11px; border-bottom:1px solid var(--line); text-align:left; font-size:14px; } th { background:var(--blue2); color:var(--blue); }
      details { border:1px solid var(--line); border-radius:12px; padding:12px 14px; margin:12px 0; background:#fbfdff; } summary { cursor:pointer; font-weight:750; color:var(--blue); }
      .term { border:1px solid #b8c9dc; border-radius:99px; background:#fff; color:var(--blue); padding:6px 10px; margin:4px; cursor:pointer; }
      .note { background:#fff8e6; border:1px solid #efd694; padding:12px 14px; border-radius:12px; } .ok { background:#effaf2; border:1px solid #b8e4c2; padding:12px; border-radius:12px; }
      .formula, .proposed { font-size:18px; background:var(--blue2); padding:14px; border-radius:12px; }
      .diagram, .flow { border:1px dashed #9fb7d0; background:#f7fbff; border-radius:16px; padding:24px; font-size:22px; line-height:1.8; color:var(--blue); text-align:center; }
      .flow { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; font-size:16px; } .flow span { background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px; }
      .conclusion { display:grid; grid-template-columns:1fr 1fr; gap:22px; } .compact-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; }
      dialog { max-width:520px; border:0; border-radius:16px; padding:22px; box-shadow:0 20px 70px rgba(0,0,0,.25); } dialog::backdrop { background:rgba(13,28,45,.35); }
      .hidden-sidebar aside { display:none; } .hidden-sidebar.app { grid-template-columns:1fr; }
      @media (max-width:900px) { .app { display:block; height:auto; } aside { position:static; max-height:none; } .stage { display:block; } main { padding:14px; } .slide { padding:24px; min-height:auto; } .split, .conclusion, .flow { grid-template-columns:1fr; } h1 { font-size:27px; } .topbar { position:sticky; top:0; z-index:2; flex-wrap:wrap; } }
      @media (prefers-reduced-motion: reduce) { .slide { animation:none; } .progress span { transition:none; } }
      @media (max-height:760px) and (min-width:901px) { .slide { padding:24px 34px; } h1 { font-size:30px; } .lead { font-size:16px; } img { max-height:360px; object-fit:contain; } }
    """
    js = f"""
      const pages = {json.dumps([sid for _, sid, _ in nav] + ['lampiran'])};
      const terms = {json.dumps(term_defs, ensure_ascii=False)};
      let current = 0;
      const app = document.querySelector('.app'), slides = [...document.querySelectorAll('.slide')], buttons = [...document.querySelectorAll('nav button')], progress = document.querySelector('.progress span');
      function show(index) {{
        current = Math.max(0, Math.min(index, pages.length - 1));
        const id = pages[current];
        slides.forEach(s => s.classList.toggle('active', s.id === id));
        buttons.forEach(b => b.classList.toggle('active', b.dataset.target === id));
        progress.style.width = ((Math.min(current + 1, 9) / 9) * 100) + '%';
        document.querySelector('#pageLabel').textContent = id === 'lampiran' ? 'Lampiran' : String(current + 1).padStart(2,'0') + ' / 09';
        history.replaceState(null, '', '#' + id);
        window.scrollTo(0,0);
      }}
      buttons.forEach(b => b.addEventListener('click', () => show(pages.indexOf(b.dataset.target))));
      document.querySelector('#prev').onclick = () => show(current - 1);
      document.querySelector('#next').onclick = () => show(current + 1);
      document.querySelector('#home').onclick = () => show(0);
      document.querySelector('#appendix').onclick = () => show(pages.indexOf('lampiran'));
      document.querySelector('#toggleSidebar').onclick = () => app.classList.toggle('hidden-sidebar');
      document.querySelector('#fullscreen').onclick = () => document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
      document.addEventListener('keydown', e => {{ if (e.key === 'ArrowRight') show(current + 1); if (e.key === 'ArrowLeft') show(current - 1); if (e.key === 'Home') show(0); }});
      const dialog = document.querySelector('#termDialog');
      document.addEventListener('click', e => {{ if (!e.target.matches('.term')) return; const term = e.target.dataset.term; dialog.querySelector('h2').textContent = term; dialog.querySelector('p').textContent = terms[term] || 'Definisi belum tersedia.'; dialog.showModal(); }});
      dialog.querySelector('button').onclick = () => dialog.close();
      show(Math.max(0, pages.indexOf(location.hash.replace('#',''))));
    """
    html_doc = f"""<!doctype html>
<html lang="id">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Storytelling Presentation Stage 6</title><style>{css}</style></head>
<body>
  <div class="app">
    <aside><p class="brand">Prediksi Konsumsi Energi<br>Interactive Research Story</p><nav>{''.join(f"<button data-target='{sid}'><span class='no'>{no}</span><span>{html.escape(label)}</span></button>" for no, sid, label in nav)}</nav></aside>
    <div class="stage">
      <div class="topbar"><button id="home">Awal</button><button id="prev">Sebelumnya</button><button id="next">Berikutnya</button><div class="progress" aria-label="Progress"><span></span></div><strong id="pageLabel">01 / 09</strong><button id="appendix">Lampiran Teknis</button><button id="toggleSidebar">Sidebar</button><button id="fullscreen">Full screen</button></div>
      <main>{''.join(slides)}{appendix}</main>
    </div>
  </div>
  <dialog id="termDialog"><h2></h2><p></p><button>Tutup</button></dialog>
  <script>{js}</script>
</body></html>
"""
    (OUT / "index.html").write_text(html_doc, encoding="utf-8")
    readme = "# Interactive Report Stage 6\n\nBuka `index.html` di browser. Gunakan tombol **Berikutnya** atau keyboard panah kanan/kiri untuk presentasi.\n\nFolder ini bersifat statis; tidak perlu menjalankan server dan tidak mengubah hasil eksperimen.\n"
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    (ROOT / "README_interactive_report_stage6.md").write_text(readme, encoding="utf-8")

    audit_md = f"""# Presentation Redesign Audit

## 1. Permasalahan desain versi lama

Versi lama lebih mirip dashboard teknis: banyak tabel dan grafik tampil sekaligus, sehingga alur penjelasan untuk dosen kurang bertahap.

## 2. Perubahan layout

Dashboard diubah menjadi presentasi storytelling 9 halaman dengan pola judul, pengantar, satu visual utama, penjelasan, pesan utama, dan transisi. Sidebar memakai nomor 01-09, tombol sebelumnya/berikutnya, progress indicator, keyboard navigation, fullscreen, dan toggle sidebar.

## 3. Perubahan storytelling

Urutan baru: masalah prediksi, dataset, FTS, XGBoost, hybrid, hasil, SHAP global, SHAP local, kesimpulan. Detail teknis dipindahkan ke Lampiran Teknis.

## 4. Sumber data per halaman

- Halaman 1: `results/stage6/gate_decision_trace.csv`, `data/hourly_energy_clean.csv`.
- Halaman 2: `energydata_complete.csv`, `data/hourly_energy_clean.csv`.
- Halaman 3: `results/fuzzy_partition_results.csv`, `data/hourly_energy_clean.csv`.
- Halaman 4: `data/hourly_features.csv`, `results/stage6/gate_decision_trace.csv`.
- Halaman 5: `results/stage6/gate_decision_trace.csv`, `results/stage6/stage6_complete_audit_summary.csv`.
- Halaman 6: `results/baseline_24h_fixed_origin.csv`, `results/stage5_final_model_comparison.csv`, `results/stage5_complexity_vs_benefit.csv`, `results/stage5_pairwise.csv`.
- Halaman 7: `results/stage6/shap_global_importance.csv`, `figures/stage6/shap_importance.png`.
- Halaman 8: `results/stage6/shap_local_cases.csv`, `figures/stage6/local_cases/low_error_waterfall.png`.
- Halaman 9: ringkasan dari Stage 5 dan Stage 6 audit.

## 5. Grafik yang digunakan

Grafik asli: SHAP importance, SHAP heatmap, daily SHAP waterfall, local SHAP waterfall, Stage 4/5 error boxplot. Grafik tambahan dibuat dari data asli: contoh forecast origin, distribusi hourly, distribusi quantile FTS, dan bar chart Daily MAE.

## 6. Pengecekan responsivitas

CSS memakai layout responsif, tabel scroll horizontal, sidebar scroll, aturan khusus untuk tinggi layar laptop sekitar 1366x768, dan media query mobile. Navigasi keyboard dan tombol fullscreen tersedia.

## 7. Masalah yang belum terselesaikan

Browser console dan screenshot lintas resolusi belum diautomasi di environment ini. Validasi yang dilakukan adalah build statis, keberadaan asset, dan konsistensi angka terhadap CSV sumber.
"""
    (reports / "presentation_redesign_audit.md").write_text(audit_md, encoding="utf-8")
    copy_if_exists(reports / "presentation_redesign_audit.md", OUT / "reports" / "presentation_redesign_audit.md")

    required = [OUT / "index.html", OUT / figs["forecast"], OUT / figs["fts_distribution"], OUT / "assets/overview/shap_importance.png", OUT / "assets/local_cases/low_error_waterfall.png", OUT / "packages/stage6_complete_audit.zip", OUT / "reports/presentation_redesign_audit.md"]
    missing_required = [str(p) for p in required if not p.exists()]
    if missing_required:
        raise FileNotFoundError(missing_required)
    if "dummy" in html_doc.lower() or "placeholder" in html_doc.lower():
        raise ValueError("Placeholder text detected in generated HTML")
    if int_id(len(hourly)) not in html_doc:
        raise ValueError("Hourly observation count did not render")

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUT.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(OUT.parent))

    print(f"Dashboard: {OUT / 'index.html'}")
    print(f"ZIP: {ZIP_PATH}")
    print("Slides: 9 + appendix")
    print(f"Copied files: {len(copied)}")
    print(f"Missing optional files: {len(missing)}")


if __name__ == "__main__":
    build()
