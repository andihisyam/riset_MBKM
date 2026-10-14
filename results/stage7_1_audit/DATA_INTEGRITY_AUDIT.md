# DATA INTEGRITY AUDIT — Stage 7.1 Phase A

## Dataset

- Raw rows: 19,735
- Hourly complete rows: 3,289
- Raw duplicate timestamps: 0
- Hourly duplicate timestamps: 0
- Incomplete hourly bins in raw resample: 1
- Max absolute difference between hourly `Appliances_kWh` and raw hourly Wh sum / 1000: 0.000000000000
- Hourly chronological order valid: True

## Split kronologis

| split      |   start_index |   end_index |    n | start               | end                 |
|:-----------|--------------:|------------:|-----:|:--------------------|:--------------------|
| train      |             0 |        2301 | 2302 | 2016-01-11 17:00:00 | 2016-04-16 14:00:00 |
| validation |          2302 |        2794 |  493 | 2016-04-16 15:00:00 | 2016-05-07 03:00:00 |
| test       |          2795 |        3288 |  494 | 2016-05-07 04:00:00 | 2016-05-27 17:00:00 |

## Forecast origin dan horizon

- Stage 6 rows: 480
- Stage 7 XGBoost test rows: 480
- Mismatched Stage 6 vs Stage 7 origin/horizon keys: 0
- Max actual target difference Stage 6 vs Stage 7: 0.0
- Seluruh model Stage 7 test memiliki 20 origin × 24 horizon: True

## Acceptance criteria

- Prediksi dapat dipasangkan dengan target timestamp melalui `origin_index + horizon`.
- Tidak ditemukan duplicate timestamp pada hourly dataset.
- Stage 6 dan Stage 7 memakai target aktual yang sama pada test origin/horizon.
- Perbedaan preprocessing utama yang teridentifikasi sampai Phase A: Stage 7 menambahkan algoritma dan protocol tuning baru, tetapi target/split/origin test konsisten.
