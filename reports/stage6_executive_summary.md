# Executive Summary Stage 6

Stage 6 menjelaskan XGBoost Direct, residual hybrid, dan Threshold Gate B dengan SHAP dan decision trace.

SHAP base value adalah nilai acuan prediksi model. SHAP value adalah kontribusi fitur terhadap selisih prediksi dari nilai acuan. Gate decision trace adalah catatan aturan threshold, input gate, dan expert yang dipilih.

Temuan utama: fitur paling dominan adalah `hour`. Gate B memilih specialist hanya pada sebagian kecil prediksi. Interpretasi mendukung kesimpulan Stage 5: XGBoost Direct tetap kandidat metode utama, sementara FTS lebih tepat sebagai analisis pembanding/eksploratif.
