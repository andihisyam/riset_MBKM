# Audit Protokol Paper

Paper menggunakan data 10-menit, 75% training dan 25% testing melalui CARET `createDataPartition`, repeated 10-fold CV dengan 3 repeat, serta predictor lingkungan, lights, NSM, status hari, dan hari minggu. Seed/split persis dari paper tidak tersedia dalam PDF yang digunakan, sehingga Track A di sini adalah approximate reproduction.

FTS membutuhkan urutan timestamp. Karena itu Track A menggunakan pembagian kronologis 75/25 agar transisi FTS tidak menyambungkan baris acak. Angka paper hanya dipakai sebagai reference table, bukan sebagai hasil eksperimen ulang. Track B menggunakan split kronologis train/validation/test dan informasi lingkungan pada forecast origin.
