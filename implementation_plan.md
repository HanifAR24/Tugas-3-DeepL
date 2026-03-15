# Rencana Pengerjaan Tugas Pengantar Deep Learning (MNIST DNN)

Tugas ini bertujuan untuk membangun sebuah model **Deep Neural Network (DNN)** menggunakan dataset **MNIST**. Model akan memiliki **2 hidden layer** dan **1 output layer** (10 neuron untuk klasifikasi digit 0-9). 

Untuk mendapatkan konfigurasi model terbaik, akan dilakukan eksperimen bertahap terkait arsitektur, parameter optimasi, dan besaran _learning rate_.

## User Review Required
> [!IMPORTANT]
> Mohon direview tahapan pengerjaan di bawah ini. Jika ada batasan tambahan dari dosen (misalnya platform yang digunakan harus Google Colab, library harus PyTorch/TensorFlow, atau nilai learning rate secara spesifik), mohon diinformasikan. Jika desain eksperimen ini sudah sesuai, saya akan langsung memulai ke tahap _coding_ dan dokumentasi eksperimen.

---

## 1. Persiapan Library & Preprocessing Data
*   **Dataset:** `keras.datasets.mnist`
*   **Preprocessing:** Image matriks 28x28 akan di-_flatten_ menjadi array 1D berukuran 784 piksel. Nilai piksel (0-255) akan dinormalisasi dengan membaginya dengan 255.0 agar berada dalam rentang `[0, 1]`. Target _labels_ akan disiapkan dalam bentuk label numerik atau di-_one-hot_ encoding.

## 2. Eksperimen Arsitektur (Menentukan Jumlah Neuron)
Kita akan membangun 2 hidden layer. Jumlah neuron akan ditentukan dengan mencoba setidaknya 3 konfigurasi untuk melihat mana yang performa akurasinya paling stabil dan baik dalam epoch singkat, contoh:
*   **Konfigurasi A (Ringan):** Hidden 1 (64 neuron) -> Hidden 2 (32 neuron)
*   **Konfigurasi B (Medium):** Hidden 1 (128 neuron) -> Hidden 2 (64 neuron)
*   **Konfigurasi C (Berat):** Hidden 1 (256 neuron) -> Hidden 2 (128 neuron)

Arsitektur terbaik dari salah satu konfigurasi di atas akan menjadi _base model_ (model dasar) untuk eksperimen selanjutnya.

## 3. Eksperimen Optimizer & Mekanisme Gradient Descent
Sesuai soal, kita harus membandingkan performa antara **RMSprop** dan **Adam** menggunakan mekanisme data _loading_ yang berbeda:
1.  **Stochastic Gradient Descent (SGD):** Bobot (_weights_) diperbarui setelah evaluasi setiap 1 data. (_Batch size = 1_). Ini biasanya sangat lambat berjalan, tapi konvergensinya sangat fluktuatif (berisi banyak noise).
2.  **Batch Gradient Descent:** Bobot diperbarui per *satu epoch penuh* menggunakan **seluruh** dataset latih sekaligus (_Batch size = Jumlah Total Data Train/60000_). Biasanya berat di RAM memory, tapi gradien sangat mulus.
3.  **Mini-batch Gradient Descent:** Mengambil jalan tengah antara di atas (_Batch size = contoh 128 atau 64_). Umumnya merupakan pilihan _default_ standar dalam _Deep Learning_.

Untuk setiap eksperimen, kita akan mencatat Metrik performa (Loss & Accuracy).

## 4. Eksperimen Learning Rate
*Learning Rate* (LR) sangat penting di dalam optimasi model. Eksperimen akan melibatkan eksplorasi 3 titik LR (contoh: **0.1, 0.01, dan 0.001**) pada kombinasi-kombinasi optimizer dan descent di atas untuk mendapatkan posisi yang benar-benar stabil & optimal _(Optimal Learning Rate)_.

## 5. Penyusunan Laporan Akhir & Justifikasi
Setelah kode Python (bisa dibuat jalan di Jupyter Notebook/Colab atau script `.py`) selesai dijalankan:
*   Saya akan membuat output **laporan terstruktur** (contoh: Markdown / PDF info) berisi tabel perbandingan akurasi, _loss_, waktu eksekusi.
*   Menyajikan **grafik evaluasi model**.
*   Menjabarkan **justifikasi analitis** tentang mana optimizer, strategi gradient descent, dan *learning rate* terbaik berdasarkan argumen performa empiris tersebut.
