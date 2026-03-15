# 🧠 Tugas 3: Pengantar Deep Learning MNIST

Repositori ini memuat pengerjaan untuk tugas **Pengantar Deep Learning** dari Universitas. Pada tugas ini, sebuah **Deep Neural Network (DNN)** dibangun dari awal menggunakan pustaka Keras / TensorFlow untuk mengenali angka tulisan tangan dari dataset [MNIST](https://keras.io/api/datasets/mnist/).

## 🎯 Objektif Tugas
- Membangun model DNN dengan tepat *dua hidden layer* dan *satu output layer* (10 neuron).
- **Mengeksplorasi Konfigurasi Neuron:** Mengabungkan jumlah neuron pada *hidden layer* yang optimal.
- **Mengeksplorasi Parameter Optimisasi:**
  1. Perbandingan performa optimizer **RMSprop** versus **Adam**.
  2. Perbandingan mekanisme metode gradient descent:
     - *Stochastic Gradient Descent (SGD)*
     - *Batch Gradient Descent*
     - *Mini-batch Gradient Descent*
- **Mengeksplorasi Skala Learning Rates:** Menguji sensitivitas model terhadap pelipatan *Learning Rate* (0.1, 0.01, 0.001) untuk mencari konvergensi yang tertib dan kokoh.

## 📂 Struktur Repositori

| Berkas / Berkas Kode | Deskripsi Fungsional |
|----------------------|----------------------|
| [Tugas_Pengantar_Deep_Learning_MNIST.ipynb](Tugas_Pengantar_Deep_Learning_MNIST.ipynb) | Jupyter Notebook primer yang berisi semua tahapan proses pembangunan model, manipulasi matriks gambar, serta eksperimen panjang beserta grafiknya (Bar Plot). |
| [Laporan_Akhir.md](Laporan_Akhir.md)                 | Rangkuman Justifikasi teoritis mengenai metrik komputasional paling optimal (misal: *Mengapa Mini-batch sangat disarankan? Mengapa Adam unggul?*) |
| [build_notebook.py](build_notebook.py)               | Skrip bahasa Python internal penunjang eksperimen (*Code generator* untuk kerangka dasar notebook ini). |

---

## 🚀 Cara Menjalankan

Bila Anda ingin menjalankan notebook-nya dan melatih ulang (retrain) model sendiri di mesin lokal Anda:

1. Unduh (Clone) repositori ini ke lokal.
2. Pastikan Anda punya lingkungan virtual Python yang terinstal **TensorFlow 2.x**, **Jupyter Notebook**, **Matplotlib**, dan **Pandas**. 
   - (Sebagai peringatan, hindari NumPy versi 2.0.0 ke atas saat ini jika instalasi library Anda cukup usang. Rekomendasi: `numpy==1.26.4`).
3. Ketikkan perintah `jupyter notebook` dari terminal pada direktori folder repo ini.
4. Buka file bertipe `.ipynb` melalui panel web UI Jupyter.
5. Klik **"Run > Run All Cells"** (Proses di *CPU core* memakan waktu sekira 3-5 menit karena melatih eksperimen pada pelbagai sub-loop parameter secara komutan).

> **Catatan Akademik:** Repositori ini diajukan untuk diserahkan ke tugas **Tugas-Pekan3 - Pengantar Deep Learning** oleh Hanif. Menggunakan model dataset MNIST 60.000 titik sampel.
