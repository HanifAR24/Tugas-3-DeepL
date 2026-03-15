# Laporan Eksperimen Deep Neural Network (DNN) pada Dataset MNIST

**Mata Kuliah:** Pengantar Deep Learning

## 1. Pendahuluan
Eksperimen ini bertujuan untuk membangun model Deep Neural Network untuk mengenali dataset angka tulisan tangan (MNIST). Arsitektur dasar model terdiri atas 2 *hidden layer* dan 1 *output layer* dengan 10 neuron (fungsi aktivasi Softmax). Tujuan utama tugas ini adalah menentukan arsitektur terbaik dan melakukan berbagai eksperimen konfigrasional dengan pengoptimal algoritma pelatihan (*training optimizers*), mekanisme *Gradient Descent*, dan rentang kecepatan pembelajaran (*Learning Rates*).

## 2. Eksperimen Tahap 1: Konfigurasi Neuron Hidden Layer
Pada eksperimen pertama, kami menguji tiga kombinasi kepadatan jumlah neuron pada 2 *hidden layer*:
*   **Konfigurasi A (Ringan)**: Layer-1 (64) dan Layer-2 (32)
*   **Konfigurasi B (Menengah)**: Layer-1 (128) dan Layer-2 (64)
*   **Konfigurasi C (Besar)**: Layer-1 (256) dan Layer-2 (128)

**Hasil (Tercatat dalam Notebook):** Konfigurasi yang paling optimal untuk dataset MNIST dalam waktu sangat singkat adalah konfigurasi dengan bobot menampung parameter yang lebih luas (B atau C), karena citra 28x28 (784 fitur input) membutuhkan kapasitas simpan fitur pola yang lumayan memadai pada lapisan pertama.

## 3. Eksperimen Tahap 2: Optimizer, Gradient Descent, & Learning Rate
Setelah base model ditentukan, eksperimen dilanjutkan dengan kombinasi simulasi perbandingan:
1.  **Optimizer**: Adam vs RMSProp
2.  **Gradient Descent**: 
    - *Stochastic Gradient Descent (batch_size=1)*: Memperbarui parameter per 1 titik data uji.
    - *Batch Gradient Descent (batch_size=Total Dataset)*: Memperbarui parameter 1 kali tiap akhir proses pembacaan 1 keseluruhan dataset.
    - *Mini-batch Gradient Descent (batch_size=128)*: Mereduksi fluktuasi namun jauh menghemat memori.
3.  **Learning Rate**: Menguji laju kecepatan pelatihan 0.1, 0.01, dan 0.001.

### Justifikasi Penggunaan Konfigurasi Paling Optimal
1.  **Mini-Batch adalah Metode Paling Optimal**: Dibandingkan dengan *Stochastic* yang komputasional terlalu rewel/ribet dan *Full Batch* yang makan memori, *Mini-Batch* menyeimbangkan konvergensi error yang sangat cepat, optimalisasi memori grafis (opsional), serta performa akhir akurasi validasi terbaik pada set pengujian dalam kurun waktu eksekusi tersebar.
2.  **Learning Rate 0.001**: Secara eksperimental, algoritma perbaruan bobot momentum moderen (seperti Adam) paling ramah dan akurat dengan skala langkah `0.001`. Menggunakan `0.1` terlalu masif sehingga loss rentan meledak dan justru menggagalkan proses pembelajaran (Akurasi mentok di angka tebakan acak 10-18%).
3.  **Adam vs RMSProp**: Adam menggabungkan kelebihan Root Mean Square Propagation (RMSProp) dengan penjagaan momentum, memberinya performa lebih licin dalam meminimalkan kerugian saat menjumpai "lembah" pola (loss).

Secara komprehensif, performa kode visual, perbandingan loss-accuracy, kecepatan pelatihan persatu kombinasi, serta kurva bar-plot tersedia sepenuhnya dalam lembar **Tugas_Pengantar_Deep_Learning_MNIST.ipynb** yang dapat dibuka di Jupyter Notebook atau Google Colab.
