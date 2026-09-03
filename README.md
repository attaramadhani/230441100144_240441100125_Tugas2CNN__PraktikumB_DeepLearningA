# Praktikum Deep Learning — Tugas 2: Studi Komparasi Arsitektur CNN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OaJkgPPQCIVmUBc3EbRTsZvLfVozHhYN?usp=sharing)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-blue)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange?logo=googlecolab&logoColor=white)
![Google Drive](https://img.shields.io/badge/Google%20Drive-Hasil%20Eksperimen-34A853?logo=googledrive&logoColor=white)

---

## 🔗 Tautan Akses Cepat (Google Colab & Google Drive)

* 🚀 **Google Colab (Notebook yang Telah Dijalankan)**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OaJkgPPQCIVmUBc3EbRTsZvLfVozHhYN?usp=sharing)  
  👉 **[https://colab.research.google.com/drive/1OaJkgPPQCIVmUBc3EbRTsZvLfVozHhYN?usp=sharing](https://colab.research.google.com/drive/1OaJkgPPQCIVmUBc3EbRTsZvLfVozHhYN?usp=sharing)**

* 📁 **Google Drive (Folder Penyimpanan Berkas & Hasil Eksperimen)**:  
  Folder Google Drive ini memuat seluruh visualisasi grafik 300 DPI, berkas `cache.pkl`, dan laporan dokumen Word:  
  👉 **[https://drive.google.com/drive/folders/1U1iInPW8c6R3An9KEEf_HHRcU3mYWx2c](https://drive.google.com/drive/folders/1U1iInPW8c6R3An9KEEf_HHRcU3mYWx2c)**

---

## 👥 Identitas Anggota Kelompok

| No. | Nama Lengkap | NIM | Akun GitHub | Program Studi | Kelas |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1. | **Attala Alif Ramadhani Tri Hida** | `230441100144` | [@attaramadhani](https://github.com/attaramadhani) | Sistem Informasi | Deep Learning (A) |
| 2. | **Nafaul Hernanda Romadlona** | `240441100125` | [@nafaul](https://github.com/nafaul) | Sistem Informasi | Deep Learning (A) |

---

## 📖 Deskripsi Proyek & Penjelasan Mendalam Arsitektur Model

Proyek ini bertujuan untuk melakukan **studi komparasi empiris dan teoretis** antara dua paradigma arsitektur Convolutional Neural Network (CNN) pada dataset citra **CIFAR-10** (10 kategori objek warna berukuran $32 \times 32$ piksel) di bawah kondisi hyperparameter dan lingkungan komputasi yang identik (*controlled experiment*).

Kedua model yang dikomparasikan dirancang dengan rincian arsitektur sebagai berikut:

### 1. Model A: Custom CNN Konvensional (Plain Baseline CNN)
* **Konsep & Filosofi Desain**:
  Model A mengadopsi arsitektur konvolusi sekuensial murni (*feed-forward plain network*) tanpa koneksi jalan pintas (*skip connection*). Model ini merepresentasikan arsitektur konvensional tipe VGG-style yang mengandalkan tumpukan lapisan konvolusi secara linear untuk mempelajari abstraksi fitur dari representasi sederhana ke kompleks.
* **Rincian Struktur Lapisan**:
  - **Blok 1 (32 Filter)**:
    - Terdiri dari $2 \times$ [`Conv2D(32, kernel_size=(3, 3), padding='same')` $\rightarrow$ `BatchNormalization` $\rightarrow$ `ReLU`].
    - Fungsi: Mengekstraksi fitur-fitur spasial tingkat rendah (*low-level visual features*) seperti tepi garis (*edges*), sudut (*corners*), dan tekstur warna dasar.
    - Diakhiri dengan `MaxPooling2D(pool_size=(2, 2))` (mereduksi resolusi dari $32 \times 32$ menjadi $16 \times 16$) dan regularisasi `Dropout(0.25)`.
  - **Blok 2 (64 Filter)**:
    - Terdiri dari $2 \times$ [`Conv2D(64, kernel_size=(3, 3), padding='same')` $\rightarrow$ `BatchNormalization` $\rightarrow$ `ReLU`].
    - Fungsi: Mengombinasikan fitur spasial dasar menjadi representasi tingkat menengah (*mid-level features*) seperti motif, kontur bentuk, dan batas objek.
    - Diakhiri dengan `MaxPooling2D(pool_size=(2, 2))` (mereduksi resolusi menjadi $8 \times 8$) dan `Dropout(0.25)`.
  - **Blok 3 (128 Filter)**:
    - Terdiri dari $2 \times$ [`Conv2D(128, kernel_size=(3, 3), padding='same')` $\rightarrow$ `BatchNormalization` $\rightarrow$ `ReLU`].
    - Fungsi: Mempelajari fitur semantik tingkat tinggi (*high-level semantic features*) seperti bagian khas objek (sayap pesawat/burung, roda kendaraan, kepala hewan).
    - Diakhiri dengan `MaxPooling2D(pool_size=(2, 2))` (mereduksi resolusi menjadi $4 \times 4$) dan `Dropout(0.25)`.
  - **Classifier Head**:
    - `GlobalAveragePooling2D`: Mengompresi peta fitur spasial $4 \times 4 \times 128$ menjadi representasi vektor 1D ($128$ nilai) dengan merata-ratakan nilai tiap feature map. Pendekatan ini jauh lebih hemat parameter dan tahan terhadap translasi dibanding `Flatten`.
    - `Dense(128, activation='relu')` $\rightarrow$ `Dropout(0.4)`: Lapisan *fully connected* penarik kesimpulan fitur dengan regularisasi ketat untuk memitigasi overfitting.
    - `Dense(10, activation='softmax')`: Lapisan output probabilitas multi-kelas untuk 10 label target CIFAR-10.
* **Karakteristik Parameter & Analisis Teoretis**:
  - **Total Parameter**: **306.602** (Trainable: 305.706, Non-trainable: 896).
  - **Kelemahan Teoretis**: Karena alur informasi hanya berjalan linear satu arah, sinyal gradien pada saat *backpropagation* rentan teredam (*vanishing gradient problem*) ketika melintasi banyak lapisan berurutan, sehingga lanskap loss (*loss landscape*) cenderung lebih terjal dan rentan berosilasi di sekitar titik minimum lokal.

---

### 2. Model B: Custom Mini-ResNet (Deep Residual Learning)
* **Konsep & Filosofi Desain**:
  Model B dirancang mengimplementasikan terobosan *Deep Residual Learning* (He et al., 2016). Alih-alih memaksa tumpukan lapisan konvolusi untuk memetakan fungsi target utuh $\mathcal{H}(x)$, blok residual dirancang untuk mempelajari fungsi deviasi/residual:
  $$\mathcal{F}(x) = \mathcal{H}(x) - x$$
  Keluaran blok kemudian diperoleh dengan menambahkan kembali masukan aslinya melalui koneksi jalan pintas (*residual shortcut/skip connection*):
  $$\mathcal{H}(x) = \mathcal{F}(x) + x$$
  Mekanisme penjumlahan elemen demi elemen ini menciptakan *gradient superhighway*, di mana sinyal gradien dapat mengalir langsung kembali ke lapisan-lapisan awal tanpa terdistorsi atau teredam oleh bobot konvolusi perantara, memecahkan masalah degradasi optimasi.
* **Rincian Struktur Lapisan**:
  - **Initial Stem Layer**:
    - `Conv2D(32, kernel_size=(3, 3), padding='same')` $\rightarrow$ `BatchNormalization` $\rightarrow$ `ReLU`. Berfungsi sebagai gerbang ekstraksi fitur masukan awal sebelum memasuki blok-blok residual.
  - **Stage 1 (Residual Block 32 Filter — Identity Shortcut)**:
    - Jalur Residual $\mathcal{F}(x)$: $2 \times$ [`Conv2D(32, 3x3)` $\rightarrow$ `BN` $\rightarrow$ `ReLU`].
    - Jalur Shortcut: Karena jumlah kanal masukan ($32$) sama persis dengan kanal target ($32$), jalur jalan pintas merupakan *Identity Shortcut* murni ($x$) tanpa memerlukan penambahan parameter.
    - Operasi Penjumlahan: `layers.add([F(x), x])` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling2D(2x2)` $\rightarrow$ `Dropout(0.25)`.
  - **Stage 2 (Residual Block 64 Filter — 1x1 Projection Shortcut)**:
    - Jalur Residual $\mathcal{F}(x)$: $2 \times$ [`Conv2D(64, 3x3)` $\rightarrow$ `BN`].
    - Jalur Shortcut: Karena terjadi transisi penambahan kanal dari $32$ ke $64$, dimensi tensor tidak dapat dijumlahkan secara langsung. Oleh karena itu, dipasang **1x1 Projection Shortcut** menggunakan `Conv2D(64, kernel_size=(1, 1))` $\rightarrow$ `BatchNormalization` untuk memproyeksikan kanal masukan ke dimensi $64$.
    - Operasi Penjumlahan: `layers.add([F(x), W_s x])` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling2D(2x2)` $\rightarrow$ `Dropout(0.25)`.
  - **Stage 3 (Residual Block 128 Filter — 1x1 Projection Shortcut)**:
    - Jalur Residual $\mathcal{F}(x)$: $2 \times$ [`Conv2D(128, 3x3)` $\rightarrow$ `BN`].
    - Jalur Shortcut: Memproyeksikan kanal dari $64$ ke $128$ menggunakan **1x1 Projection Shortcut** (`Conv2D(128, 1x1)` + `BN`).
    - Operasi Penjumlahan: `layers.add([F(x), W_s x])` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling2D(2x2)` $\rightarrow$ `Dropout(0.25)`.
  - **Classifier Head**:
    - Dikonstruksi **100% identik dengan Model A** (`GlobalAveragePooling2D` $\rightarrow$ `Dense(128, ReLU)` $\rightarrow$ `Dropout(0.4)` $\rightarrow$ `Dense(10, Softmax)`) agar disparitas performa murni mencerminkan efektivitas arsitektur backbone residual, bukan karena perbedaan dense classifier.
* **Karakteristik Parameter & Analisis Teoretis**:
  - **Total Parameter**: **327.178** (Trainable: 325.834, Non-trainable: 1.344).
  - **Selisih Parameter**: Model B hanya memiliki kelebihan $20.576$ parameter ($+6,72\%$) yang seluruhnya berasal dari layer proyeksi konvolusi $1 \times 1$ pada Stage 2 dan Stage 3.
  - **Keunggulan Teoretis**: Aliran gradien yang tanpa redaman membuat Model B menghasilkan permukaan fungsi loss yang lebih halus (*loss landscape smoothing*), terbukti menghasilkan **Test Loss yang lebih rendah (0,5971 vs 0,6172)** dan ketahanan generalisasi yang lebih tinggi terhadap data uji unseen.

---

### ⚙️ Konfigurasi Hyperparameter Pelatihan Terkontrol (Identik)
* **Dataset Split**: 40.000 citra latih (Train Set), 10.000 citra validasi (Validation Set), dan 10.000 citra uji (Test Set unseen).
* **Normalisasi Piksel**: Skala interval $[0.0, 1.0]$ (`float32`) dan target One-Hot Encoding 10 dimensi.
* **Jumlah Epoch**: `20`
* **Ukuran Batch (Batch Size)**: `64`
* **Optimizer**: `Adam` (Learning Rate = `0.001`)
* **Fungsi Loss**: `Categorical Crossentropy`
* **Akselerator Komputasi**: Google Colab GPU (Tesla T4)

---

## 📊 Hasil Eksperimen Aktual (Experimental Results)

Berikut adalah ringkasan hasil kuantitatif pengujian kedua model berdasarkan data aktual yang diperoleh dari eksekusi pelatihan (`cache.pkl` dan `Laporan_Hasil_Eksperimen.docx`):

### 1. Tabel Komparasi Performa Utama

| Metrik Evaluasi | Model A (Custom CNN) | Model B (Mini-ResNet) | Keterangan / Analisis |
| :--- | :---: | :---: | :--- |
| **Total Parameter** | 306.602 | 327.178 | Model B memiliki $+6,72\%$ parameter lebih banyak karena adanya shortcut proyeksi $1 \times 1$ |
| **Trainable Parameter** | 305.706 | 325.834 | Bobot aktif yang dioptimasi selama backpropagation |
| **Non-Trainable Parameter** | 896 | 1.344 | Parameter moving mean & variance pada layer Batch Normalization |
| **Durasi Pelatihan (GPU T4)** | **147,60 s** (2,46 m) | **182,06 s** (3,03 m) | Model B membutuhkan waktu sedikit lebih lama karena komputasi penjumlahan tensor residual |
| **Final Training Loss** | 0,4006 | 0,4250 | Nilai loss data latih pada epoch ke-20 |
| **Final Training Accuracy** | 86,18% | 85,51% | Akurasi pada 40.000 citra latih |
| **Final Validation Loss** | 0,6143 | **0,5853** | **Validation Loss Model B lebih rendah (kemampuan generalisasi lebih unggul)** |
| **Final Validation Accuracy** | **81,40%** | 80,98% | Akurasi pada 10.000 citra validasi |
| **Test Loss (Unseen Test Set)** | 0,6172 | **0,5971** | **Model B menghasilkan Test Loss yang lebih optimal pada data uji unseen** |
| **Test Accuracy (Unseen Test Set)**| **81,18%** | 80,65% | Akurasi evaluasi akhir pada 10.000 citra uji unseen |

---

### 2. Analisis Kesalahan Prediksi Dua Arah (*Two-Way Error Analysis*)

Evaluasi komparatif mendalam dilakukan secara simetris terhadap 10.000 sampel data uji unseen:
* **Total Kesalahan Model A (CNN)**: `1.882` citra salah (Akurasi: $81,18\%$)
* **Total Kesalahan Model B (Mini-ResNet)**: `1.935` citra salah (Akurasi: $80,65\%$)
* **Kasus 1 (Keunggulan ResNet)**: Terdapat **637 citra** yang **salah diprediksi oleh Model A (CNN), tetapi berhasil dijawab dengan BENAR oleh Model B (ResNet)**. Hal ini membuktikan keunggulan residual connection dalam mengekstraksi representasi fitur spasial yang lebih kaya dan tangguh.
* **Kasus 2 (Keunggulan CNN)**: Terdapat **690 citra** yang **salah pada Model B, namun berhasil dijawab BENAR oleh Model A**. Ini menunjukkan bahwa pada pola visual bertekstur sederhana, representasi sekuensial konvensional tetap memiliki keunggulan lokal.

---

## 📁 Struktur Direktori Folder Tugas 2

```text
Tugas 2/
├── Tugas_2_CNN_CIFAR10.ipynb          # Kode program utama Jupyter Notebook (Google Colab)
├── Laporan_Hasil_Eksperimen.docx      # Laporan resmi lengkap dengan tabel & grafik visualisasi
├── cache.pkl                          # Cache serialisasi data metrik, evaluasi, & riwayat training
└── README.md                          # Dokumentasi lengkap proyek, identitas, & analisis hasil
```

---

## 💡 Kesimpulan (Berdasarkan Hasil Eksperimen Nyata)

1. **Kinerja Akurasi dan Generalisasi Loss**:  
   Pada pelatihan 20 epoch di CIFAR-10, kedua model mencapai performa yang kompetitif dan relatif seimbang pada data uji *unseen*. Model A (Custom CNN) mencatatkan akurasi pengujian sebesar **81,18%** (Loss: 0,6172), sedangkan Model B (Mini-ResNet) memperoleh akurasi **80,65%** namun unggul pada nilai **Test Loss yang lebih rendah yaitu 0,5971** (serta Validation Loss **0,5853** vs 0,6143 pada Model A). Hal ini membuktikan secara empiris bahwa *residual skip connection* menghasilkan generalisasi dan kalibrasi probabilitas prediksi yang lebih baik terhadap data baru.

2. **Efektivitas Residual Skip Connection pada Jaringan Moderat**:  
   Pada kedalaman 3 blok konvolusi, arsitektur konvensional Model A yang dipadukan dengan Batch Normalization sudah cukup efektif dalam menahan laju *vanishing gradient*. Namun, keberadaan jalur pintas residual $F(x) + x$ pada Model B terbukti memperhalus dinamika penurunan loss (*loss landscape smoothing*) dan meminimalkan fluktuasi osilasi validasi pada epoch-epoch akhir.

3. **Karakteristik Komputasi & Analisis Fitur Komplementer**:  
   Penambahan parameter sebesar **6,72%** pada Model B (akibat shortcut proyeksi $1 \times 1$) hanya memberikan selisih waktu komputasi yang wajar di GPU Tesla T4 (182,06 detik vs 147,60 detik). Analisis kesalahan dua arah membuktikan bahwa Model B berhasil mengoreksi **637 citra yang gagal diprediksi oleh Model A**, mengindikasikan bahwa kedua arsitektur mengekstraksi representasi fitur yang saling melengkapi dan sangat potensial untuk digabungkan menggunakan metode *Ensemble Learning*.