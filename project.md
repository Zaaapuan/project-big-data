# PROJECT PLAN: Customer Review Analysis — Sentiment Clustering

---

## 1. Gambaran Besar Proyek

Program ini adalah sistem analisis ulasan pelanggan (customer review) yang bekerja secara otomatis. Sistem hanya menerima data review pelanggan dalam format **CSV**, kemudian melakukan pengelompokan (clustering) ulasan ke dalam tiga sentimen: **Positif, Negatif, dan Netral**.

Fokus utama sistem ini adalah menggunakan teknik Natural Language Processing (NLP) tingkat lanjut, seperti penanganan *stopwords* multi-bahasa dan negasi kompleks, lalu memvisualisasikan hasil clustering menggunakan K-Means secara intuitif bagi manajemen.

---

## 2. Struktur Folder Proyek

Struktur folder disederhanakan agar lebih rapi dan mudah di-maintenance:

```
project_root/
│
├── main.py                     ← File utama yang menjalankan seluruh proses analisis
├── requirements.txt            ← Daftar library yang perlu diinstall
│
├── data/                       ← Folder untuk meletakkan file input dataset (.csv)
└── output/                     ← Folder untuk menyimpan semua visualisasi dan laporan hasil analisis
```

**Daftar Library (`requirements.txt`)**:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
wordcloud
nltk
Sastrawi
```

---

## 3. Metode dan Teknik NLP yang Digunakan

### K-Means Clustering (K=3)
K-Means digunakan untuk mengelompokkan data teks ke dalam 3 kelompok besar berdasarkan kedekatan/kemiripan fitur kata.
- **K = 2** (Ditetapkan secara hardcode untuk sentimen Positif dan Negatif).

### Stopwords (Inggris & Indonesia)
Untuk membuang kata-kata umum yang tidak bermakna (seperti "dan", "yang", "is", "the"), program menggunakan library standar:
- **Bahasa Inggris**: Menggunakan library `nltk` (Natural Language Toolkit).
- **Bahasa Indonesia**: Menggunakan library `Sastrawi` (atau `nltk` corpus bahasa Indonesia jika tersedia).

### Penanganan Negasi Kompleks
Sistem rentan salah mengklasifikasikan frasa seperti "tidak buruk" karena kata "buruk" biasanya berkonotasi negatif. Untuk menangani hal ini:
- **Penggabungan Token (Negation Handling)**: Sebelum diproses oleh TF-IDF, program menggunakan Regular Expression (RegEx) untuk menggabungkan kata negasi dengan kata sifat setelahnya. 
- *Contoh*: Teks `"tidak buruk"` dan `"kurang bagus"` akan diubah strukturnya menjadi `"tidak_buruk"` dan `"kurang_bagus"`. Dengan demikian, algoritma akan menganggapnya sebagai satu kesatuan kata yang memiliki makna tersendiri, bukan memisahnya menjadi kata negatif.

---

## 4. Alur Kerja Sistem (Pipeline)

1. **Input Data**: Membaca file CSV dari folder `data/`.
2. **Preprocessing & NLP**:
   - Pembersihan teks (huruf kecil, hapus angka dan tanda baca).
   - Penanganan Negasi Kompleks (menggabungkan kata negasi seperti "tidak", "bukan", "kurang" dengan kata berikutnya).
   - Penghapusan Stopwords menggunakan *Sastrawi* dan *NLTK*.
3. **Ekstraksi Fitur**: Mengubah teks bersih menjadi matriks angka menggunakan TF-IDF.
4. **Clustering**: Menjalankan K-Means dengan K=3.
5. **Pelabelan Otomatis (Labeling)**: Menentukan label (Positif, Negatif) pada masing-masing cluster menggunakan lexicon base scoring sederhana atau analisis proporsi *Top Keywords*.
6. **Ekstraksi Insight**: Mengambil *Top Keywords* dan representasi kalimat (*Representative Reviews*) untuk setiap cluster.
7. **Visualisasi & Output**: Menghasilkan scatter plot, grafik distribusi, dan word cloud.

---

## 5. Output Sistem yang Dihasilkan

Sesuai dengan kebutuhan pengambilan keputusan, output yang dihasilkan disederhanakan hanya menjadi 4 komponen utama yang langsung memberikan *insight*:

### 1. Visualisasi Cluster (Peta Kelompok)
Scatter Plot yang mengubah dimensi data TF-IDF yang kompleks menjadi 2 Dimensi (2D).
- **Metode**: Menggunakan algoritma **PCA** (Principal Component Analysis) atau **t-SNE**.
- **Fungsi**: Memperlihatkan secara visual sebaran "bola" kelompok review tersebut. Manajemen dapat melihat apakah ulasan Positif dan Negatif terpisah secara tegas atau ada yang menumpuk di area abu-abu. Disimpan sebagai gambar (misal: `cluster_scatter.png`).

### 2. Label Cluster & Interpretasi (*The "Why"*)
Sistem tidak hanya mencetak angka "Cluster 0, 1, 2", melainkan memberikan konteks dan pembuktian (disimpan ke dalam file teks laporan atau dicetak di terminal):
- **Top Keywords per Cluster**: Menampilkan 5-10 kata yang paling sering muncul dan mendominasi tiap cluster (contoh: Positif dominan dengan "cepat, ramah, puas"; Negatif dominan dengan "lambat, kecewa, rusak, packing").
- **Representatif Review**: Menampilkan 3-5 kalimat ulasan asli (*raw text*) dari masing-masing cluster yang posisinya paling dekat dengan titik pusat (centroid) cluster tersebut. Ini membantu pengguna memvalidasi secara cepat apakah pengelompokan sudah logis.

### 3. Statistik Distribusi (Persentase)
Laporan persentase untuk pengambilan keputusan manajemen secara cepat.
- **Visualisasi**: **Pie Chart** atau **Bar Chart**.
- **Fungsi**: Menunjukkan berapa persen komposisi ulasan pelanggan (contoh: "60% Positif, 25% Netral, 15% Negatif"). Disimpan sebagai gambar (misal: `distribusi_sentimen.png`).

### 4. Word Cloud (Awan Kata per Cluster)
Visualisasi ukuran kata yang proporsional dengan frekuensi kemunculannya.
- **Visualisasi**: 3 buah Word Cloud terpisah untuk setiap cluster (Positif, Negatif).
- **Fungsi**: Sangat efektif untuk bahan presentasi. Word Cloud untuk cluster Negatif, misalnya, akan langsung menonjolkan permasalahan utama (misal kata "ONGKIR" atau "RESPON" muncul paling besar). Disimpan sebagai gambar (misal: `wordcloud_positif.png`, `wordcloud_negatif.png`, `wordcloud_netral.png`).

---

## 6. Aturan Pengembangan Sistem

1. **Hanya CSV**: Sistem mem-bypass semua file selain `.csv`.
2. **Modular namun Sederhana**: Semua alur dipusatkan di file `main.py` (bisa dengan memecah fungsionalitas ke dalam class/fungsi di file yang sama) demi menjaga folder tetap simpel.
3. **Penyimpanan Otomatis**: Semua visualisasi plot dan laporan teks (*Top keywords & representative reviews*) harus tersimpan secara otomatis ke dalam folder `output/`.
4. **Proteksi OOM (Out of Memory)**: Matriks TF-IDF dan t-SNE bisa memakan memori tinggi. Bila dirasa berat, disarankan men-sample data maksimum 10.000 baris atau menggunakan PCA.
