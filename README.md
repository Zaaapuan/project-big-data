# Customer Review Analysis - Sentiment Clustering

Proyek ini adalah aplikasi command-line berbasis Python untuk menganalisis teks
ulasan pelanggan dari file CSV. Program mengelompokkan ulasan menjadi dua
sentimen, yaitu **Positif** dan **Negatif**, lalu menghasilkan laporan teks dan
visualisasi yang dapat dipakai untuk melihat pola umum dalam kumpulan ulasan.

Pendekatan yang digunakan bersifat **hybrid**:

1. **Unsupervised learning** melalui TF-IDF dan K-Means untuk membentuk cluster.
2. **Lexicon-based sentiment rules** untuk memperkuat kata sentimen, memberi
   nama pada cluster, dan memperbaiki ulasan yang masuk ke cluster yang tidak
   sesuai.

Dengan pendekatan tersebut, program tidak hanya mengelompokkan ulasan
berdasarkan kemiripan topik, tetapi mencoba mendorong pemisahan berdasarkan
emosi positif dan negatif.

## Daftar Isi

- [Tujuan Proyek](#tujuan-proyek)
- [Fitur Utama](#fitur-utama)
- [Arsitektur dan Struktur Folder](#arsitektur-dan-struktur-folder)
- [Alur Pemrosesan Data](#alur-pemrosesan-data)
- [Detail Metode](#detail-metode)
- [Format Data Masukan](#format-data-masukan)
- [Instalasi](#instalasi)
- [Cara Menjalankan](#cara-menjalankan)
- [Output Program](#output-program)
- [Konfigurasi Penting](#konfigurasi-penting)
- [Penjelasan Setiap Modul](#penjelasan-setiap-modul)
- [Keterbatasan](#keterbatasan)
- [Troubleshooting](#troubleshooting)
- [Pengembangan Lanjutan](#pengembangan-lanjutan)

## Tujuan Proyek

Program ini dibuat untuk membantu eksplorasi kumpulan ulasan pelanggan tanpa
memerlukan label sentimen yang sudah disiapkan sebelumnya. Masalah yang ingin
diselesaikan adalah:

- membaca kumpulan review dari CSV;
- membersihkan teks berbahasa Indonesia dan Inggris;
- mengubah teks menjadi fitur numerik;
- memisahkan ulasan ke dalam cluster positif dan negatif;
- menunjukkan kata-kata dominan pada setiap cluster;
- memilih beberapa ulasan yang mewakili setiap cluster;
- menyajikan distribusi sentimen dan sebaran cluster dalam bentuk gambar.

Program cocok digunakan sebagai proyek pembelajaran NLP, eksplorasi awal data
review, atau prototipe analisis sentimen. Hasilnya bukan pengganti model
sentimen terlatih untuk kebutuhan produksi yang memerlukan akurasi tinggi.

## Fitur Utama

- Input berupa file `.csv` dari folder `big_data/`.
- Pemilihan kolom teks secara interaktif.
- Penghapusan baris yang mempunyai nilai kosong.
- Sampling otomatis maksimal 10.000 baris untuk mengurangi risiko kehabisan
  memori.
- Pembersihan teks dan penanganan negasi sederhana.
- Stopword bahasa Indonesia, bahasa Inggris, dan kata umum domain e-commerce.
- Ekstraksi fitur TF-IDF unigram dan bigram.
- Penguatan bobot kata sentimen sebesar 10 kali.
- K-Means dengan dua cluster.
- Pemberian label cluster Positif dan Negatif secara otomatis.
- Pemindahan ulang ulasan yang sentimennya jelas berlawanan dengan label
  cluster.
- Ekstraksi 10 keyword dan maksimal 3 representative review per cluster.
- Reduksi dimensi menggunakan PCA untuk scatter plot.
- Pembuatan pie chart distribusi sentimen dan word cloud.

## Arsitektur dan Struktur Folder

```text
project-big-data/
|-- main.py
|-- requirements.txt
|-- README.md
|-- project.md
|-- big_data/
|   `-- reviews.csv
|-- modules/
|   |-- loader.py
|   |-- preprocessor.py
|   |-- clustering.py
|   `-- visualizer.py
`-- output/                    # Dibuat otomatis saat program dijalankan
```

Keterangan:

| Lokasi | Tanggung jawab |
|---|---|
| `main.py` | Entry point dan pengatur urutan seluruh pipeline. |
| `modules/loader.py` | Validasi nama file, membaca CSV, dan membuang baris kosong. |
| `modules/preprocessor.py` | Pemilihan kolom teks, cleaning, stopword removal, dan TF-IDF. |
| `modules/clustering.py` | Sentiment boosting, K-Means, labeling, reassignment, keyword, representative review, dan PCA. |
| `modules/visualizer.py` | Membuat folder output dan seluruh visualisasi. |
| `big_data/` | Tempat file CSV yang akan dianalisis. |
| `output/` | Tempat laporan dan gambar hasil analisis. |
| `project.md` | Dokumen rencana awal proyek; beberapa bagiannya tidak lagi sama dengan implementasi terbaru. |

## Alur Pemrosesan Data

Secara ringkas, alur data program adalah:

```text
CSV
 |
 v
Load dan hapus baris kosong
 |
 v
Sampling maksimal 10.000 baris
 |
 v
Pilih kolom review
 |
 v
Cleaning + penanganan negasi
 |
 v
TF-IDF unigram/bigram
 |
 +--------------------------+
 |                          |
 v                          v
Sentiment feature boost     PCA 2D untuk visualisasi
 |
 v
K-Means (K=2)
 |
 v
Top keywords awal
 |
 v
Label cluster Positif/Negatif
 |
 v
Reassignment berbasis leksikon
 |
 v
Keyword dan representative review akhir
 |
 v
Laporan teks + scatter plot + pie chart + word cloud
```

Tahapan yang dijalankan oleh `main.py`:

1. Membuat folder `output/` apabila belum ada.
2. Meminta nama file CSV yang berada di dalam `big_data/`.
3. Membaca data dan menghapus seluruh baris yang mengandung minimal satu nilai
   kosong.
4. Mengambil sampel acak 10.000 baris jika dataset lebih besar dari batas
   tersebut. Sampling memakai `random_state=42`, sehingga hasilnya konsisten.
5. Menampilkan semua kolom bertipe teks dan meminta pengguna memilih kolom
   review.
6. Membersihkan teks dan membangun matriks TF-IDF.
7. Memperbesar bobot fitur yang dikenali sebagai kata sentimen.
8. Menjalankan K-Means untuk membentuk dua cluster.
9. Menghitung koordinat PCA dari matriks TF-IDF asli.
10. Mengambil keyword awal dan menentukan cluster mana yang Positif atau
    Negatif.
11. Memeriksa setiap review dengan leksikon, kemudian memindahkannya jika
    sentimen individual jelas berlawanan dengan label cluster.
12. Menghitung ulang keyword dan mengambil representative review.
13. Menulis laporan dan membuat seluruh visualisasi.

## Detail Metode

### 1. Membaca dan Membersihkan Baris Data

`load_data()` hanya menerima ekstensi `.csv`. File selalu dicari relatif
terhadap folder `big_data/`.

CSV dibaca oleh Pandas menggunakan:

```python
pd.read_csv(filepath, thousands=',', na_values=['-'])
```

Konsekuensinya:

- tanda koma dapat diperlakukan sebagai pemisah ribuan pada kolom numerik;
- nilai `-` diperlakukan sebagai nilai kosong;
- apabila ada nilai kosong di kolom mana pun, satu baris penuh akan dihapus
  dengan `dropna()`.

### 2. Pemilihan Kolom Review

Program mendeteksi kolom Pandas dengan tipe `object` atau `string`. Semua kolom
teks akan ditampilkan sebagai daftar bernomor. Pengguna kemudian memilih kolom
yang berisi isi review.

Kolom numerik seperti rating tidak digunakan langsung dalam clustering.
Dengan kata lain, keputusan sentimen berasal dari teks, bukan dari nilai
rating.

### 3. Text Cleaning

Fungsi `clean_text()` melakukan langkah berikut:

1. Mengubah semua huruf menjadi lowercase.
2. Menggabungkan kata negasi Indonesia dengan satu kata sesudahnya.
3. Menghapus angka.
4. Menghapus tanda baca dan karakter khusus.
5. Merapikan spasi.

Contoh:

```text
Input : "Barang ini TIDAK bagus, nilainya 2/10!"
Output: "barang ini tidak_bagus nilainya"
```

Kata negasi yang ditangani adalah:

- `tidak`
- `bukan`
- `kurang`
- `jangan`

Underscore dipertahankan supaya bentuk seperti `tidak_bagus` dianggap sebagai
satu token oleh vectorizer.

### 4. Stopword Removal

Daftar stopword merupakan gabungan dari:

- stopword bahasa Inggris dari NLTK;
- stopword bahasa Indonesia dari Sastrawi;
- `CUSTOM_DOMAIN_STOPWORDS` yang didefinisikan di
  `modules/preprocessor.py`.

Stopword domain berisi kata umum yang dianggap lebih menunjukkan topik
daripada emosi, misalnya `barang`, `produk`, `harga`, `shipping`, `delivery`,
`customer`, dan `quality`.

Tujuannya adalah mencegah cluster hanya terbagi berdasarkan jenis barang,
pengiriman, atau topik umum lainnya.

### 5. TF-IDF

Program menggunakan `TfidfVectorizer` dengan konfigurasi:

```python
TfidfVectorizer(
    max_features=5000,
    stop_words=combined_stopwords,
    ngram_range=(1, 2)
)
```

Artinya:

- maksimal 5.000 fitur digunakan;
- stopword dibuang;
- fitur dapat berupa satu kata atau dua kata berurutan;
- kata yang khas pada suatu review mendapat bobot lebih besar daripada kata
  yang sering muncul di semua review.

Matriks TF-IDF disimpan sebagai sparse matrix agar tahap ekstraksi fitur lebih
hemat memori.

### 6. Sentiment Feature Boosting

K-Means biasa cenderung memisahkan teks berdasarkan topik. Untuk menggeser
fokusnya ke sentimen, `boost_sentiment_features()` mencari fitur yang cocok
dengan leksikon `POSITIVE_WORDS` atau `NEGATIVE_WORDS`, lalu mengalikan kolom
fitur tersebut dengan:

```python
SENTIMENT_BOOST_FACTOR = 10.0
```

Contoh kata positif:

```text
bagus, puas, mantap, good, excellent, love, recommend
```

Contoh kata negatif:

```text
buruk, kecewa, rusak, bad, terrible, broken, useless
```

Matriks hasil boosting hanya dipakai saat K-Means. Keyword dan PCA tetap
menggunakan TF-IDF asli agar hasil interpretasi tidak sepenuhnya didominasi
bobot buatan tersebut.

### 7. K-Means

Clustering memakai konfigurasi:

```python
KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)
```

K-Means mencari dua centroid dan menetapkan setiap review ke centroid terdekat
pada ruang fitur yang sudah di-boost. Nilai `random_state=42` membuat
inisialisasi dapat direproduksi, sedangkan `n_init=10` mencoba sepuluh
inisialisasi dan memilih hasil terbaik berdasarkan inertia.

Nomor cluster `0` dan `1` belum mempunyai arti sentimen pada tahap ini.

### 8. Top Keywords dan Auto-Labeling

Untuk setiap cluster, program menghitung rata-rata bobot TF-IDF setiap fitur,
kemudian mengambil 10 fitur dengan rata-rata tertinggi.

Keyword tersebut diberi skor:

- keyword positif: `+1`;
- keyword negatif: `-1`;
- n-gram yang mengandung kata positif: `+0.5`;
- n-gram yang mengandung kata negatif: `-0.5`.

Cluster dengan selisih skor positif-negatif tertinggi diberi label
`Positif`. Cluster dengan skor terendah diberi label `Negatif`. Karena
implementasi utama menetapkan `K=2`, program selalu memetakan satu cluster ke
masing-masing label, bahkan ketika kedua cluster memiliki sinyal sentimen yang
lemah.

### 9. Post-Clustering Reassignment

Setelah cluster diberi nama, setiap review diperiksa kembali. Kata pada review
mentah dibandingkan dengan leksikon positif dan negatif.

- Jika jumlah kata positif lebih banyak minimal satu, review dianggap Positif.
- Jika jumlah kata negatif lebih banyak minimal satu, review dianggap Negatif.
- Jika skor seimbang, label K-Means dipertahankan.
- Jika sentimen individual berlawanan dengan label cluster, review dipindahkan
  ke cluster lain.

Tahap ini membuat hasil akhir bukan K-Means murni. Label akhir merupakan
kombinasi hasil clustering dan aturan sentimen.

### 10. Representative Reviews

Program menghitung centroid baru dari TF-IDF asli untuk anggota setiap cluster,
lalu mengurutkan review berdasarkan jarak Euclidean ke centroid tersebut.

Maksimal sembilan kandidat terdekat diperiksa untuk memperoleh tiga review
yang:

- dekat dengan pusat cluster; dan
- skor leksikonnya sesuai dengan label cluster.

Jika tidak ada kandidat yang sesuai, program memakai review terdekat tanpa
filter sebagai fallback.

### 11. PCA

Untuk menampilkan data berdimensi tinggi pada bidang 2D, program menjalankan:

```python
PCA(n_components=2, random_state=42)
```

PCA dijalankan pada TF-IDF asli yang diubah menjadi dense array. Koordinat
hasilnya disimpan sementara pada kolom `PCA1` dan `PCA2`, kemudian dipakai
untuk scatter plot.

PCA hanya dipakai untuk visualisasi. Koordinat PCA tidak digunakan dalam
proses K-Means.

## Format Data Masukan

Letakkan file CSV di dalam folder `big_data/`. Minimal harus ada satu kolom
teks. Nama kolom bebas karena program akan meminta pengguna memilihnya.

Contoh:

```csv
review_text,rating,product
"This laptop is amazing and has a great battery life.",5,Laptop
"Terrible product. The screen was broken upon arrival.",1,Laptop
"Sangat puas dengan barang ini! Bagus sekali.",5,Smartphone
"Jelek sekali, barangnya rusak saat sampai.",1,Smartphone
```

Dataset contoh `big_data/reviews.csv` memiliki:

- kolom `review_text` sebagai teks utama;
- kolom `rating` sebagai metadata numerik;
- kolom `product` sebagai metadata teks;
- 10 baris ulasan setelah header;
- campuran ulasan bahasa Indonesia dan Inggris.

Saat memilih kolom, pilih `review_text`. Jika memilih `product`, program akan
mengelompokkan nama kategori produk, bukan isi ulasan.

Catatan format:

- encoding CSV sebaiknya UTF-8;
- delimiter yang diharapkan adalah koma;
- file harus memiliki header;
- nilai kosong atau `-` menyebabkan seluruh baris terkait dibuang;
- dataset perlu memiliki setidaknya dua sampel karena K-Means menggunakan dua
  cluster;
- teks tidak boleh seluruhnya kosong atau seluruh tokennya terhapus sebagai
  stopword.

## Instalasi

Direkomendasikan menggunakan virtual environment agar dependency proyek tidak
bercampur dengan instalasi Python global.

### macOS/Linux

```bash
cd /path/ke/project-big-data
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
cd C:\path\ke\project-big-data
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependency utama:

| Library | Kegunaan |
|---|---|
| Pandas | Membaca dan memanipulasi CSV/DataFrame. |
| NumPy | Operasi array dan indeks cluster. |
| Matplotlib | Membuat dan menyimpan plot. |
| Seaborn | Membuat scatter plot. |
| scikit-learn | TF-IDF, K-Means, PCA, dan jarak Euclidean. |
| WordCloud | Membuat awan kata. |
| NLTK | Stopword bahasa Inggris. |
| Sastrawi | Stopword bahasa Indonesia. |

`scipy` juga diimpor langsung oleh kode dan normalnya terpasang sebagai
dependency scikit-learn.

Pada eksekusi pertama, modul preprocessing akan mencari corpus stopword NLTK.
Jika belum tersedia, program otomatis memakai daftar stopword bahasa Inggris
bawaan scikit-learn. Corpus NLTK juga dapat diunduh secara manual:

```bash
python -m nltk.downloader stopwords
```

## Cara Menjalankan

Jalankan program dari root proyek karena path `big_data/` dan `output/`
bersifat relatif terhadap working directory:

```bash
python main.py
```

Contoh interaksi:

```text
Masukkan nama file data CSV (di dalam folder big_data/): reviews.csv

Kolom teks yang tersedia:
  1. review_text
  2. product

Masukkan nomor urut kolom teks utama (review): 1
Kolom teks yang dipilih: 'review_text'
```

Ekstensi `.csv` boleh tidak ditulis:

```text
Masukkan nama file data CSV (di dalam folder big_data/): reviews
```

Program akan otomatis mengubahnya menjadi `reviews.csv`.

Selama proses berjalan, terminal menampilkan:

- jumlah fitur sentimen yang mendapat boosting;
- status K-Means dan PCA;
- skor positif-negatif tiap cluster;
- jumlah review yang dipindahkan saat reassignment;
- laporan keyword dan representative review;
- path setiap file output.

## Output Program

Semua hasil disimpan di folder `output/`, yang dibuat otomatis.

```text
output/
|-- laporan_insight_sentimen.txt
|-- cluster_scatter.png
|-- distribusi_sentimen.png
|-- wordcloud_positif.png
`-- wordcloud_negatif.png
```

### `laporan_insight_sentimen.txt`

Berisi ringkasan untuk setiap sentimen:

- jumlah ulasan;
- 10 top keywords;
- maksimal 3 representative reviews.

Struktur laporan:

```text
=== LAPORAN CLUSTERING SENTIMEN ===

[POSITIF] - <jumlah> Ulasan
Top Keywords: <keyword 1>, <keyword 2>, ...
Representative Reviews:
  1. <review>
  2. <review>
--------------------------------------------------
```

File ditimpa setiap kali program dijalankan.

### `cluster_scatter.png`

Scatter plot koordinat PCA:

- hijau menunjukkan Positif;
- merah menunjukkan Negatif;
- setiap titik mewakili satu review;
- sumbu adalah dua principal component, bukan satuan bisnis langsung.

Titik yang tumpang tindih menandakan representasi TF-IDF kedua review terlihat
mirip pada proyeksi 2D. Plot tidak selalu menunjukkan seluruh struktur pada
ruang fitur asli karena PCA mereduksi ribuan dimensi menjadi dua.

### `distribusi_sentimen.png`

Pie chart persentase jumlah review pada masing-masing label sentimen setelah
reassignment.

### `wordcloud_positif.png` dan `wordcloud_negatif.png`

Word cloud dibuat dari teks yang sudah dibersihkan. Ukuran kata menunjukkan
frekuensi kemunculan, bukan bobot TF-IDF atau tingkat kepentingan kausal.

Word cloud hanya dibuat jika cluster terkait memiliki teks. File lama di
folder `output/` tidak dibersihkan otomatis.

## Konfigurasi Penting

Konfigurasi masih ditulis langsung di source code.

| Konfigurasi | Lokasi | Nilai saat ini | Dampak |
|---|---|---:|---|
| Batas sampling | `main.py` | `10000` | Membatasi jumlah review yang dianalisis. |
| Jumlah cluster | `main.py` | `2` | Menghasilkan label Positif dan Negatif. |
| Maksimal fitur | `modules/preprocessor.py` | `5000` | Membatasi ukuran vocabulary TF-IDF. |
| Rentang n-gram | `modules/preprocessor.py` | `(1, 2)` | Menggunakan unigram dan bigram. |
| Sentiment boost | `modules/clustering.py` | `10.0` | Menentukan kuatnya kata sentimen memengaruhi K-Means. |
| Top keywords | `modules/clustering.py` | `10` | Jumlah keyword per cluster. |
| Representative review | `modules/clustering.py` | `3` | Jumlah contoh review per cluster. |
| Random seed | `main.py` dan `modules/clustering.py` | `42` | Membantu reproduksibilitas sampling dan clustering. |
| Maksimal kata word cloud | `modules/visualizer.py` | `100` | Membatasi kata yang dirender. |

Untuk menyesuaikan domain, bagian yang paling relevan adalah:

- `CUSTOM_DOMAIN_STOPWORDS`;
- `POSITIVE_WORDS`;
- `NEGATIVE_WORDS`;
- `SENTIMENT_BOOST_FACTOR`;
- `SAMPLE_SIZE`.

## Penjelasan Setiap Modul

### `main.py`

Mengatur pipeline dan interaksi pengguna. File ini tidak berisi detail
algoritma utama, tetapi menghubungkan loader, preprocessor, clustering, dan
visualizer.

DataFrame selama proses mendapat kolom tambahan:

| Kolom | Isi |
|---|---|
| `Cluster` | ID cluster akhir setelah reassignment. |
| `PCA1` | Koordinat principal component pertama. |
| `PCA2` | Koordinat principal component kedua. |
| `Sentimen` | Label hasil mapping cluster. |
| `Cleaned_Text` | Teks setelah cleaning untuk word cloud. |

DataFrame tersebut tidak diekspor menjadi CSV pada implementasi saat ini.

### `modules/loader.py`

Fungsi publik:

```python
load_data(filename)
```

Tanggung jawab:

- memastikan ekstensi `.csv`;
- membentuk path `big_data/<filename>`;
- menghentikan program jika file tidak ditemukan;
- membaca CSV;
- menghapus baris yang mengandung nilai kosong.

Kesalahan loader ditangani dengan pesan di terminal dan `sys.exit(1)`.

### `modules/preprocessor.py`

Fungsi publik:

```python
get_text_column(df)
clean_text(text)
prepare_features(df, text_col)
```

Modul ini juga menginisialisasi stopword ketika pertama kali diimpor. Jika
corpus NLTK belum tersedia, modul memakai stopword scikit-learn sehingga akses
internet tidak diperlukan.

### `modules/clustering.py`

Fungsi utama:

```python
boost_sentiment_features(...)
run_kmeans(...)
get_top_keywords(...)
auto_label_clusters(...)
reassign_by_sentiment(...)
get_representative_reviews(...)
reduce_dimensions_pca(...)
```

Modul ini menyimpan leksikon sentimen sebagai Python `set`. Pencocokan
leksikon memakai exact token matching, sehingga variasi kata yang tidak
terdaftar tidak akan diberi skor.

### `modules/visualizer.py`

Fungsi utama:

```python
setup_output_dir()
plot_scatter_pca(...)
plot_sentiment_distribution(...)
plot_wordclouds(...)
```

Setiap fungsi menyimpan gambar dengan resolusi 300 DPI dan menutup figure
setelah penyimpanan, sehingga program tidak membuka jendela plot interaktif.

## Keterbatasan

### Hanya Dua Sentimen

Pipeline utama menggunakan `K=2`, sehingga hanya menghasilkan Positif dan
Negatif. Review netral, ambigu, atau campuran tetap harus masuk ke salah satu
cluster. Kata `Netral` di beberapa fungsi merupakan dukungan fallback untuk
konfigurasi tiga cluster, bukan perilaku default program.

### Hasil Dipengaruhi Leksikon Manual

Kata sentimen yang tidak ada dalam `POSITIVE_WORDS` atau `NEGATIVE_WORDS`
tidak mendapat boosting dan tidak memengaruhi reassignment. Slang, typo,
imbuhan, emoji, dan ekspresi domain khusus dapat terlewat.

### Penanganan Negasi Terbatas

Cleaning menggabungkan negasi bahasa Indonesia dengan satu token berikutnya,
tetapi leksikon belum menafsirkan semua bentuk gabungan secara semantik.
Contohnya, `tidak_bagus` tidak otomatis diperlakukan sebagai lawan dari
`bagus` pada semua tahap. Negasi bahasa Inggris seperti `not good` juga tidak
digabung oleh fungsi cleaning.

### Tokenisasi Reassignment Sederhana

Reassignment memakai `lower().split()` pada teks mentah. Tanda baca tetap
menempel pada token. Kata seperti `great!` atau `rusak.` tidak sama persis
dengan `great` atau `rusak`, sehingga dapat tidak terhitung.

### Auto-Labeling Bersifat Relatif

Cluster dengan skor tertinggi selalu dinamai Positif dan yang terendah selalu
dinamai Negatif. Jika dataset seluruhnya positif atau seluruhnya negatif,
program tetap memaksakan dua label.

### Penggunaan Memori

Beberapa tahap mengubah sparse matrix menjadi dense:

- `get_top_keywords()` melalui `X_tfidf.todense()`;
- `reduce_dimensions_pca()` melalui `X_tfidf.toarray()`.

Sampling 10.000 baris mengurangi risiko, tetapi kombinasi 10.000 baris dan
5.000 fitur masih dapat memakai memori ratusan megabyte. Batas sampling bukan
jaminan bahwa program selalu aman pada mesin dengan RAM terbatas.

### Baris Kosong Dihapus Secara Agresif

Satu baris akan dihapus jika kolom mana pun kosong, meskipun kolom tersebut
tidak digunakan untuk analisis. Dataset dengan metadata opsional dapat
kehilangan banyak review.

### Belum Ada Evaluasi Akurasi

Program tidak memakai label ground truth dan tidak menghitung accuracy,
precision, recall, F1-score, silhouette score, atau metrik evaluasi lainnya.
Output perlu divalidasi secara manual melalui keyword dan representative
review.

### Belum Ada Penyimpanan Hasil Tabular

Label akhir hanya berada di DataFrame selama program berjalan. Program belum
menyimpan CSV baru yang berisi review beserta label sentimennya.

## Troubleshooting

### `ModuleNotFoundError`

Pastikan virtual environment aktif dan dependency sudah terpasang:

```bash
python -m pip install -r requirements.txt
```

### NLTK gagal mengunduh stopword

Unduh corpus secara manual saat koneksi internet tersedia:

```bash
python -m nltk.downloader stopwords
```

### File CSV tidak ditemukan

Pastikan:

- program dijalankan dari root proyek;
- file berada di folder `big_data/`;
- nama dan kapitalisasi file sesuai;
- input berupa nama file, bukan path absolut.

Contoh yang benar:

```text
big_data/my_reviews.csv
```

Kemudian masukkan:

```text
my_reviews.csv
```

### `empty vocabulary`

Error ini dapat muncul jika semua teks kosong atau seluruh token terhapus
sebagai stopword. Periksa kolom yang dipilih dan isi
`CUSTOM_DOMAIN_STOPWORDS`.

### Jumlah sampel lebih sedikit daripada jumlah cluster

K-Means membutuhkan minimal dua baris valid karena `K=2`. Tambahkan data atau
pastikan proses `dropna()` tidak menghapus hampir seluruh dataset.

### PCA kehabisan memori

Kurangi `SAMPLE_SIZE` di `main.py` atau `max_features` di
`modules/preprocessor.py`. Untuk pengembangan lebih lanjut, PCA dense dapat
diganti dengan `TruncatedSVD`, yang dapat bekerja langsung pada sparse matrix.

### Word cloud tidak dibuat

Word cloud dilewati jika cluster tidak memiliki teks. Periksa jumlah anggota
cluster dan pastikan teks tidak kosong setelah cleaning.

## Pengembangan Lanjutan

Beberapa peningkatan yang paling relevan:

1. Menambahkan argumen command-line agar nama file dan kolom dapat diberikan
   tanpa input interaktif.
2. Mengekspor DataFrame beserta kolom `Cluster` dan `Sentimen` ke CSV.
3. Memakai preprocessing yang sama untuk reassignment agar tanda baca dan
   negasi ditangani secara konsisten.
4. Menambahkan stemming atau lemmatization untuk variasi bentuk kata.
5. Memperluas leksikon, termasuk slang, emoji, dan istilah domain.
6. Menambahkan label Netral dengan strategi yang tidak sekadar memaksakan
   cluster ketiga.
7. Menggunakan `MiniBatchKMeans` dan `TruncatedSVD` untuk dataset lebih besar.
8. Menambahkan pengujian unit untuk cleaning, labeling, dan reassignment.
9. Menambahkan evaluasi terhadap dataset berlabel.
10. Membandingkan hasil dengan model supervised atau transformer multilingual.

## Ringkasan

Proyek ini mengubah review mentah menjadi insight melalui rangkaian:

```text
cleaning -> TF-IDF -> sentiment boosting -> K-Means ->
auto-labeling -> reassignment -> report dan visualisasi
```

Kekuatan utamanya adalah pipeline yang sederhana, modular, bilingual, dan
mudah dijelaskan. Hal yang perlu diperhatikan adalah label akhir sangat
dipengaruhi daftar kata manual dan program selalu memaksa pembagian ke dua
sentimen. Karena itu, hasil paling tepat dipakai sebagai analisis eksploratif
dan tetap perlu diperiksa melalui keyword serta representative review yang
dihasilkan.
