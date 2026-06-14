# Simulasi Clustering Profil Karyawan

Aplikasi desktop lokal untuk melakukan segmentasi profil karyawan menggunakan
K-Means dan SVM. Antarmuka disusun sebagai wizard akademik untuk tugas big
data: dataset lama, input, preprocessing, K-Means, SVM, dan ringkasan
ditampilkan sebagai enam tahap presentasi. UI dibuat dengan HTML, CSS, dan
JavaScript, disajikan oleh Flask, lalu dibuka sebagai window native melalui
pywebview.

Saat menjalankan:

```bash
python main.py
```

aplikasi langsung membuka window simulasi. Tidak ada browser eksternal,
layanan cloud, atau input interaktif di terminal.

## Daftar Isi

- [Tujuan](#tujuan)
- [Kategori Profil](#kategori-profil)
- [Arsitektur](#arsitektur)
- [Dataset](#dataset)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Panduan Membaca Kode](#panduan-membaca-kode)
- [Instalasi](#instalasi)
- [Menjalankan Aplikasi](#menjalankan-aplikasi)
- [Menggunakan Aplikasi](#menggunakan-aplikasi)
- [REST API Internal](#rest-api-internal)
- [Validasi Input](#validasi-input)
- [Cache Model](#cache-model)
- [Pengujian](#pengujian)
- [Kompatibilitas](#kompatibilitas)
- [Keterbatasan](#keterbatasan)
- [Troubleshooting](#troubleshooting)

## Tujuan

Aplikasi menerima empat atribut:

- umur;
- total pengalaman kerja;
- tingkat pendidikan;
- departemen.

Data tersebut diproses oleh model lokal untuk mencari kategori profil yang
paling serupa dengan segmentasi pada dataset IBM HR.

Hasil yang ditampilkan meliputi:

- jejak validasi input;
- rumus StandardScaler untuk setiap fitur numerik;
- hasil one-hot encoding departemen;
- vector akhir yang masuk ke model;
- jarak input terhadap setiap centroid K-Means;
- probabilitas setiap kategori dari SVM;
- kategori hasil klasifikasi SVM;
- kategori cluster K-Means;
- confidence SVM;
- status apakah kedua model memberikan kategori yang sama;
- ringkasan data masukan;
- penjelasan singkat untuk kategori hasil.

## Kategori Profil

### Emerging Talent

Profil talenta awal karier dengan usia relatif muda, pendidikan menengah, dan
pengalaman kerja yang masih berkembang.

### Academic Achiever

Profil karyawan relatif muda dengan tingkat pendidikan lebih tinggi
dibandingkan kelompok awal karier lainnya.

### Seasoned Veteran

Profil karyawan senior dengan usia dan total pengalaman kerja paling tinggi
dalam segmentasi dataset.

Label tersebut ditetapkan dari statistik centroid K-Means, bukan dari kolom
`PerformanceRating`.

## Arsitektur

```text
project-big-data/
|-- main.py
|-- requirements.txt
|-- requirements-dev.txt
|-- pytest.ini
|-- data/
|   `-- employee_attrition.csv
|-- employee_app/
|   |-- core/
|   |   |-- config.py
|   |   |-- data_loader.py
|   |   |-- preprocessing.py
|   |   |-- model_bundle.py
|   |   |-- training.py
|   |   `-- predictor.py
|   |-- models/
|   |   |-- kmeans.py
|   |   `-- svm.py
|   |-- ui/
|   |   |-- templates/index.html
|   |   `-- static/
|   |       |-- css/styles.css
|   |       |-- js/app.js
|   |       `-- fonts/Rubik-Variable.ttf
|   |-- api.py
|   `-- desktop.py
|-- artifacts/
|   `-- employee_profile_model.joblib  # Dibuat otomatis
`-- tests/
    |-- conftest.py
    |-- test_data_loader.py
    |-- test_training.py
    |-- test_kmeans.py
    |-- test_svm.py
    |-- test_predictor.py
    `-- test_api.py
```

### Komponen

| Komponen | Tanggung jawab |
|---|---|
| `main.py` | Entry point aplikasi desktop. |
| `core/config.py` | Path, fitur, kategori, dan konfigurasi pipeline. |
| `core/data_loader.py` | Membaca, memvalidasi, dan menghitung hash dataset. |
| `core/preprocessing.py` | StandardScaler, OneHotEncoder, dan trace transformasi. |
| `core/model_bundle.py` | Struktur artefak model yang disimpan Joblib. |
| `core/training.py` | Mengatur training dan cache tanpa logika algoritma. |
| `core/predictor.py` | Validasi request dan koordinasi inferensi. |
| `models/kmeans.py` | Training, labeling, evaluasi, prediksi, dan trace K-Means. |
| `models/svm.py` | Training, cross-validation, prediksi, dan trace SVM. |
| `api.py` | Wizard dan REST API Flask internal. |
| `desktop.py` | Membuat window native pywebview. |
| `ui/` | Template dan aset offline untuk wizard presentasi. |

Flask diberikan langsung sebagai aplikasi WSGI kepada pywebview. pywebview
menjalankan server lokal internal dan menampilkan hasilnya menggunakan web
renderer bawaan sistem operasi.

UI menggunakan font Rubik yang disimpan lokal di
`employee_app/ui/static/fonts/Rubik-Variable.ttf`. Dengan demikian, tampilan
tetap konsisten dan tidak membutuhkan Google Fonts atau koneksi internet.

## Dataset

Dataset disimpan lokal di:

```text
data/employee_attrition.csv
```

Sumber:

```text
https://raw.githubusercontent.com/pplonski/datasets-for-start/master/
employee_attrition/HR-Employee-Attrition-All.csv
```

Karakteristik dataset:

- 1.470 baris;
- 35 kolom asli;
- umur 18-60 tahun;
- total pengalaman 0-40 tahun;
- pendidikan dalam kode 1-5;
- tiga departemen.

Pipeline hanya menggunakan:

| Input aplikasi | Kolom dataset |
|---|---|
| `age` | `Age` |
| `years_experience` | `TotalWorkingYears` |
| `education_level` | `Education` |
| `department` | `Department` |

### Kode Pendidikan

| Kode | Label |
|---:|---|
| 1 | Below College |
| 2 | College |
| 3 | Bachelor |
| 4 | Master |
| 5 | Doctor |

### Departemen

- Human Resources
- Research & Development
- Sales

Dataset disertakan di repo agar aplikasi tetap dapat melakukan training dan
inferensi tanpa koneksi internet.

## Machine Learning Pipeline

### 1. Validasi Dataset

Loader memastikan:

- seluruh fitur model tersedia;
- tidak ada nilai kosong;
- kolom numerik dapat dikonversi;
- rentang numerik sesuai dataset;
- departemen hanya berisi nilai yang didukung.

### 2. Preprocessing

`ColumnTransformer` menerapkan:

- `StandardScaler` pada `Age`, `TotalWorkingYears`, dan `Education`;
- `OneHotEncoder(handle_unknown="ignore")` pada `Department`.

Output dibuat dense karena dataset hanya memiliki sedikit fitur.

### 3. K-Means

Konfigurasi:

```python
KMeans(
    n_clusters=3,
    n_init=20,
    random_state=42,
)
```

K-Means membentuk tiga segmentasi tanpa label target.

### 4. Pemberian Nama Cluster

Nama cluster tidak bergantung pada nomor cluster karena ID K-Means tidak
memiliki makna tetap.

Aturan labeling:

1. Cluster dengan rata-rata pengalaman tertinggi, lalu umur tertinggi,
   menjadi `Seasoned Veteran`.
2. Dari dua cluster tersisa, cluster dengan pendidikan rata-rata tertinggi
   menjadi `Academic Achiever`.
3. Cluster terakhir menjadi `Emerging Talent`.

Aturan ini membuat label konsisten meskipun nomor cluster berubah.

### 5. SVM

SVM RBF dilatih untuk mempelajari label cluster K-Means:

```python
SVC(
    kernel="rbf",
    class_weight="balanced",
    random_state=42,
)
```

`CalibratedClassifierCV` dengan kalibrasi sigmoid digunakan agar model dapat
memberikan probabilitas kategori.

Confidence yang ditampilkan berarti keyakinan SVM terhadap segmentasi profil.
Confidence bukan probabilitas bahwa karyawan akan memiliki performa kerja
tertentu.

### 6. Metrik

Aplikasi menyimpan:

- accuracy SVM terhadap label K-Means;
- balanced accuracy SVM terhadap label K-Means;
- silhouette score K-Means.

Balanced accuracy dihitung menggunakan stratified 5-fold cross-validation.
Pada dataset yang disertakan, konsistensi SVM terhadap K-Means sekitar 98%.
Angka ini tidak mengukur kebenaran performa kerja.

### 7. Visualisasi PCA 2D

Setelah K-Means dilatih, seluruh vektor enam dimensi diproyeksikan menjadi dua
dimensi menggunakan `PCA(n_components=2)`. Centroid K-Means diproyeksikan
dengan objek PCA yang sama.

Plot ini digunakan untuk:

- menunjukkan pola dataset lama dan posisi setiap centroid;
- menempatkan data karyawan baru pada bidang visual yang sama;
- menghasilkan PNG sebelum dan sesudah prediksi untuk laporan proyek.

PCA hanya digunakan sebagai visualisasi. Penentuan cluster tetap memakai
jarak Euclidean pada seluruh fitur hasil preprocessing, bukan hanya koordinat
PC1 dan PC2.

### 8. Plot Batas Keputusan SVM RBF

Tahap SVM menampilkan satu plot `SVC(kernel="rbf")` pada koordinat PCA 2D.
Area berwarna merupakan wilayah keputusan tiap kategori, lingkaran kosong
menunjukkan support vectors, dan marker lime menunjukkan data baru.

Model visual ini sengaja dipisahkan dari model prediksi utama:

- model visual dilatih pada dua koordinat PCA agar batas keputusan dapat
  digambar;
- model utama menggunakan enam fitur hasil preprocessing;
- confidence berasal dari `CalibratedClassifierCV` dengan base estimator
  `SVC(kernel="rbf")`.

Karena PCA mereduksi dimensi, hasil visual 2D dapat berbeda dari model utama.
Jika terjadi, aplikasi menampilkan kedua kategori secara jujur dan hasil akhir
tetap mengikuti model utama.

## Panduan Membaca Kode

Kode dipisahkan berdasarkan tahapan pipeline agar alurnya mudah dipelajari
untuk tugas big data:

```text
core/data_loader.py -> core/preprocessing.py
                              |
                              v
                models/kmeans.py -> models/svm.py
                              |
                              v
              core/training.py -> core/predictor.py
                              |
                              v
             api.py -> ui/static/js/app.js -> wizard
```

Urutan yang disarankan ketika mempelajari kode:

1. Baca `employee_app/core/config.py` untuk melihat fitur dan konstanta.
2. Baca `employee_app/core/data_loader.py` dan `preprocessing.py`.
3. Pelajari `employee_app/models/kmeans.py` dari training hingga trace.
4. Pelajari `employee_app/models/svm.py` dari evaluasi hingga probabilitas.
5. Lihat `build_cluster_projection()` untuk memahami proyeksi PCA laporan.
6. Lihat `build_svm_decision_projection()` untuk batas keputusan RBF 2D.
7. Ikuti orkestrasi `train_model()` di `employee_app/core/training.py`.
8. Ikuti `predict()` di `employee_app/core/predictor.py`.
9. Lihat endpoint `POST /api/predict` di `employee_app/api.py`.
10. Lihat `employee_app/ui/static/js/app.js` untuk plot dan navigasi.

Alur training dapat dibaca sebagai pseudocode berikut:

```python
data = load_dataset()
vector = preprocessor.fit_transform(data)
cluster_id = kmeans.fit_predict(vector)
svm.fit(vector, cluster_id)
save_model()
```

Alur prediksi:

```python
employee = validate_input(request)
vector = preprocessor.transform(employee)
kmeans_result = kmeans.predict(vector)
svm_result = svm.predict(vector)
probability = svm.predict_proba(vector)
```

Trace preprocessing berada di `core/preprocessing.py`. Trace jarak centroid
berada di `models/kmeans.py`, sedangkan trace probabilitas berada di
`models/svm.py`. Nilainya diambil langsung dari objek yang dipakai saat
inferensi, bukan dari perhitungan model lain.

## Instalasi

Disarankan menggunakan Python 3.11 atau lebih baru.

### macOS

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
| Pandas | Membaca dan memvalidasi dataset. |
| NumPy | Representasi numerik hasil model. |
| scikit-learn | Preprocessing, K-Means, SVM, kalibrasi, dan metrik. |
| Joblib | Menyimpan dan membaca cache model. |
| Flask | Wizard dan API internal. |
| pywebview | Window desktop native lintas platform. |

## Menjalankan Aplikasi

Aktifkan virtual environment, lalu:

```bash
python main.py
```

Pada eksekusi pertama:

1. dataset lokal divalidasi;
2. K-Means dan SVM dilatih;
3. artefak disimpan ke `artifacts/employee_profile_model.joblib`;
4. window desktop dibuka.

Eksekusi berikutnya memakai cache sehingga startup lebih cepat.

Menutup window akan menghentikan aplikasi.

## Menggunakan Aplikasi

1. Pelajari plot dataset lama dan posisi tiga centroid.
2. Gunakan **Unduh PNG Laporan** jika plot awal akan dimasukkan ke laporan.
3. Klik **Lanjut ke Input Data**.
4. Isi umur, pengalaman, pendidikan, dan departemen.
5. Klik **Mulai Analisis**.
6. Pelajari hasil standardisasi dan one-hot encoding, lalu klik
   **Lanjut ke K-Means**.
7. Amati marker data baru pada plot, bandingkan jarak centroid, dan unduh PNG
   hasil prediksi jika diperlukan.
8. Klik **Lanjut ke SVM**, pelajari batas keputusan RBF, support vectors, dan
   confidence, lalu buka Ringkasan.

Antarmuka menggunakan wizard enam tahap: Dataset Lama, Input Data,
Preprocessing, K-Means, SVM, dan Ringkasan. Perpindahan dilakukan manual agar
alur pengolahan dapat dijelaskan saat presentasi. Hasil akhir baru ditampilkan
pada Ringkasan. Badge `Model Sepakat` berarti SVM dan K-Means memberikan
kategori yang sama.

## REST API Internal

API dipakai oleh JavaScript di dalam window desktop. API tidak ditujukan untuk
deployment publik.

### `GET /`

Menampilkan wizard simulasi.

### `GET /api/health`

Contoh:

```json
{
  "model_loaded": true,
  "pipeline_version": "2.2.0",
  "status": "ready"
}
```

### `GET /api/model-info`

Mengembalikan jumlah data, daftar fitur, kategori, metrik, versi pipeline,
status cache, dan data visualisasi.

### `POST /api/predict`

Request:

```json
{
  "age": 30,
  "years_experience": 7,
  "education_level": 4,
  "department": "Research & Development"
}
```

Respons ringkas:

```json
{
  "category": "Academic Achiever",
  "svm": {
    "cluster_id": 1,
    "category": "Academic Achiever",
    "confidence": 0.9844
  },
  "kmeans": {
    "cluster_id": 1,
    "category": "Academic Achiever"
  },
  "models_agree": true,
  "process": {
    "preprocessing": {},
    "kmeans": {},
    "svm": {}
  }
}
```

Objek `process` berisi nilai perhitungan yang ditampilkan pada setiap tahap
wizard, termasuk koordinat PCA data baru pada `cluster_plot.new_point`.
Respons juga berisi deskripsi dan ringkasan input.

## Validasi Input

| Field | Aturan |
|---|---|
| `age` | Bilangan bulat 18-60. |
| `years_experience` | Bilangan bulat 0-40. |
| `education_level` | Bilangan bulat 1-5. |
| `department` | Salah satu dari tiga departemen dataset. |

Pengalaman tidak boleh melebihi `age - 14`. Aturan ini mencegah kombinasi
yang secara kronologis tidak masuk akal.

Request tidak valid mendapat status HTTP `400`:

```json
{
  "error": "validation_error",
  "message": "Data input tidak valid.",
  "fields": {
    "age": "Umur harus berada pada rentang 18-60."
  }
}
```

## Cache Model

Artefak Joblib menyimpan:

- preprocessor;
- K-Means;
- calibrated SVM;
- mapping ID cluster ke kategori;
- metrik;
- jumlah baris dataset;
- SHA-256 dataset;
- versi pipeline.

Model otomatis dilatih ulang jika:

- file artefak tidak ada;
- file tidak dapat dibaca;
- hash dataset berubah;
- `PIPELINE_VERSION` berubah.

Artefak diabaikan Git karena dapat dibuat ulang dari dataset lokal.

Untuk memaksa training ulang:

```bash
rm artifacts/employee_profile_model.joblib
python main.py
```

Windows PowerShell:

```powershell
Remove-Item artifacts\employee_profile_model.joblib
python main.py
```

## Pengujian

Pasang dependency development:

```bash
python -m pip install -r requirements-dev.txt
```

Jalankan:

```bash
python -m pytest
```

Test suite mencakup:

- schema dan hash dataset;
- kode pendidikan dan departemen;
- kelengkapan label cluster;
- proyeksi PCA dataset, centroid, dan data baru;
- mesh keputusan RBF, support vectors, dan marker data baru;
- determinisme training;
- ambang konsistensi SVM minimal 90%;
- penggunaan dan invalidasi cache;
- validasi seluruh field;
- kontrak endpoint Flask;
- rendering wizard simulasi;
- kelengkapan trace preprocessing, K-Means, dan SVM.

## Kompatibilitas

### macOS

pywebview menggunakan Cocoa/WebKit melalui PyObjC. Dependency platform
dipasang otomatis oleh pip.

### Windows

pywebview menggunakan renderer native Windows. Windows modern biasanya sudah
memiliki Microsoft Edge WebView2 Runtime.

Seluruh path aplikasi menggunakan `pathlib`, sehingga tidak bergantung pada
format separator macOS atau Windows.

## Keterbatasan

### Bukan Model Performa Kerja

Dataset memiliki `PerformanceRating`, tetapi empat input yang digunakan tidak
mempunyai daya prediksi yang memadai terhadap rating tersebut. Karena itu,
aplikasi secara sengaja memodelkan segmentasi profil K-Means.

### Segmentasi Bersifat Deskriptif

Kategori menggambarkan pola statistik dalam satu dataset IBM HR. Kategori
tidak menyatakan kualitas, produktivitas, loyalitas, atau potensi seseorang.

### Fitur Terbatas

Model hanya memakai umur, pengalaman, pendidikan, dan departemen. Banyak
aspek penting seperti peran, skill, pencapaian, konteks organisasi, dan
preferensi individu tidak tersedia.

### Dataset Bukan Data Organisasi Pengguna

Hasil merefleksikan distribusi dataset contoh. Untuk penggunaan nyata,
pipeline perlu dievaluasi ulang menggunakan data organisasi yang sah,
representatif, dan telah ditinjau dari sisi fairness serta privasi.

### Confidence Bukan Jaminan

Confidence berasal dari SVM yang dikalibrasi terhadap label K-Means. Nilai
tinggi hanya menunjukkan bahwa input dekat dengan pola segmentasi yang
dipelajari.

## Troubleshooting

### `ModuleNotFoundError`

Aktifkan virtual environment dan pasang ulang dependency:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Window tidak muncul di macOS

Pastikan menjalankan Python framework build atau Python resmi dari
python.org/Homebrew dan seluruh dependency pywebview terpasang:

```bash
python -m pip install --force-reinstall pywebview
```

Jalankan aplikasi dari session desktop macOS, bukan SSH tanpa GUI.

### Window kosong di Windows

Pasang atau perbarui Microsoft Edge WebView2 Runtime, kemudian jalankan ulang
aplikasi.

### Model gagal dimuat

Hapus artefak agar pipeline melakukan training ulang:

```bash
rm artifacts/employee_profile_model.joblib
python main.py
```

### Dataset dianggap tidak valid

Pastikan `data/employee_attrition.csv` tidak diubah dan tetap memiliki header
asli. Dataset yang dimodifikasi akan divalidasi sebelum training.

### Training terjadi setiap startup

Pastikan aplikasi mempunyai izin tulis pada folder `artifacts/` dan file
dataset tidak berubah.
