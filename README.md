# Dashboard Clustering Profil Karyawan

Aplikasi desktop lokal untuk melakukan segmentasi profil karyawan menggunakan
K-Means dan SVM. Antarmuka disusun sebagai dashboard akademik untuk tugas big
data: informasi dataset, konfigurasi algoritma, proses inferensi, dan hasil
ditampilkan dalam satu halaman. UI dibuat dengan HTML, CSS, dan JavaScript,
disajikan oleh Flask, lalu dibuka sebagai window native melalui pywebview.

Saat menjalankan:

```bash
python main.py
```

aplikasi langsung membuka window dashboard. Tidak ada browser eksternal,
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
- [Menggunakan Dashboard](#menggunakan-dashboard)
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
- penjelasan kategori dan disclaimer.

Kategori bukan penilaian performa kerja objektif. Aplikasi tidak boleh
digunakan sebagai satu-satunya dasar rekrutmen, promosi, kompensasi, atau
keputusan HR lainnya.

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
|   |-- config.py
|   |-- data_loader.py
|   |-- model_trainer.py
|   |-- predictor.py
|   |-- api.py
|   `-- desktop.py
|-- templates/
|   `-- index.html
|-- static/
|   |-- css/styles.css
|   `-- js/app.js
|-- artifacts/
|   `-- employee_profile_model.joblib  # Dibuat otomatis
`-- tests/
    |-- conftest.py
    |-- test_data_loader.py
    |-- test_model_trainer.py
    |-- test_predictor.py
    `-- test_api.py
```

### Komponen

| Komponen | Tanggung jawab |
|---|---|
| `main.py` | Entry point aplikasi desktop. |
| `config.py` | Path, fitur, kategori, dan konfigurasi pipeline. |
| `data_loader.py` | Membaca, memvalidasi, dan menghitung hash dataset. |
| `model_trainer.py` | Preprocessing, K-Means, labeling, SVM, metrik, dan cache. |
| `predictor.py` | Validasi request dan inferensi model. |
| `api.py` | Dashboard dan REST API Flask internal. |
| `desktop.py` | Membuat window native pywebview. |
| `templates/` dan `static/` | UI offline dashboard. |

Flask diberikan langsung sebagai aplikasi WSGI kepada pywebview. pywebview
menjalankan server lokal internal dan menampilkan hasilnya menggunakan web
renderer bawaan sistem operasi.

Dashboard menggunakan font Rubik yang disimpan lokal di
`static/fonts/Rubik-Variable.ttf`. Dengan demikian, tampilan tetap konsisten
dan tidak membutuhkan Google Fonts atau koneksi internet saat aplikasi dibuka.

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

## Panduan Membaca Kode

Kode dipisahkan berdasarkan tahapan pipeline agar alurnya mudah dipelajari
untuk tugas big data:

```text
data_loader.py
    |
    v
model_trainer.py
    |
    v
predictor.py
    |
    v
api.py -> app.js -> dashboard
```

Urutan yang disarankan ketika mempelajari kode:

1. Baca `employee_app/config.py` untuk melihat fitur, kategori, dan konstanta.
2. Baca `employee_app/data_loader.py` untuk memahami validasi dataset.
3. Ikuti fungsi `train_model()` di `employee_app/model_trainer.py`.
4. Ikuti fungsi `predict()` di `employee_app/predictor.py`.
5. Lihat endpoint `POST /api/predict` di `employee_app/api.py`.
6. Lihat `playEducationalTrace()` di `static/js/app.js` untuk memahami cara
   hasil perhitungan ditampilkan bertahap.

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

Fungsi helper pada `predictor.py` menghasilkan trace edukatif. Trace tersebut
tidak melakukan perhitungan baru yang berbeda dari model; nilainya diambil
langsung dari scaler, encoder, K-Means, dan SVM yang dipakai saat inferensi.

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
| Flask | Dashboard dan API internal. |
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

## Menggunakan Dashboard

1. Isi umur 18-60 tahun.
2. Isi total pengalaman 0-40 tahun.
3. Pilih tingkat pendidikan.
4. Pilih departemen.
5. Klik **Proses Data**.

Dashboard tidak langsung membuka hasil. Panel `Execution Trace` menjalankan
empat presentasi tahap:

1. memvalidasi input;
2. menampilkan standardisasi dan one-hot encoding;
3. membandingkan jarak ke tiga centroid K-Means;
4. menampilkan probabilitas kategori SVM.

Interpretasi akhir baru dibuka setelah seluruh tahap selesai. Badge
`Models agree` berarti SVM dan K-Means memberikan kategori sama.

## REST API Internal

API dipakai oleh JavaScript di dalam window desktop. API tidak ditujukan untuk
deployment publik.

### `GET /`

Menampilkan dashboard.

### `GET /api/health`

Contoh:

```json
{
  "model_loaded": true,
  "pipeline_version": "1.1.0",
  "status": "ready"
}
```

### `GET /api/model-info`

Mengembalikan jumlah data, daftar fitur, kategori, metrik, versi pipeline,
status cache, dan disclaimer.

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
    "input_validation": {},
    "preprocessing": {},
    "kmeans": {},
    "svm": {}
  }
}
```

Objek `process` berisi nilai perhitungan yang ditampilkan pada execution
trace. Respons juga berisi deskripsi, ringkasan input, dan disclaimer.

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
- determinisme training;
- ambang konsistensi SVM minimal 90%;
- penggunaan dan invalidasi cache;
- validasi seluruh field;
- kontrak endpoint Flask;
- rendering dashboard.
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
