# Revisi Script K-Means Clustering

Dokumen ini berisi panduan logis selangkah demi selangkah untuk melakukan perbaikan pada file `kmeans_clustering.py`. Instruksi ini disusun agar dapat dieksekusi dengan mudah oleh *junior programmer* maupun model AI tanpa perlu diberikan referensi kode eksplisit.

**ATURAN UTAMA:** Implementasikan instruksi di bawah ini dengan menyusun kode Anda sendiri. Anda diharapkan untuk menggunakan kemampuan logika dan penyelesaian masalah dasar.

## Langkah 1: Pengelolaan Direktori `Hasil Visualisasi`
Agar file proyek lebih rapi, seluruh keluaran berupa gambar/grafik hasil clustering tidak boleh lagi disimpan di direktori utama, melainkan di dalam sebuah folder khusus.

1. **Penentuan Lokasi Folder:** Buat logika untuk menentukan *path* (jalur) ke sebuah folder bernama `Hasil Visualisasi`. Folder ini harus berada dalam direktori yang sama dengan tempat beradanya script `kmeans_clustering.py`.
2. **Pembuatan Folder Otomatis:** Tambahkan sebuah instruksi bersyarat di awal script untuk memeriksa keberadaan folder `Hasil Visualisasi` tersebut. Jika folder tersebut belum ada pada sistem, program harus membuat folder tersebut secara otomatis.

## Langkah 2: Penamaan File Output Dinamis (Auto-Numbering)
Untuk mencegah file hasil visualisasi sebelumnya tertimpa (overwrite) saat program dijalankan kembali, terapkan logika penamaan file dinamis.

1. **Identifikasi Titik Simpan:** Cari bagian di dalam fungsi penyimpanan plot/gambar hasil clustering.
2. **Logika Penamaan:** Tetapkan nama dasar untuk file keluaran, yaitu `hasil_clustering` beserta ekstensi file gambar yang digunakan.
3. **Pengecekan Ketersediaan:** Buat sebuah perulangan untuk memeriksa apakah file dengan nama dasar tersebut sudah ada di dalam folder `Hasil Visualisasi`.
4. **Penambahan Nomor Urut:** Jika file sudah ada, tambahkan nomor urut berformat tanda kurung setelah nama dasar. Mulailah dari angka 1. Contoh: `hasil_clustering (1)`.
5. **Pencarian Nama Unik:** Jika nama `hasil_clustering (1)` juga sudah ada, lanjutkan perulangan ke `hasil_clustering (2)`, dan seterusnya, sampai ditemukan kombinasi nama file yang belum ada di dalam folder tersebut.
6. **Simpan File:** Setelah program menemukan nama file yang kosong (belum terpakai), simpan gambar hasil clustering menggunakan nama final tersebut di dalam folder `Hasil Visualisasi`.

## Langkah 3: Pemilihan Kolom Menggunakan Angka
Bantu pengguna agar tidak perlu lagi mengetik nama kolom secara utuh saat akan memilih sumbu visualisasi. Pengguna cukup memasukkan nomor urutnya saja.

1. **Penomoran Daftar Kolom:** Saat program menampilkan daftar kolom yang bisa dipilih (kolom bertipe numerik), modifikasi tampilannya. Berikan nomor urut untuk setiap nama kolom. Anda bisa memulai penomoran dari 1.
2. **Perubahan Permintaan Input:** Modifikasi kalimat pada fungsi input agar meminta pengguna untuk mengetik angka/nomor dari kolom, bukan namanya.
3. **Validasi Input Pengguna:** Pastikan Anda menambahkan pengecekan bahwa *input* yang dimasukkan oleh pengguna adalah benar-benar angka dan angka tersebut sesuai dengan rentang nomor kolom yang tersedia. Jika pengguna salah memasukkan (misalnya memasukkan huruf atau angka yang di luar batas), berikan pesan error dan minta *input* kembali.
4. **Pemetaan Nomor ke Nama Kolom:** Setelah pengguna memasukkan angka yang valid, buat logika untuk mengambil nama kolom sebenarnya berdasarkan urutan angka tersebut. Gunakan nama kolom yang didapat untuk meneruskan proses K-Means clustering seperti biasa.
