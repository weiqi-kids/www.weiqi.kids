---
sidebar_position: 1
title: Aturan Go
---

# Aturan Go

Aturan Go sangat sederhana, tetapi variasi yang dihasilkan tak terbatas. Inilah daya tarik Go.

## Konsep Dasar

### Papan dan Batu

- **Papan**: Papan standar adalah 19 jalur (19×19 garis), pemula sering menggunakan 9 atau 13 jalur
- **Batu**: Dua warna, hitam dan putih, hitam bermain lebih dulu
- **Posisi Batu**: Batu ditempatkan di persimpangan garis, bukan di dalam kotak

### Tujuan Permainan

Tujuan Go adalah **mengelilingi wilayah**. Di akhir permainan, pihak yang mengelilingi lebih banyak wilayah kosong menang.

---

## Konsep Liberti

**Liberti** adalah konsep terpenting dalam Go. Liberti adalah titik persimpangan kosong di sekitar batu, yaitu "garis kehidupan" batu.

### Liberti Satu Batu

Satu batu di posisi berbeda memiliki jumlah liberti berbeda:

| Posisi | Jumlah Liberti |
|:---:|:---:|
| Tengah | 4 liberti |
| Sisi | 3 liberti |
| Sudut | 2 liberti |

### Batu yang Terhubung

Ketika batu sewarna berdekatan (terhubung atas-bawah-kiri-kanan), mereka menjadi satu kesatuan, berbagi semua liberti.

:::info Konsep Penting
Batu yang terhubung hidup dan mati bersama. Jika kelompok ini ditangkap, semua batu yang terhubung akan diambil bersama.
:::

---

## Penangkapan

Ketika semua liberti sebuah kelompok batu diblokir oleh lawan (liberti = 0), kelompok itu "ditangkap" dan diambil dari papan.

### Langkah-langkah Penangkapan

1. Batu lawan hanya tersisa satu liberti
2. Anda memainkan satu langkah untuk memblokir liberti terakhir
3. Batu lawan ditangkap (diambil dari papan)

### Atari

Ketika sebuah kelompok batu hanya tersisa satu liberti, kondisi ini disebut **atari**. Saat itu lawan harus mencoba melarikan diri atau mengorbankan batu.

---

## Titik Terlarang (Titik Bunuh Diri)

Beberapa posisi tidak boleh dimainkan, ini disebut **titik terlarang** atau **titik bunuh diri**.

### Menentukan Titik Terlarang

Sebuah posisi adalah titik terlarang jika memenuhi semua kondisi berikut:

1. Setelah bermain di sana, batu Anda sendiri tidak memiliki liberti
2. Dan tidak dapat menangkap batu lawan manapun

:::tip Cara Mudah Mengingat
Jika bermain di sana dapat menangkap batu lawan, maka bukan titik terlarang.
:::

### Langkah Bunuh Diri

Bermain satu langkah yang membuat batu Anda sendiri tanpa liberti, dan tidak dapat menangkap batu lawan, disebut "bunuh diri". Aturan Go melarang bunuh diri.

---

## Aturan Ko

**Ko** adalah bentuk khusus dalam Go yang menyebabkan situasi siklus tak terbatas.

### Apa itu Ko

Ketika kedua belah pihak dapat saling menangkap satu batu, dan setelah ditangkap pihak lain juga dapat langsung menangkap kembali, terbentuklah ko.

### Aturan Ko

**Tidak boleh langsung menangkap kembali**. Setelah batu ko ditangkap, harus bermain di tempat lain terlebih dahulu (disebut "mencari ancaman ko"), baru kemudian boleh menangkap kembali.

### Proses Pertarungan Ko

1. Pihak A menangkap ko
2. Pihak B tidak boleh langsung menangkap kembali, harus bermain di tempat lain dulu
3. Pihak A merespons langkah pihak B
4. Pihak B menangkap kembali ko
5. Begitu seterusnya, sampai satu pihak menyerah

:::note Mengapa Ada Aturan Ini
Tanpa aturan ko, kedua belah pihak dapat saling menangkap tanpa batas, permainan tidak akan pernah berakhir.
:::

---

## Mata dan Kelompok Hidup

**Mata** adalah salah satu konsep terpenting dalam Go. Memahami mata berarti memahami hidup-mati batu.

### Apa itu Mata

Mata adalah titik kosong yang sepenuhnya dikelilingi oleh batu Anda sendiri. Lawan tidak dapat bermain di posisi mata (akan menjadi titik terlarang).

### Kondisi Kelompok Hidup

**Sebuah kelompok batu untuk hidup, harus memiliki dua atau lebih mata sejati.**

Mengapa perlu dua mata?

- Jika hanya satu mata, lawan dapat secara bertahap memperketat liberti dari luar
- Akhirnya mata ini adalah liberti terakhir, lawan dapat masuk dan menangkap seluruh kelompok
- Dengan dua mata, lawan tidak dapat menduduki kedua posisi mata secara bersamaan, kelompok ini tidak akan pernah bisa ditangkap

### Mata Sejati dan Mata Palsu

- **Mata Sejati**: Mata lengkap, lawan tidak dapat merusak
- **Mata Palsu**: Terlihat seperti mata, tetapi memiliki cacat, mungkin dapat dirusak lawan

Menentukan mata sejati atau palsu memerlukan melihat posisi diagonal mata, ini adalah pengetahuan hidup-mati tingkat lanjut.

---

## Penentuan Pemenang

Di akhir pertandingan, perlu menghitung wilayah kedua belah pihak untuk menentukan pemenang. Ada dua metode penghitungan utama.

### Metode Menghitung Area (Aturan Jepang/Korea)

Menghitung jumlah titik kosong yang dikelilingi kedua belah pihak.

**Cara Menghitung**:
- Titik kosong (moku) yang dikelilingi pihak sendiri
- Tidak menghitung batu pihak sendiri di papan

**Penentuan Pemenang**: Pihak dengan lebih banyak moku menang.

### Metode Menghitung Batu (Aturan Tiongkok)

Menghitung total batu dan wilayah pihak sendiri.

**Cara Menghitung**:
- Jumlah batu hidup pihak sendiri di papan (zi)
- Ditambah titik kosong yang dikelilingi pihak sendiri (moku)

**Penentuan Pemenang**:
- Papan standar memiliki 361 titik persimpangan
- Pihak yang melebihi 180,5 poin menang

### Komi (Kompensasi)

Karena hitam bermain lebih dulu memiliki keuntungan, putih mendapat kompensasi poin yang disebut "komi".

| Aturan | Komi |
|:---:|:---:|
| Aturan Tiongkok | Hitam memberikan 3,75 zi (7,5 moku) |
| Aturan Jepang | Hitam memberikan 6,5 moku |
| Aturan Korea | Hitam memberikan 6,5 moku |

:::tip Saran untuk Pemula
Saat baru belajar Go, tidak perlu terlalu memperhatikan detail penentuan pemenang. Pahami dulu konsep "liberti" dan "mata" dengan baik, ini adalah dasar terpenting.
:::

---

## Akhir Permainan

### Kapan Berakhir

Ketika kedua belah pihak merasa tidak ada tempat yang bisa dimainkan, pertandingan berakhir. Secara praktis, kedua belah pihak pass berturut-turut.

### Penanganan Akhir Permainan

1. Konfirmasi semua batu mati
2. Ambil batu mati dari papan
3. Hitung wilayah kedua belah pihak
4. Tentukan pemenang berdasarkan aturan

### Menyerah

Selama permainan berlangsung, jika satu pihak merasa tidak ada kemungkinan untuk menang, dapat menyerah kapan saja. Menyerah adalah cara mengakhiri permainan yang umum dan sopan.

