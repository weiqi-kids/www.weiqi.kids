---
sidebar_position: 2
title: Sejarah Perkembangan AI Go
---

# Sejarah Perkembangan AI Go

Untuk waktu yang lama, Go dianggap sebagai permainan paling sulit ditaklukkan oleh kecerdasan buatan. Papan dengan 19×19 = 361 titik persimpangan, setiap titik dapat diisi, jumlah variasi melebihi total atom di alam semesta (sekitar 10^170 kemungkinan permainan). Metode pencarian brute-force tradisional sama sekali tidak efektif menghadapi Go.

Namun, antara tahun 2015 hingga 2017, seri program AlphaGo dari DeepMind mengubah semua ini secara total. Revolusi ini tidak hanya mempengaruhi Go, tetapi juga mendorong perkembangan seluruh bidang kecerdasan buatan.

## Mengapa Go Begitu Sulit?

### Ruang Pencarian yang Sangat Besar

Ambil catur internasional sebagai contoh, rata-rata setiap langkah memiliki sekitar 35 langkah legal, satu permainan sekitar 80 langkah. Sedangkan Go rata-rata setiap langkah memiliki sekitar 250 langkah legal, satu permainan sekitar 150 langkah. Ini berarti ruang pencarian Go lebih besar ratusan orde besaran dari catur internasional.

### Posisi yang Sulit Dievaluasi

Setiap bidak catur internasional memiliki nilai yang jelas (ratu 9 poin, benteng 5 poin, dll.), dapat mengevaluasi posisi dengan rumus sederhana. Tetapi dalam Go, nilai satu batu tergantung pada hubungannya dengan batu-batu di sekitarnya, tidak ada metode evaluasi yang sederhana.

Apakah sekelompok batu hidup atau mati? Berapa nilai sebuah pengaruh? Pertanyaan-pertanyaan ini bahkan untuk ahli manusia sering memerlukan perhitungan dan penilaian mendalam.

### Dilema Program Go Awal

Sebelum AlphaGo, program Go terkuat hanya memiliki kemampuan setara amatir 5-6 dan, jauh dari pemain profesional. Program-program ini terutama menggunakan metode "Monte Carlo Tree Search" (MCTS), melalui simulasi acak dalam jumlah besar untuk mengevaluasi posisi.

Tetapi metode ini memiliki keterbatasan yang jelas: simulasi acak tidak dapat menangkap pemikiran strategis dalam Go, program sering membuat kesalahan yang terlihat sangat bodoh bagi manusia.

## Dua Era AI Go

### [Era AlphaGo (2015-2017)](/docs/evolution/ai-history/alphago-era)

Era ini dimulai dari AlphaGo mengalahkan Fan Hui, berakhir dengan publikasi makalah AlphaZero. DeepMind dalam waktu singkat dua tahun, mewujudkan lompatan dari mengalahkan pemain profesional hingga melampaui batas manusia.

Tonggak sejarah utama:
- 2015.10: Mengalahkan Fan Hui (pertama kali mengalahkan pemain profesional)
- 2016.03: Mengalahkan Lee Sedol (4:1)
- 2017.01: Master 60 kemenangan berturut-turut online
- 2017.05: Mengalahkan Ke Jie (3:0)
- 2017.10: AlphaZero dipublikasikan

### [Era KataGo (2019-Sekarang)](/docs/evolution/ai-history/katago-era)

Setelah AlphaGo pensiun, komunitas open source mengambil alih estafet. AI open source seperti KataGo, Leela Zero membuat setiap orang dapat menggunakan mesin Go tingkat atas, secara total mengubah cara belajar dan berlatih Go.

Karakteristik era ini:
- Demokratisasi alat AI
- Pemain profesional secara luas menggunakan AI untuk berlatih
- Gaya bermain manusia menjadi seperti AI
- Peningkatan keseluruhan level Go

## Dampak Kognitif yang Dibawa AI

### Mendefinisikan Ulang "Cara Bermain yang Benar"

Sebelum AI muncul, manusia melalui akumulasi ribuan tahun, membangun seperangkat teori Go yang dianggap "benar". Namun, banyak cara bermain AI bertentangan dengan pemahaman tradisional manusia:

- **Bermain san-san**: Konsep tradisional menganggap langsung bermain san-san saat pembukaan adalah "langkah biasa", AI justru sering melakukan ini
- **Bahu tekan**: Yang dulu dianggap "langkah buruk", di beberapa posisi terbukti oleh AI sebagai pilihan terbaik
- **Serangan kontak dekat**: AI suka pertarungan jarak dekat, berbeda dengan konsep tradisional manusia "serangan dimulai dari jauh"

### Keterbatasan dan Potensi Manusia

Kemunculan AI membuat manusia menyadari keterbatasan diri sendiri, tetapi juga menunjukkan potensi manusia.

Dengan bantuan AI, kecepatan pertumbuhan pemain muda sangat meningkat. Level yang dulu memerlukan sepuluh tahun untuk dicapai, sekarang mungkin hanya perlu tiga sampai lima tahun. Seluruh level Go sedang meningkat.

### Masa Depan Go

Ada yang khawatir AI akan membuat Go kehilangan makna - karena selamanya tidak bisa mengalahkan AI, mengapa masih bermain catur?

Tetapi fakta membuktikan, kekhawatiran ini berlebihan. AI tidak mengakhiri Go, melainkan membuka era baru Go. Pertandingan antara manusia dengan manusia masih penuh dengan kreativitas, emosi, dan ketidakpastian - inilah esensi yang membuat Go menarik.

---

Selanjutnya, mari kita pahami secara detail perkembangan spesifik dua era ini.

- **[Era AlphaGo](/docs/evolution/ai-history/alphago-era)** - Dari mengalahkan pemain profesional hingga melampaui batas manusia
- **[Era KataGo](/docs/evolution/ai-history/katago-era)** - AI open source dan ekosistem baru Go

