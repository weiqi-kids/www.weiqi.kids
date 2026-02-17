---
sidebar_position: 1
title: Latar Belakang Pengetahuan
---

# Ikhtisar Latar Belakang Pengetahuan

Sebelum masuk ke praktik KataGo, memahami sejarah perkembangan dan teknologi inti AI Go sangat penting. Bab ini akan membawa Anda memahami evolusi teknis dari AlphaGo hingga AI Go modern.

## Mengapa Perlu Memahami Latar Belakang Pengetahuan?

Perkembangan AI Go adalah salah satu terobosan paling menarik di bidang kecerdasan buatan. Pertandingan AlphaGo mengalahkan Lee Sedol pada 2016 bukan hanya tonggak sejarah Go, tetapi juga menandai kesuksesan besar kombinasi deep learning dan reinforcement learning.

Memahami latar belakang pengetahuan ini dapat membantu Anda:

- **Membuat keputusan teknis yang lebih baik**: Memahami kelebihan dan kekurangan berbagai metode, memilih solusi yang cocok untuk proyek Anda
- **Debugging lebih efektif**: Memahami prinsip dasar, lebih mudah mendiagnosis masalah
- **Mengikuti perkembangan terbaru**: Menguasai pengetahuan dasar, lebih mudah memahami makalah dan teknologi baru
- **Berkontribusi pada proyek open source**: Berpartisipasi dalam pengembangan proyek seperti KataGo memerlukan pemahaman mendalam tentang filosofi desainnya

## Isi Bab Ini

### [Pembahasan Makalah AlphaGo](./alphago.md)

Analisis mendalam makalah klasik DeepMind, termasuk:

- Signifikansi sejarah dan dampak AlphaGo
- Desain Policy Network dan Value Network
- Prinsip dan implementasi Monte Carlo Tree Search (MCTS)
- Inovasi metode pelatihan Self-play
- Evolusi dari AlphaGo ke AlphaGo Zero ke AlphaZero

### [Pembahasan Makalah KataGo](./katago-paper.md)

Memahami inovasi teknis AI Go open source terkuat saat ini:

- Perbaikan KataGo dibandingkan AlphaGo
- Metode pelatihan dan penggunaan sumber daya yang lebih efisien
- Implementasi teknis dukungan berbagai aturan Go
- Desain prediksi tingkat kemenangan dan selisih poin secara bersamaan
- Mengapa KataGo dapat mencapai kemampuan bermain lebih kuat dengan sumber daya lebih sedikit

### [Pengenalan AI Go Lainnya](./zen.md)

Memahami ekosistem AI Go secara komprehensif:

- AI Komersial: Tengen (Zen), Jueyi (Tencent), Xingzhen
- AI Open Source: Leela Zero, ELF OpenGo, SAI
- Perbandingan fitur teknis dan skenario aplikasi berbagai AI

## Timeline Perkembangan Teknis

| Waktu | Peristiwa | Kepentingan |
|------|------|--------|
| Oktober 2015 | AlphaGo mengalahkan Fan Hui | AI pertama kali mengalahkan pemain profesional |
| Maret 2016 | AlphaGo mengalahkan Lee Sedol | Pertandingan manusia vs mesin yang mengejutkan dunia |
| Mei 2017 | AlphaGo mengalahkan Ke Jie | Mengkonfirmasi AI melampaui level puncak manusia |
| Oktober 2017 | AlphaGo Zero dipublikasikan | Self-play murni, tanpa catatan permainan manusia |
| Desember 2017 | AlphaZero dipublikasikan | Desain universal, sekaligus menaklukkan Go, catur, shogi |
| 2018 | Leela Zero mencapai level super manusia | Kemenangan komunitas open source |
| 2019 | KataGo dipublikasikan | Metode pelatihan lebih efisien |
| 2020-sekarang | KataGo terus diperbaiki | Menjadi AI Go open source terkuat |

## Preview Konsep Inti

Sebelum membaca bab detail, berikut pengenalan singkat beberapa konsep inti:

### Peran Neural Network dalam Go

```
Status papan → Neural Network → { Policy (probabilitas bermain), Value (evaluasi tingkat kemenangan) }
```

Neural network menerima status papan saat ini sebagai input, menghasilkan dua jenis informasi:
- **Policy**: Probabilitas bermain di setiap posisi, memandu arah pencarian
- **Value**: Perkiraan tingkat kemenangan posisi saat ini, untuk mengevaluasi posisi

### Monte Carlo Tree Search (MCTS)

MCTS adalah algoritma pencarian, menggabungkan neural network untuk menentukan langkah terbaik:

1. **Selection**: Dari root node pilih jalur paling menjanjikan
2. **Expansion**: Perluas kemungkinan langkah baru di leaf node
3. **Evaluation**: Evaluasi nilai posisi dengan neural network
4. **Backpropagation**: Kirim hasil evaluasi kembali memperbarui semua node di jalur

### Self-play

AI bermain melawan dirinya sendiri untuk menghasilkan data pelatihan:

```
Model awal → Self-play → Kumpulkan catatan permainan → Latih model baru → Model lebih kuat → Ulangi
```

Siklus ini memungkinkan AI terus meningkatkan diri, tidak perlu bergantung pada catatan permainan manusia.

## Urutan Membaca yang Disarankan

1. **Baca Pembahasan Makalah AlphaGo dulu**: Membangun kerangka teori dasar
2. **Lalu baca Pembahasan Makalah KataGo**: Memahami perbaikan dan optimisasi terbaru
3. **Terakhir baca Pengenalan AI Go Lainnya**: Memperluas wawasan, memahami berbagai cara implementasi

Siap? Mari kita mulai dari [Pembahasan Makalah AlphaGo](./alphago.md)!

