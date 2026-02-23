---
sidebar_position: 10
title: Aturan Kustom dan Varian
description: Penjelasan detail set aturan Go yang didukung KataGo, varian papan, dan konfigurasi kustom
---

# Aturan Kustom dan Varian

Artikel ini memperkenalkan berbagai aturan Go, varian ukuran papan yang didukung KataGo, serta cara mengkonfigurasi aturan kustom.

---

## Gambaran Set Aturan

### Perbandingan Aturan Utama

| Set Aturan | Metode Penghitungan | Komi | Bunuh Diri | Pengembalian Ko |
|------------|---------------------|------|------------|-----------------|
| **Chinese** | Penghitungan area | 7.5 | Dilarang | Dilarang |
| **Japanese** | Penghitungan teritori | 6.5 | Dilarang | Dilarang |
| **Korean** | Penghitungan teritori | 6.5 | Dilarang | Dilarang |
| **AGA** | Campuran | 7.5 | Dilarang | Dilarang |
| **New Zealand** | Penghitungan area | 7 | Diizinkan | Dilarang |
| **Tromp-Taylor** | Penghitungan area | 7.5 | Diizinkan | Dilarang |

### Konfigurasi KataGo

```ini
# config.cfg
rules = chinese           # Set aturan
komi = 7.5               # Komi
boardXSize = 19          # Lebar papan
boardYSize = 19          # Tinggi papan
```

---

## Aturan Tiongkok (Chinese)

### Karakteristik

```
Metode penghitungan: Penghitungan area (batu dan teritori)
Komi: 7.5 poin
Bunuh diri: Dilarang
Pengembalian ko: Dilarang (aturan sederhana)
```

### Penjelasan Penghitungan Area

```
Skor akhir = Jumlah batu sendiri + Jumlah titik kosong sendiri

Contoh:
Batu hitam 120 + Teritori hitam 65 = 185 poin
Batu putih 100 + Teritori putih 75 + Komi 7.5 = 182.5 poin
Hitam menang 2.5 poin
```

### Konfigurasi KataGo

```ini
rules = chinese
komi = 7.5
```

---

## Aturan Jepang (Japanese)

### Karakteristik

```
Metode penghitungan: Penghitungan teritori (hanya titik kosong)
Komi: 6.5 poin
Bunuh diri: Dilarang
Pengembalian ko: Dilarang
Perlu menandai batu mati
```

### Penjelasan Penghitungan Teritori

```
Skor akhir = Titik kosong sendiri + Batu lawan yang ditangkap

Contoh:
Teritori hitam 65 + Tangkapan 10 = 75 poin
Teritori putih 75 + Tangkapan 5 + Komi 6.5 = 86.5 poin
Putih menang 11.5 poin
```

### Penentuan Batu Mati

Aturan Jepang memerlukan kedua belah pihak untuk menyetujui batu mana yang mati:

```python
def is_dead_by_japanese_rules(group, game_state):
    """Penentuan batu mati di bawah aturan Jepang"""
    # Perlu membuktikan bahwa grup batu tidak bisa membuat dua mata
    # Ini adalah kompleksitas aturan Jepang
    pass
```

### Konfigurasi KataGo

```ini
rules = japanese
komi = 6.5
```

---

## Aturan AGA

### Karakteristik

Aturan American Go Association (AGA) menggabungkan keunggulan aturan Tiongkok dan Jepang:

```
Metode penghitungan: Campuran (area atau teritori, hasilnya sama)
Komi: 7.5 poin
Bunuh diri: Dilarang
Putih perlu mengisi satu batu untuk pass
```

### Aturan Pass

```
Hitam pass: Tidak perlu mengisi batu
Putih pass: Perlu menyerahkan satu batu ke Hitam

Ini membuat hasil penghitungan area dan teritori konsisten
```

### Konfigurasi KataGo

```ini
rules = aga
komi = 7.5
```

---

## Aturan Tromp-Taylor

### Karakteristik

Aturan Go paling ringkas, cocok untuk implementasi program:

```
Metode penghitungan: Penghitungan area
Komi: 7.5 poin
Bunuh diri: Diizinkan
Pengembalian ko: Super Ko (dilarang mengulangi posisi apapun)
Tidak perlu menentukan batu mati
```

### Super Ko

```python
def is_superko_violation(new_state, history):
    """Periksa apakah melanggar Super Ko"""
    for past_state in history:
        if new_state == past_state:
            return True
    return False
```

### Penentuan Akhir Permainan

```
Tidak perlu kedua belah pihak menyetujui batu mati
Permainan berlanjut sampai:
1. Kedua belah pihak pass berturut-turut
2. Kemudian gunakan pencarian atau mainkan sampai selesai untuk menentukan teritori
```

### Konfigurasi KataGo

```ini
rules = tromp-taylor
komi = 7.5
```

---

## Varian Ukuran Papan

### Ukuran yang Didukung

KataGo mendukung berbagai ukuran papan:

| Ukuran | Karakteristik | Penggunaan yang Disarankan |
|--------|---------------|---------------------------|
| 9×9 | ~81 titik | Pemula, permainan cepat |
| 13×13 | ~169 titik | Pembelajaran lanjutan |
| 19×19 | 361 titik | Pertandingan standar |
| Kustom | Apa saja | Penelitian, pengujian |

### Cara Konfigurasi

```ini
# Papan 9×9
boardXSize = 9
boardYSize = 9
komi = 5.5

# Papan 13×13
boardXSize = 13
boardYSize = 13
komi = 6.5

# Papan non-persegi
boardXSize = 19
boardYSize = 9
```

### Saran Komi

| Ukuran | Aturan Tiongkok | Aturan Jepang |
|--------|-----------------|---------------|
| 9×9 | 5.5 | 5.5 |
| 13×13 | 6.5 | 6.5 |
| 19×19 | 7.5 | 6.5 |

---

## Pengaturan Handicap

### Permainan Handicap

Handicap adalah cara untuk menyesuaikan perbedaan kekuatan:

```ini
# Handicap 2 batu
handicap = 2

# Handicap 9 batu
handicap = 9
```

### Posisi Handicap

```python
HANDICAP_POSITIONS = {
    2: [(3, 15), (15, 3)],
    3: [(3, 15), (15, 3), (15, 15)],
    4: [(3, 15), (15, 3), (3, 3), (15, 15)],
    # 5-9 batu menggunakan titik bintang + tengen
}
```

### Komi untuk Handicap

```ini
# Tradisional: Tidak ada komi atau setengah komi untuk handicap
komi = 0.5

# Modern: Sesuaikan berdasarkan jumlah handicap
# Setiap batu bernilai sekitar 10-15 poin
```

---

## Pengaturan Aturan Mode Analysis

### Perintah GTP

```gtp
# Atur aturan
kata-set-rules chinese

# Atur komi
komi 7.5

# Atur ukuran papan
boardsize 19
```

### Analysis API

```json
{
  "id": "query1",
  "moves": [["B", "Q4"], ["W", "D4"]],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "overrideSettings": {
    "maxVisits": 1000
  }
}
```

---

## Opsi Aturan Lanjutan

### Pengaturan Bunuh Diri

```ini
# Larang bunuh diri (default)
allowSuicide = false

# Izinkan bunuh diri (gaya Tromp-Taylor)
allowSuicide = true
```

### Aturan Ko

```ini
# Simple Ko (hanya larang pengembalian langsung)
koRule = SIMPLE

# Positional Super Ko (larang mengulangi posisi apapun, tidak peduli giliran siapa)
koRule = POSITIONAL

# Situational Super Ko (larang mengulangi posisi yang dimainkan pemain yang sama)
koRule = SITUATIONAL
```

### Aturan Penilaian

```ini
# Penghitungan area (Tiongkok, AGA)
scoringRule = AREA

# Penghitungan teritori (Jepang, Korea)
scoringRule = TERRITORY
```

### Aturan Pajak

Beberapa aturan memiliki penilaian khusus untuk area seki:

```ini
# Tanpa pajak
taxRule = NONE

# Seki tanpa poin
taxRule = SEKI

# Semua mata tanpa poin
taxRule = ALL
```

---

## Pelatihan Multi-Aturan

### Keunggulan KataGo

KataGo menggunakan satu model untuk mendukung berbagai aturan:

```python
def encode_rules(rules):
    """Encode aturan sebagai input neural network"""
    features = np.zeros(RULE_FEATURE_SIZE)

    # Metode penilaian
    features[0] = 1.0 if rules.scoring == 'area' else 0.0

    # Bunuh diri
    features[1] = 1.0 if rules.allow_suicide else 0.0

    # Aturan ko
    features[2:5] = encode_ko_rule(rules.ko)

    # Komi (dinormalisasi)
    features[5] = rules.komi / 15.0

    return features
```

### Input Sadar Aturan

```
Input neural network mencakup:
- Status papan (19×19×N)
- Vektor fitur aturan (K dimensi)

Ini memungkinkan model yang sama memahami aturan yang berbeda
```

---

## Contoh Pergantian Aturan

### Kode Python

```python
from katago import KataGo

engine = KataGo(model_path="kata.bin.gz")

# Analisis dengan aturan Tiongkok
result_cn = engine.analyze(
    moves=[("B", "Q4"), ("W", "D4")],
    rules="chinese",
    komi=7.5
)

# Analisis dengan aturan Jepang (posisi yang sama)
result_jp = engine.analyze(
    moves=[("B", "Q4"), ("W", "D4")],
    rules="japanese",
    komi=6.5
)

# Bandingkan perbedaan
print(f"Winrate Hitam aturan Tiongkok: {result_cn['winrate']:.1%}")
print(f"Winrate Hitam aturan Jepang: {result_jp['winrate']:.1%}")
```

### Analisis Dampak Aturan

```python
def compare_rules_impact(position, rules_list):
    """Bandingkan dampak aturan berbeda pada evaluasi posisi"""
    results = {}

    for rules in rules_list:
        analysis = engine.analyze(
            moves=position,
            rules=rules,
            komi=get_default_komi(rules)
        )
        results[rules] = {
            'winrate': analysis['winrate'],
            'score': analysis['scoreLead'],
            'best_move': analysis['moveInfos'][0]['move']
        }

    return results
```

---

## Pertanyaan Umum

### Perbedaan Aturan Menyebabkan Hasil Berbeda

```
Permainan yang sama, aturan berbeda bisa menghasilkan hasil berbeda:
- Perbedaan penilaian area vs teritori
- Penanganan area seki
- Dampak langkah kosong (pass)
```

### Aturan Mana yang Dipilih?

| Skenario | Aturan yang Disarankan |
|----------|------------------------|
| Pemula | Chinese (intuitif, tanpa sengketa) |
| Pertandingan online | Default platform (biasanya Chinese) |
| Nihon Ki-in | Japanese |
| Implementasi program | Tromp-Taylor (paling ringkas) |
| Pertandingan profesional Tiongkok | Chinese |

### Apakah Model Perlu Dilatih untuk Aturan Tertentu?

Model multi-aturan KataGo sudah sangat kuat. Tetapi jika hanya menggunakan satu aturan, bisa dipertimbangkan:

```ini
# Pelatihan aturan tetap (mungkin sedikit meningkatkan kekuatan untuk aturan tertentu)
rules = chinese
```

---

## Bacaan Lanjutan

- [Analisis Mekanisme Pelatihan KataGo](../training) — Implementasi pelatihan multi-aturan
- [Integrasi ke Proyek Anda](../../hands-on/integration) — Contoh penggunaan API
- [Evaluasi dan Benchmark](../evaluation) — Pengujian kekuatan dengan aturan berbeda
