---
sidebar_position: 2
title: Perintah Umum
---

# Perintah Umum KataGo

Artikel ini memperkenalkan dua mode operasi utama KataGo: protokol GTP dan Analysis Engine, serta penjelasan detail perintah yang umum digunakan.

## Pengenalan Protokol GTP

GTP (Go Text Protocol) adalah protokol standar untuk komunikasi antar program Go. Sebagian besar GUI Go (seperti Sabaki, Lizzie) menggunakan GTP untuk berkomunikasi dengan mesin AI.

### Menjalankan Mode GTP

```bash
katago gtp -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### Format Dasar Protokol GTP

```
[id] command_name [arguments]
```

- `id`: Nomor perintah opsional, digunakan untuk melacak respons
- `command_name`: Nama perintah
- `arguments`: Parameter perintah

Format respons:
```
=[id] response_data     # Berhasil
?[id] error_message     # Gagal
```

### Contoh Dasar

```
1 name
=1 KataGo

2 version
=2 1.15.3

3 boardsize 19
=3

4 komi 7.5
=4

5 play black Q16
=5

6 genmove white
=6 D4
```

## Perintah GTP Umum

### Informasi Program

| Perintah | Penjelasan | Contoh |
|------|------|------|
| `name` | Mendapatkan nama program | `name` → `= KataGo` |
| `version` | Mendapatkan nomor versi | `version` → `= 1.15.3` |
| `list_commands` | Menampilkan semua perintah yang didukung | `list_commands` |
| `protocol_version` | Versi protokol GTP | `protocol_version` → `= 2` |

### Pengaturan Papan

```
# Mengatur ukuran papan (9, 13, 19)
boardsize 19

# Mengatur komi
komi 7.5

# Membersihkan papan
clear_board

# Mengatur aturan (ekstensi KataGo)
kata-set-rules chinese    # Aturan Tiongkok
kata-set-rules japanese   # Aturan Jepang
kata-set-rules tromp-taylor
```

### Terkait Bermain

```
# Menempatkan batu
play black Q16    # Hitam bermain di Q16
play white D4     # Putih bermain di D4
play black pass   # Hitam pass

# Membuat AI bermain
genmove black     # Menghasilkan langkah hitam
genmove white     # Menghasilkan langkah putih

# Membatalkan
undo              # Membatalkan satu langkah

# Mengatur batas kunjungan
kata-set-param maxVisits 1000    # Mengatur jumlah pencarian maksimum
```

### Query Posisi

```
# Menampilkan papan
showboard

# Mendapatkan pemain saat ini
kata-get-player

# Mendapatkan hasil analisis
kata-analyze black 100    # Menganalisis hitam, 100 kunjungan
```

### Terkait Aturan

```
# Mendapatkan aturan saat ini
kata-get-rules

# Mengatur aturan
kata-set-rules chinese

# Mengatur handicap
fixed_handicap 4     # Posisi handicap 4 batu standar
place_free_handicap 4  # Handicap bebas
```

## Perintah Ekstensi KataGo

KataGo menyediakan banyak perintah ekstensi di luar GTP standar:

### kata-analyze

Menganalisis posisi saat ini secara real-time:

```
kata-analyze [player] [visits] [interval]
```

Parameter:
- `player`: Pihak mana yang dianalisis (black/white)
- `visits`: Jumlah pencarian
- `interval`: Interval laporan (centidetik, 1/100 detik)

Contoh:
```
kata-analyze black 1000 100
```

Output:
```
info move Q3 visits 523 winrate 0.5432 scoreMean 2.31 scoreSelfplay 2.45 prior 0.1234 order 0 pv Q3 R4 Q5 ...
info move D4 visits 312 winrate 0.5123 scoreMean 1.82 scoreSelfplay 1.95 prior 0.0987 order 1 pv D4 C6 E3 ...
...
```

Penjelasan field output:

| Field | Penjelasan |
|------|------|
| `move` | Posisi langkah |
| `visits` | Jumlah kunjungan pencarian |
| `winrate` | Tingkat kemenangan (0-1) |
| `scoreMean` | Prediksi selisih poin |
| `scoreSelfplay` | Prediksi poin self-play |
| `prior` | Probabilitas prior neural network |
| `order` | Urutan peringkat |
| `pv` | Variasi utama (Principal Variation) |

### kata-raw-nn

Mendapatkan output neural network mentah:

```
kata-raw-nn [symmetry]
```

Output mencakup:
- Distribusi probabilitas Policy
- Prediksi Value
- Prediksi wilayah, dll.

### kata-debug-print

Menampilkan informasi pencarian detail, digunakan untuk debugging:

```
kata-debug-print move Q16
```

### Penyesuaian Kekuatan

```
# Mengatur jumlah kunjungan maksimum
kata-set-param maxVisits 100      # Lebih lemah
kata-set-param maxVisits 10000    # Lebih kuat

# Mengatur waktu berpikir
kata-time-settings main 60 0      # 60 detik per pihak
kata-time-settings byoyomi 30 5   # Byoyomi 30 detik 5 kali
```

## Penggunaan Analysis Engine

Analysis Engine adalah mode operasi lain yang disediakan KataGo, menggunakan format JSON untuk komunikasi, lebih cocok untuk penggunaan terprogram.

### Menjalankan Analysis Engine

```bash
katago analysis -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### Alur Penggunaan Dasar

```
Program Anda ──Permintaan JSON──> KataGo Analysis Engine ──Respons JSON──> Program Anda
```

### Format Permintaan

Setiap permintaan adalah objek JSON, harus dalam satu baris:

```json
{
  "id": "query1",
  "moves": [["B","Q16"],["W","D4"],["B","Q4"]],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "analyzeTurns": [2]
}
```

### Penjelasan Field Permintaan

| Field | Wajib | Penjelasan |
|------|------|------|
| `id` | Ya | ID query, digunakan untuk mencocokkan respons |
| `moves` | Tidak | Urutan langkah `[["B","Q16"],["W","D4"]]` |
| `initialStones` | Tidak | Batu awal `[["B","Q16"],["W","D4"]]` |
| `rules` | Ya | Nama aturan |
| `komi` | Ya | Komi |
| `boardXSize` | Ya | Lebar papan |
| `boardYSize` | Ya | Tinggi papan |
| `analyzeTurns` | Tidak | Langkah yang akan dianalisis (0-indexed) |
| `maxVisits` | Tidak | Override maxVisits dari file konfigurasi |

### Format Respons

```json
{
  "id": "query1",
  "turnNumber": 2,
  "moveInfos": [
    {
      "move": "D16",
      "visits": 1234,
      "winrate": 0.5678,
      "scoreMean": 3.21,
      "scoreStdev": 15.4,
      "scoreLead": 3.21,
      "prior": 0.0892,
      "order": 0,
      "pv": ["D16", "Q10", "R14"]
    }
  ],
  "rootInfo": {
    "visits": 5000,
    "winrate": 0.5234,
    "scoreLead": 2.1,
    "scoreSelfplay": 2.3
  },
  "ownership": [...],
  "policy": [...]
}
```

### Penjelasan Field Respons

#### Field moveInfos

| Field | Penjelasan |
|------|------|
| `move` | Koordinat langkah |
| `visits` | Jumlah kunjungan pencarian untuk langkah itu |
| `winrate` | Tingkat kemenangan (0-1, untuk pemain saat ini) |
| `scoreMean` | Prediksi selisih poin akhir |
| `scoreStdev` | Standar deviasi poin |
| `scoreLead` | Poin memimpin saat ini |
| `prior` | Probabilitas prior neural network |
| `order` | Peringkat (0 = terbaik) |
| `pv` | Urutan variasi utama |

#### Field rootInfo

| Field | Penjelasan |
|------|------|
| `visits` | Total kunjungan pencarian |
| `winrate` | Tingkat kemenangan posisi saat ini |
| `scoreLead` | Poin memimpin saat ini |
| `scoreSelfplay` | Prediksi poin self-play |

#### Field ownership

Array satu dimensi, panjang boardXSize × boardYSize, setiap nilai antara -1 dan 1:
- -1: Diprediksi sebagai wilayah putih
- +1: Diprediksi sebagai wilayah hitam
- 0: Belum ditentukan/batas

### Opsi Query Lanjutan

#### Mendapatkan Peta Kepemilikan

```json
{
  "id": "ownership_query",
  "moves": [...],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "analyzeTurns": [10],
  "includeOwnership": true
}
```

#### Mendapatkan Distribusi Policy

```json
{
  "id": "policy_query",
  "moves": [...],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "analyzeTurns": [10],
  "includePolicy": true
}
```

#### Membatasi Jumlah Langkah yang Dilaporkan

```json
{
  "id": "limited_query",
  "moves": [...],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "analyzeTurns": [10],
  "maxMoves": 5
}
```

#### Menganalisis Langkah Tertentu

```json
{
  "id": "specific_moves",
  "moves": [...],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "analyzeTurns": [10],
  "allowMoves": [["B","Q16"],["B","D4"],["B","Q4"]]
}
```

### Contoh Lengkap: Integrasi Python

```python
import subprocess
import json

class KataGoEngine:
    def __init__(self, katago_path, model_path, config_path):
        self.process = subprocess.Popen(
            [katago_path, 'analysis', '-model', model_path, '-config', config_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.query_id = 0

    def analyze(self, moves, rules='chinese', komi=7.5):
        self.query_id += 1

        query = {
            'id': f'query_{self.query_id}',
            'moves': moves,
            'rules': rules,
            'komi': komi,
            'boardXSize': 19,
            'boardYSize': 19,
            'analyzeTurns': [len(moves)],
            'maxVisits': 500,
            'includeOwnership': True
        }

        # Kirim query
        self.process.stdin.write(json.dumps(query) + '\n')
        self.process.stdin.flush()

        # Baca respons
        response_line = self.process.stdout.readline()
        return json.loads(response_line)

    def close(self):
        self.process.terminate()


# Contoh penggunaan
engine = KataGoEngine(
    '/usr/local/bin/katago',
    '/path/to/model.bin.gz',
    '/path/to/config.cfg'
)

# Menganalisis posisi
result = engine.analyze([
    ['B', 'Q16'],
    ['W', 'D4'],
    ['B', 'Q4'],
    ['W', 'D16']
])

# Mencetak langkah terbaik
best_move = result['moveInfos'][0]
print(f"Langkah terbaik: {best_move['move']}")
print(f"Tingkat kemenangan: {best_move['winrate']:.1%}")
print(f"Poin memimpin: {best_move['scoreLead']:.1f}")

engine.close()
```

### Contoh Lengkap: Integrasi Node.js

```javascript
const { spawn } = require('child_process');
const readline = require('readline');

class KataGoEngine {
  constructor(katagoPath, modelPath, configPath) {
    this.process = spawn(katagoPath, [
      'analysis',
      '-model', modelPath,
      '-config', configPath
    ]);

    this.rl = readline.createInterface({
      input: this.process.stdout,
      crlfDelay: Infinity
    });

    this.queryId = 0;
    this.callbacks = new Map();

    this.rl.on('line', (line) => {
      try {
        const response = JSON.parse(line);
        const callback = this.callbacks.get(response.id);
        if (callback) {
          callback(response);
          this.callbacks.delete(response.id);
        }
      } catch (e) {
        console.error('Parse error:', e);
      }
    });
  }

  analyze(moves, options = {}) {
    return new Promise((resolve) => {
      this.queryId++;
      const id = `query_${this.queryId}`;

      const query = {
        id,
        moves,
        rules: options.rules || 'chinese',
        komi: options.komi || 7.5,
        boardXSize: 19,
        boardYSize: 19,
        analyzeTurns: [moves.length],
        maxVisits: options.maxVisits || 500,
        includeOwnership: true
      };

      this.callbacks.set(id, resolve);
      this.process.stdin.write(JSON.stringify(query) + '\n');
    });
  }

  close() {
    this.process.kill();
  }
}

// Contoh penggunaan
async function main() {
  const engine = new KataGoEngine(
    '/usr/local/bin/katago',
    '/path/to/model.bin.gz',
    '/path/to/config.cfg'
  );

  const result = await engine.analyze([
    ['B', 'Q16'],
    ['W', 'D4'],
    ['B', 'Q4']
  ]);

  console.log('Langkah terbaik:', result.moveInfos[0].move);
  console.log('Tingkat kemenangan:', (result.moveInfos[0].winrate * 100).toFixed(1) + '%');

  engine.close();
}

main();
```

## Sistem Koordinat

KataGo menggunakan sistem koordinat Go standar:

### Koordinat Huruf

```
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . . 19
18 . . . . . . . . . . . . . . . . . . . 18
17 . . . . . . . . . . . . . . . . . . . 17
16 . . . + . . . . . + . . . . . + . . . 16
15 . . . . . . . . . . . . . . . . . . . 15
...
 4 . . . + . . . . . + . . . . . + . . .  4
 3 . . . . . . . . . . . . . . . . . . .  3
 2 . . . . . . . . . . . . . . . . . . .  2
 1 . . . . . . . . . . . . . . . . . . .  1
   A B C D E F G H J K L M N O P Q R S T
```

Catatan: Tidak ada huruf I (untuk menghindari kebingungan dengan angka 1).

### Konversi Koordinat

```python
def coord_to_gtp(x, y, board_size=19):
    """Mengkonversi koordinat (x, y) ke format GTP"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    return f"{letters[x]}{board_size - y}"

def gtp_to_coord(gtp_coord, board_size=19):
    """Mengkonversi koordinat GTP ke (x, y)"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    x = letters.index(gtp_coord[0].upper())
    y = board_size - int(gtp_coord[1:])
    return (x, y)
```

## Pola Penggunaan Umum

### Mode Bermain

```bash
# Menjalankan mode GTP
katago gtp -model model.bin.gz -config gtp.cfg

# Urutan perintah GTP
boardsize 19
komi 7.5
play black Q16
genmove white
play black Q4
genmove white
...
```

### Mode Analisis Batch

```python
# Menganalisis semua langkah dalam satu permainan
sgf_moves = parse_sgf('game.sgf')

for i in range(len(sgf_moves)):
    result = engine.analyze(sgf_moves[:i+1])
    winrate = result['rootInfo']['winrate']
    print(f"Langkah {i+1}: Tingkat kemenangan {winrate:.1%}")
```

### Mode Analisis Real-time

Menggunakan `kata-analyze` untuk analisis real-time:

```
kata-analyze black 1000 50
```

Akan menghasilkan output hasil analisis setiap 0.5 detik, sampai mencapai 1000 kunjungan.

## Optimisasi Performa

### Pengaturan Pencarian

```ini
# Meningkatkan jumlah pencarian untuk akurasi lebih tinggi
maxVisits = 1000

# Atau menggunakan kontrol waktu
maxTime = 10  # Maksimal 10 detik berpikir per langkah
```

### Pengaturan Multi-threading

```ini
# Jumlah thread CPU
numSearchThreads = 8

# Pemrosesan batch GPU
numNNServerThreadsPerModel = 2
nnMaxBatchSize = 16
```

### Pengaturan Memori

```ini
# Mengurangi penggunaan memori
nnCacheSizePowerOfTwo = 20  # Default 23
```

## Langkah Selanjutnya

Setelah memahami penggunaan perintah, jika Anda ingin meneliti implementasi KataGo secara mendalam, silakan lanjutkan membaca [Arsitektur Kode Sumber](./architecture.md).

