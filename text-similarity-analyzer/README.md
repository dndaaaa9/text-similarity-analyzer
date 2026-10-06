# Text Similarity Analyzer

> Sistem analisis kemiripan teks menggunakan **Binary Representation** dan **Cosine Similarity**.

---

## 📌 Tujuan Sistem

Sistem ini dirancang untuk:
1. Menerima sejumlah teks/status dari pengguna
2. Melakukan preprocessing teks secara transparan
3. Mengubah setiap teks menjadi representasi **Binary (0/1)**
4. Menghitung **Cosine Similarity** antar setiap pasangan teks
5. Menampilkan hasil dalam tabel dan visualisasi yang mudah dipahami

Sistem ini cocok untuk demonstrasi mata kuliah **Text Mining**, **Sistem Informasi**, atau **Information Retrieval**.

---

## 🛠️ Teknologi

| Library | Kegunaan |
|---------|----------|
| **Python 3.9+** | Bahasa pemrograman utama |
| **Streamlit** | Framework antarmuka pengguna web |
| **pandas** | Pengolahan dan tampilan tabel/DataFrame |
| **NumPy** | Komputasi vektor dan matriks |
| **string** (stdlib) | Preprocessing punctuation |
| **re** (stdlib) | Regular expression untuk normalisasi spasi |

> Tidak menggunakan library NLP berat (NLTK, spaCy), TF-IDF, Word2Vec, atau embedding.

---

## 📁 Struktur Project

```
text-similarity-analyzer/
├── app.py                  # Aplikasi Streamlit utama (UI)
├── core/
│   ├── __init__.py
│   ├── preprocessing.py    # Modul preprocessing teks
│   └── similarity.py       # Modul Binary Representation & Cosine Similarity
├── requirements.txt        # Dependensi Python
└── README.md               # Dokumentasi ini
```

---

## ⚙️ Instalasi

### 1. Clone / Unduh Project

```bash
# Masuk ke direktori project
cd text-similarity-analyzer
```

### 2. Buat Virtual Environment (Opsional tapi Direkomendasikan)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

---

## 🚀 Cara Menjalankan

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser secara otomatis di alamat:
```
http://localhost:8501
```

---

## 📋 Metode Preprocessing

### 1. Case Folding
Seluruh karakter dalam teks diubah menjadi **huruf kecil** menggunakan `.lower()`.

```
"Pelayanan CEPAT dan Ramah" → "pelayanan cepat dan ramah"
```

### 2. Penghapusan Tanda Baca
Semua karakter dalam `string.punctuation` dihapus/diganti spasi, kemudian spasi ganda dinormalisasi.

```
"pelayanan cepat, dan ramah!" → "pelayanan cepat  dan ramah"
                              → "pelayanan cepat dan ramah"
```

### 3. Tokenisasi
Teks dipecah menjadi daftar kata (token) menggunakan `.split()`.

```
"pelayanan cepat dan ramah" → ["pelayanan", "cepat", "dan", "ramah"]
```

### 4. Pembangunan Vocabulary
Seluruh kata unik dari semua teks digabung dan diurutkan secara alfabetis.

```
Vocabulary: ["baik", "cepat", "dan", "informasi", "jelas", ...]
```

---

## 🔢 Binary Representation

Setiap teks diubah menjadi **vector biner** berukuran `|Vocabulary|`.

**Aturan:**
- Nilai `1` → kata **hadir** dalam teks tersebut
- Nilai `0` → kata **tidak hadir** dalam teks tersebut

**Contoh:**

| Status | pelayanan | cepat | mudah | lambat |
|--------|-----------|-------|-------|--------|
| S1     | 1         | 1     | 0     | 0      |
| S2     | 1         | 1     | 1     | 0      |
| S3     | 0         | 0     | 1     | 1      |

---

## 📐 Rumus Cosine Similarity

$$\text{cosine}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

Di mana:

- $A \cdot B$ adalah **dot product** antara vector A dan B:
  $$A \cdot B = \sum_{i=1}^{n} A_i \times B_i$$

- $\|A\|$ adalah **magnitude/norm** vector A:
  $$\|A\| = \sqrt{\sum_{i=1}^{n} A_i^2}$$

- $\|B\|$ adalah **magnitude/norm** vector B:
  $$\|B\| = \sqrt{\sum_{i=1}^{n} B_i^2}$$

**Properti:**
- Nilai berkisar antara **0.0** (tidak ada kata yang sama) hingga **1.0** (identik)
- Jika vector zero (teks kosong), similarity = **0.0** (menghindari division by zero)

---

## 🎯 Kategori Interpretasi Similarity

| Rentang Nilai | Kategori |
|---------------|----------|
| 0.80 – 1.00 | Sangat Tinggi |
| 0.60 – 0.79 | Tinggi |
| 0.40 – 0.59 | Sedang |
| 0.20 – 0.39 | Rendah |
| 0.00 – 0.19 | Sangat Rendah |

---

## 📊 Fitur Aplikasi

- ✅ Input teks dinamis (tambah/hapus status bebas)
- ✅ Preprocessing transparan (tabel 4 kolom: Asli → Case Fold → No Punct → Token)
- ✅ Tabel Binary Representation dengan scroll horizontal
- ✅ Matriks Similarity dengan heatmap warna
- ✅ Tabel pasangan similarity dengan interpretasi
- ✅ Detail perhitungan per pasangan (dot product, norm, breakdown per kata)
- ✅ Download 4 file CSV: preprocessing, binary, similarity matrix, pasangan
- ✅ Sidebar dengan panduan penggunaan dan kategori similarity
- ✅ Contoh data built-in (5 status)

---

## 👨‍💻 Catatan Pengembangan

- Logika perhitungan sepenuhnya terpisah dari UI (`core/` vs `app.py`)
- Semua perhitungan menggunakan **NumPy** untuk efisiensi
- Tidak ada database — data bersifat sementara dalam session Streamlit
- Tidak menggunakan API eksternal
