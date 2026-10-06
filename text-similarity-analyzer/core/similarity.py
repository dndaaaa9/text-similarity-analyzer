"""
similarity.py
=============
Modul untuk membuat Binary Representation dan menghitung Cosine Similarity.

Metode:
- Binary Representation : 1 jika kata ada dalam teks, 0 jika tidak ada
- Cosine Similarity     : cosine(A,B) = (A · B) / (||A|| × ||B||)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


def build_binary_vector(tokens: List[str], vocabulary: List[str]) -> np.ndarray:
    """
    Membuat binary vector untuk satu teks berdasarkan vocabulary.

    Aturan:
    - 1 jika kata terdapat dalam daftar token teks tersebut
    - 0 jika kata tidak terdapat dalam daftar token

    Args:
        tokens    : Daftar token dari satu teks (sudah di-preprocess).
        vocabulary: Daftar kata unik dari seluruh teks (vocabulary global).

    Returns:
        numpy array berisi 0 dan 1 (panjang = jumlah kata dalam vocabulary).
    """
    token_set = set(tokens)  # Gunakan set untuk pencarian O(1)
    vector = np.array([1 if word in token_set else 0 for word in vocabulary], dtype=float)
    return vector


def build_binary_matrix(
    preprocessed_results: List[Dict],
    vocabulary: List[str]
) -> np.ndarray:
    """
    Membuat matrix binary representation untuk seluruh teks.

    Args:
        preprocessed_results: Hasil preprocessing (output dari preprocess_all).
        vocabulary          : Vocabulary global dari seluruh teks.

    Returns:
        numpy 2D array dengan shape (n_teks, n_vocabulary).
        Setiap baris adalah binary vector satu teks.
    """
    vectors = []
    for result in preprocessed_results:
        vector = build_binary_vector(result['tokens'], vocabulary)
        vectors.append(vector)
    return np.array(vectors)


def cosine_similarity_pair(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Menghitung Cosine Similarity antara dua vector.

    Rumus: cosine(A,B) = (A · B) / (||A|| × ||B||)

    Catatan:
    - Jika salah satu vector adalah zero vector (teks kosong setelah preprocessing),
      fungsi ini mengembalikan 0.0 untuk menghindari division by zero.
    - Jika A == B (identik), hasilnya akan bernilai 1.0.

    Args:
        vec_a: Binary vector teks A.
        vec_b: Binary vector teks B.

    Returns:
        Nilai cosine similarity antara 0.0 dan 1.0.
    """
    # Hitung dot product (A · B)
    dot_product = np.dot(vec_a, vec_b)

    # Hitung magnitude/norm masing-masing vector
    norm_a = np.linalg.norm(vec_a)  # ||A||
    norm_b = np.linalg.norm(vec_b)  # ||B||

    # Hindari division by zero jika salah satu teks kosong
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)

    # Pastikan hasil dalam rentang [0, 1] (bisa ada floating point error kecil)
    return float(np.clip(similarity, 0.0, 1.0))


def build_similarity_matrix(binary_matrix: np.ndarray) -> np.ndarray:
    """
    Membangun matriks Cosine Similarity untuk seluruh pasangan teks.

    Matriks ini bersifat simetris (similarity(A,B) == similarity(B,A))
    dan diagonal bernilai 1 (similarity(A,A) == 1).

    Args:
        binary_matrix: Matrix binary representation (n_teks × n_vocabulary).

    Returns:
        numpy 2D array (n_teks × n_teks) berisi nilai similarity antar teks.
    """
    n = len(binary_matrix)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sim_matrix[i][j] = cosine_similarity_pair(binary_matrix[i], binary_matrix[j])

    return sim_matrix


def get_similarity_pairs(
    sim_matrix: np.ndarray,
    status_ids: List[str]
) -> pd.DataFrame:
    """
    Menghasilkan tabel pasangan teks beserta nilai similarity dan interpretasinya.

    Hanya pasangan unik yang ditampilkan (i < j), tidak termasuk pasangan diri sendiri.

    Args:
        sim_matrix : Matriks Cosine Similarity.
        status_ids : Daftar ID/label untuk setiap teks (misalnya ['S1', 'S2', ...]).

    Returns:
        DataFrame dengan kolom: Status_1, Status_2, Cosine_Similarity, Interpretasi.
    """
    pairs = []
    n = len(status_ids)

    for i in range(n):
        for j in range(i + 1, n):  # Hanya ambil i < j untuk menghindari duplikasi
            similarity = sim_matrix[i][j]
            interpretation = interpret_similarity(similarity)
            pairs.append({
                'Status 1': status_ids[i],
                'Status 2': status_ids[j],
                'Cosine Similarity': round(similarity, 4),
                'Interpretasi': interpretation
            })

    return pd.DataFrame(pairs)


def interpret_similarity(value: float) -> str:
    """
    Memberikan label interpretasi berdasarkan nilai Cosine Similarity.

    Kategori:
    - 0.80 – 1.00 : Sangat Tinggi
    - 0.60 – 0.79 : Tinggi
    - 0.40 – 0.59 : Sedang
    - 0.20 – 0.39 : Rendah
    - 0.00 – 0.19 : Sangat Rendah

    Args:
        value: Nilai cosine similarity (0.0 – 1.0).

    Returns:
        String kategori interpretasi.
    """
    if value >= 0.80:
        return "Sangat Tinggi"
    elif value >= 0.60:
        return "Tinggi"
    elif value >= 0.40:
        return "Sedang"
    elif value >= 0.20:
        return "Rendah"
    else:
        return "Sangat Rendah"


def get_calculation_detail(
    id_a: str,
    id_b: str,
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    vocabulary: List[str]
) -> Dict:
    """
    Menghasilkan detail matematis perhitungan Cosine Similarity untuk satu pasangan.

    Args:
        id_a      : Label/ID teks A.
        id_b      : Label/ID teks B.
        vec_a     : Binary vector teks A.
        vec_b     : Binary vector teks B.
        vocabulary: Daftar kata dalam vocabulary.

    Returns:
        Dictionary berisi semua komponen perhitungan:
        - dot_product   : nilai A · B
        - norm_a        : nilai ||A||
        - norm_b        : nilai ||B||
        - denominator   : nilai ||A|| × ||B||
        - similarity    : nilai cosine similarity
        - dot_breakdown : daftar perkalian per kata untuk A · B
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    denominator = norm_a * norm_b

    similarity = 0.0
    if denominator > 0:
        similarity = float(np.clip(dot_product / denominator, 0.0, 1.0))

    # Breakdown perkalian elemen per elemen untuk A · B
    dot_breakdown = []
    for word, a_val, b_val in zip(vocabulary, vec_a, vec_b):
        dot_breakdown.append({
            'kata': word,
            'a': int(a_val),
            'b': int(b_val),
            'hasil': int(a_val * b_val)
        })

    return {
        'id_a': id_a,
        'id_b': id_b,
        'dot_product': float(dot_product),
        'norm_a': float(norm_a),
        'norm_b': float(norm_b),
        'denominator': float(denominator),
        'similarity': round(similarity, 6),
        'dot_breakdown': dot_breakdown
    }
