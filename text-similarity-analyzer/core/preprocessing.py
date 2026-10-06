"""
preprocessing.py
================
Modul untuk melakukan preprocessing teks sebelum perhitungan Cosine Similarity.

Tahapan preprocessing:
1. Case Folding  : Mengubah teks menjadi huruf kecil
2. Hapus Tanda Baca : Menghapus seluruh punctuation
3. Tokenisasi    : Memecah teks menjadi daftar token/kata
"""

import re
import string
from typing import List, Dict


def case_folding(text: str) -> str:
    """
    Mengubah seluruh karakter teks menjadi huruf kecil (lowercase).

    Args:
        text: Teks asli dari pengguna.

    Returns:
        Teks dalam huruf kecil.
    """
    return text.lower()


def remove_punctuation(text: str) -> str:
    """
    Menghapus seluruh tanda baca (punctuation) dari teks.

    Menggunakan string.punctuation yang mencakup: !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~
    Selain itu juga mengganti tanda hubung/dash agar tidak menyisakan spasi ganda.

    Args:
        text: Teks setelah case folding.

    Returns:
        Teks tanpa tanda baca.
    """
    # Buat translation table untuk menghapus semua karakter punctuation
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    cleaned = text.translate(translator)

    # Hapus spasi berlebih yang mungkin terbentuk
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def tokenize(text: str) -> List[str]:
    """
    Memecah teks menjadi daftar token (kata-kata individual).

    Args:
        text: Teks yang sudah bersih (setelah case folding & hapus punctuation).

    Returns:
        Daftar token/kata. Mengembalikan list kosong jika teks kosong.
    """
    if not text.strip():
        return []
    return text.strip().split()


def preprocess_text(text: str) -> Dict[str, object]:
    """
    Menjalankan seluruh pipeline preprocessing pada satu teks.

    Pipeline:
        Teks Asli → Case Folding → Hapus Tanda Baca → Tokenisasi

    Args:
        text: Teks mentah dari pengguna.

    Returns:
        Dictionary berisi hasil setiap tahap preprocessing:
        {
            'original'        : str  - teks asli,
            'case_folded'     : str  - hasil case folding,
            'no_punctuation'  : str  - hasil setelah hapus tanda baca,
            'tokens'          : list - daftar token
        }
    """
    original = text.strip()
    case_folded = case_folding(original)
    no_punct = remove_punctuation(case_folded)
    tokens = tokenize(no_punct)

    return {
        'original': original,
        'case_folded': case_folded,
        'no_punctuation': no_punct,
        'tokens': tokens
    }


def preprocess_all(texts: List[str]) -> List[Dict[str, object]]:
    """
    Menjalankan preprocessing pada seluruh daftar teks.

    Args:
        texts: Daftar teks mentah dari pengguna.

    Returns:
        Daftar hasil preprocessing (list of dict) untuk setiap teks.
    """
    results = []
    for text in texts:
        result = preprocess_text(text)
        results.append(result)
    return results


def build_vocabulary(preprocessed_results: List[Dict]) -> List[str]:
    """
    Membangun vocabulary (kumpulan kata unik) dari seluruh teks yang telah di-tokenisasi.

    Vocabulary diurutkan secara alfabetis agar konsisten.

    Args:
        preprocessed_results: Hasil preprocessing (output dari preprocess_all).

    Returns:
        Daftar kata unik yang terurut secara alfabetis.
    """
    vocab_set = set()
    for result in preprocessed_results:
        for token in result['tokens']:
            vocab_set.add(token)

    # Urutkan alfabetis agar urutan kolom konsisten
    return sorted(list(vocab_set))
