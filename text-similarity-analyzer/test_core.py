"""
test_core.py
============
Script pengujian untuk memverifikasi logika preprocessing dan similarity.
Dijalankan sebelum Streamlit untuk memastikan semua perhitungan benar.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from core.preprocessing import preprocess_all, build_vocabulary
from core.similarity import (
    build_binary_matrix,
    build_similarity_matrix,
    get_similarity_pairs,
    get_calculation_detail,
    cosine_similarity_pair,
    interpret_similarity,
)

# ─────────────────────────────────────────────
# DATA PENGUJIAN (5 contoh teks)
# ─────────────────────────────────────────────
texts = [
    "Pelayanan sangat cepat dan petugas ramah",
    "Pelayanan cepat dan petugas sangat ramah",
    "Proses pelayanan mudah dan cepat",
    "Petugas memberikan pelayanan dengan baik",
    "Informasi pelayanan kurang jelas dan lambat",
]

status_ids = ["S1", "S2", "S3", "S4", "S5"]

print("=" * 60)
print("PENGUJIAN TEXT SIMILARITY ANALYZER")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. PREPROCESSING
# ─────────────────────────────────────────────
print("\n[1] PREPROCESSING")
preprocessed = preprocess_all(texts)
for i, (sid, res) in enumerate(zip(status_ids, preprocessed)):
    print(f"\n  {sid}: {res['original']}")
    print(f"     Case Fold  : {res['case_folded']}")
    print(f"     No Punct   : {res['no_punctuation']}")
    print(f"     Tokens     : {res['tokens']}")

# ─────────────────────────────────────────────
# 2. VOCABULARY
# ─────────────────────────────────────────────
print("\n[2] VOCABULARY")
vocabulary = build_vocabulary(preprocessed)
print(f"  Jumlah kata unik: {len(vocabulary)}")
print(f"  Vocabulary: {vocabulary}")

# ─────────────────────────────────────────────
# 3. BINARY MATRIX
# ─────────────────────────────────────────────
print("\n[3] BINARY MATRIX")
binary_matrix = build_binary_matrix(preprocessed, vocabulary)
print(f"  Shape: {binary_matrix.shape} (n_teks × n_vocabulary)")
for sid, vec in zip(status_ids, binary_matrix):
    print(f"  {sid}: {vec.astype(int)}")

# ─────────────────────────────────────────────
# 4. SIMILARITY MATRIX
# ─────────────────────────────────────────────
print("\n[4] COSINE SIMILARITY MATRIX")
sim_matrix = build_similarity_matrix(binary_matrix)

# Header
header = "       " + "  ".join(f"{sid:>6}" for sid in status_ids)
print(f"  {header}")
for sid, row in zip(status_ids, sim_matrix):
    values = "  ".join(f"{v:6.4f}" for v in row)
    print(f"  {sid}:  {values}")

# ─────────────────────────────────────────────
# 5. VERIFIKASI
# ─────────────────────────────────────────────
print("\n[5] VERIFIKASI")

# Cek diagonal = 1.0
diagonal_ok = all(abs(sim_matrix[i][i] - 1.0) < 1e-9 for i in range(len(status_ids)))
print(f"  ✓ Diagonal matrix bernilai 1.0  : {'PASS' if diagonal_ok else 'FAIL'}")

# Cek simetris
symmetric_ok = np.allclose(sim_matrix, sim_matrix.T)
print(f"  ✓ Matrix simetris               : {'PASS' if symmetric_ok else 'FAIL'}")

# Cek nilai dalam rentang [0,1]
range_ok = np.all((sim_matrix >= 0) & (sim_matrix <= 1))
print(f"  ✓ Nilai dalam rentang [0, 1]    : {'PASS' if range_ok else 'FAIL'}")

# Cek S1 vs S2 (hampir identik, seharusnya sangat tinggi)
s1_s2 = sim_matrix[0][1]
print(f"  ✓ S1 vs S2 similarity           : {s1_s2:.4f} ({interpret_similarity(s1_s2)})")

# Cek division by zero dengan teks kosong
vec_zero = np.zeros(len(vocabulary))
vec_normal = binary_matrix[0]
zero_check = cosine_similarity_pair(vec_zero, vec_normal)
print(f"  ✓ Division by zero (empty text) : {zero_check} ({'PASS' if zero_check == 0.0 else 'FAIL'})")

# ─────────────────────────────────────────────
# 6. TABEL PASANGAN
# ─────────────────────────────────────────────
print("\n[6] TABEL PASANGAN SIMILARITY")
pairs_df = get_similarity_pairs(sim_matrix, status_ids)
print(pairs_df.to_string(index=False))

# ─────────────────────────────────────────────
# 7. DETAIL PERHITUNGAN S1 vs S2
# ─────────────────────────────────────────────
print("\n[7] DETAIL PERHITUNGAN S1 vs S2")
detail = get_calculation_detail("S1", "S2", binary_matrix[0], binary_matrix[1], vocabulary)
print(f"  A · B (dot product) = {detail['dot_product']:.4f}")
print(f"  ‖A‖ (norm S1)       = {detail['norm_a']:.4f}")
print(f"  ‖B‖ (norm S2)       = {detail['norm_b']:.4f}")
print(f"  ‖A‖ × ‖B‖           = {detail['denominator']:.4f}")
print(f"  Cosine Similarity   = {detail['similarity']:.6f}")
print(f"  Interpretasi        : {interpret_similarity(detail['similarity'])}")

print("\n" + "=" * 60)
print("SEMUA PENGUJIAN SELESAI")
print("=" * 60)
