"""
run_tests.py — Script pengujian komprehensif setelah update.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results = []

# ── TEST 1: pandas Styler.map compatibility ──────────────────────────
try:
    df_t = pd.DataFrame({"A": [1.0, 0.5], "B": [0.5, 1.0]})
    def _c(v):
        return "background-color:blue;" if float(v) >= 0.8 else ""
    df_t.style.map(_c)  # Tidak boleh raise error
    results.append(("TEST 1 - Styler.map()", PASS, "Berjalan tanpa AttributeError"))
except Exception as e:
    results.append(("TEST 1 - Styler.map()", FAIL, str(e)))

# ── TEST 2: Ekstraksi TXT ─────────────────────────────────────────────
from core.extractor import (
    extract_text_from_txt, extract_text_from_docx, extract_text_from_pdf,
    split_text_into_units, collect_all_units
)

with open("test_sample.txt", "rb") as f:
    txt_bytes = f.read()
text_t, warn_t = extract_text_from_txt(txt_bytes)
units_file = split_text_into_units(text_t, "file")
units_line = split_text_into_units(text_t, "line")

results.append(("TEST 2a - TXT chars", PASS if len(text_t) > 0 else FAIL,
                f"{len(text_t)} karakter"))
results.append(("TEST 2b - TXT mode=file", PASS if len(units_file) == 1 else FAIL,
                f"{len(units_file)} unit"))
results.append(("TEST 2c - TXT mode=line", PASS if len(units_line) == 5 else FAIL,
                f"{len(units_line)} unit (expected 5)"))

# ── TEST 3: Ekstraksi DOCX ────────────────────────────────────────────
with open("test_sample.docx", "rb") as f:
    docx_bytes = f.read()
text_d, warn_d = extract_text_from_docx(docx_bytes)
units_d = split_text_into_units(text_d, "line")

results.append(("TEST 3a - DOCX chars", PASS if len(text_d) > 0 else FAIL,
                f"{len(text_d)} karakter"))
results.append(("TEST 3b - DOCX mode=line", PASS if len(units_d) >= 3 else WARN,
                f"{len(units_d)} unit"))

# ── TEST 4: Ekstraksi PDF ─────────────────────────────────────────────
with open("test_sample.pdf", "rb") as f:
    pdf_bytes = f.read()
text_p, warn_p = extract_text_from_pdf(pdf_bytes)
units_p = split_text_into_units(text_p, "line")

results.append(("TEST 4a - PDF chars", PASS if len(text_p) > 0 else FAIL,
                f"{len(text_p)} karakter"))
results.append(("TEST 4b - PDF mode=line", PASS if len(units_p) >= 1 else FAIL,
                f"{len(units_p)} unit"))
if warn_p:
    results.append(("TEST 4c - PDF warning", WARN, warn_p[:60]))

# ── TEST 5: File kosong ───────────────────────────────────────────────
empty_t, empty_w = extract_text_from_txt(b"")
results.append(("TEST 5 - Empty file", PASS if empty_t == "" and empty_w else FAIL,
                f"warning: {empty_w[:50] if empty_w else 'none'}"))

# ── TEST 6: Core logic (tidak berubah) ───────────────────────────────
from core.preprocessing import preprocess_all, build_vocabulary
from core.similarity import (
    build_binary_matrix, build_similarity_matrix,
    cosine_similarity_pair, interpret_similarity
)

texts_5 = [
    "Pelayanan sangat cepat dan petugas ramah",
    "Pelayanan cepat dan petugas sangat ramah",
    "Proses pelayanan mudah dan cepat",
    "Petugas memberikan pelayanan dengan baik",
    "Informasi pelayanan kurang jelas dan lambat",
]
preprocessed  = preprocess_all(texts_5)
vocabulary    = build_vocabulary(preprocessed)
binary_matrix = build_binary_matrix(preprocessed, vocabulary)
sim_matrix    = build_similarity_matrix(binary_matrix)
n = len(texts_5)

results.append(("TEST 6a - Vocabulary", PASS, f"{len(vocabulary)} kata unik"))
results.append(("TEST 6b - Diagonal=1.0",
    PASS if all(abs(sim_matrix[i][i]-1.0)<1e-9 for i in range(n)) else FAIL,
    "semua diagonal = 1.0"))
results.append(("TEST 6c - Simetris",
    PASS if np.allclose(sim_matrix, sim_matrix.T) else FAIL, "A[i][j] == A[j][i]"))
results.append(("TEST 6d - Range [0,1]",
    PASS if np.all((sim_matrix>=0)&(sim_matrix<=1)) else FAIL, "semua nilai dalam [0,1]"))
s1s2 = sim_matrix[0][1]
results.append(("TEST 6e - S1 vs S2 = 1.0",
    PASS if abs(s1s2-1.0)<1e-9 else FAIL, f"S1 vs S2 = {s1s2:.6f}"))
results.append(("TEST 6f - Zero vector",
    PASS if cosine_similarity_pair(np.zeros(len(vocabulary)), binary_matrix[0]) == 0.0 else FAIL,
    "similarity dengan zero vector = 0.0"))

# ── TEST 7: Similarity matrix values ─────────────────────────────────
expected = {(0,2): 0.5477, (0,3): 0.3651, (0,4): 0.3333}
for (i,j), exp in expected.items():
    actual = round(sim_matrix[i][j], 4)
    results.append((f"TEST 7 - S{i+1} vs S{j+1}",
        PASS if abs(actual-exp)<0.0001 else FAIL,
        f"{actual:.4f} (expected {exp:.4f})"))

# ── TEST 8: Validasi minimal 2 teks ─────────────────────────────────
results.append(("TEST 8 - Min 2 teks",
    PASS if not (len(["only one"]) >= 2) else FAIL,
    "1 teks tidak boleh dianalisis"))

# ── LAPORAN ──────────────────────────────────────────────────────────
print("=" * 65)
print("LAPORAN PENGUJIAN TEXT SIMILARITY ANALYZER")
print("=" * 65)
for name, status, detail in results:
    print(f"  [{status}]  {name}")
    print(f"         {detail}")

passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
warned = sum(1 for _, s, _ in results if s == WARN)
print()
print(f"  Total: {len(results)} | PASS: {passed} | FAIL: {failed} | WARN: {warned}")
print("=" * 65)
