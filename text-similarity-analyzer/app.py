"""
app.py
======
Text Similarity Analyzer — Aplikasi Streamlit (Light Theme).

Sistem analisis kemiripan teks menggunakan:
- Preprocessing (Case Folding, Hapus Tanda Baca, Tokenisasi)
- Binary Representation (0/1)
- Cosine Similarity

Mendukung dua sumber input:
1. Input Manual   : pengguna memasukkan status/teks secara langsung
2. Upload Dokumen : pengguna mengupload dokumen PDF, DOCX, atau TXT
"""

import numpy as np
import pandas as pd
import streamlit as st

# Modul core yang sudah ada (TIDAK DIUBAH)
from core.preprocessing import preprocess_all, build_vocabulary
from core.similarity import (
    build_binary_matrix,
    build_similarity_matrix,
    get_similarity_pairs,
    interpret_similarity,
)
from core.extractor import process_uploaded_files, collect_all_units

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Text Similarity Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS KUSTOM — LIGHT THEME PROFESIONAL & AKADEMIS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global Light Theme ── */
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1200px !important;
}

/* ── Header ── */
.app-header {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #2563eb;
    border-radius: 10px;
    padding: 1.25rem 1.6rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
.app-header h1 {
    color: #0f172a;
    font-size: 1.55rem;
    font-weight: 700;
    margin: 0 0 0.2rem 0;
    letter-spacing: -0.3px;
}
.app-header p {
    color: #64748b;
    font-size: 0.88rem;
    margin: 0;
    font-weight: 400;
}

/* ── Section Title ── */
.section-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #0f172a;
    margin: 1.25rem 0 0.65rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #e2e8f0;
}

/* ── Info & Warning Boxes ── */
.info-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a;
    padding: 0.7rem 0.95rem;
    border-radius: 6px;
    font-size: 0.85rem;
    color: #166534;
    margin-bottom: 0.85rem;
    line-height: 1.45;
}
.info-box-blue {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-left: 4px solid #0284c7;
    padding: 0.7rem 0.95rem;
    border-radius: 6px;
    font-size: 0.85rem;
    color: #0369a1;
    margin-bottom: 0.85rem;
    line-height: 1.45;
}
.warn-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    padding: 0.7rem 0.95rem;
    border-radius: 6px;
    font-size: 0.85rem;
    color: #92400e;
    margin-bottom: 0.65rem;
    line-height: 1.45;
}

/* ── Metric Cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.95rem 0.75rem;
    text-align: center;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}
.metric-card .mv {
    font-size: 1.55rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.metric-card .ml {
    font-size: 0.72rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

/* ── Status Card Container ── */
.status-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.6rem 0.85rem;
    margin-bottom: 0.65rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
}
.status-badge {
    font-size: 0.78rem;
    font-weight: 700;
    color: #2563eb;
    margin-bottom: 0.25rem;
}

/* ── Step Badge ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    color: #334155;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.25rem 0.65rem;
    border-radius: 16px;
    margin-bottom: 0.75rem;
}

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 2.5rem 1rem;
    color: #94a3b8;
    background: #ffffff;
    border: 1px dashed #cbd5e1;
    border-radius: 8px;
}
.empty-state .es-icon { font-size: 2.2rem; margin-bottom: 0.4rem; }
.empty-state .es-text { font-size: 0.9rem; color: #64748b; }

/* ── Calculation Box (Detail) ── */
.calc-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.75rem;
}
.calc-formula-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.88rem;
    color: #0f172a;
    line-height: 1.5;
    white-space: pre-wrap;
    margin: 0.6rem 0 0.2rem 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
.sidebar-title {
    font-size: 1rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: 0.5px;
    margin-bottom: 0.25rem;
    line-height: 1.2;
}
.sidebar-desc {
    font-size: 0.8rem;
    color: #64748b;
    margin-bottom: 1rem;
    line-height: 1.45;
}
.sidebar-heading {
    font-size: 0.72rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin: 0.85rem 0 0.4rem 0;
}
.sidebar-item {
    font-size: 0.82rem;
    color: #334155;
    margin-bottom: 0.25rem;
    line-height: 1.4;
}
.sidebar-stat-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.6rem 0.75rem;
    margin-top: 0.5rem;
}
.sidebar-stat-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    padding: 0.2rem 0;
}

/* ── Streamlit UI Elements ── */
[data-testid="stTab"] {
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #64748b !important;
}
[data-testid="stTab"][aria-selected="true"] {
    color: #2563eb !important;
    border-bottom-color: #2563eb !important;
}
[data-testid="stDataFrame"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    margin-bottom: 0.5rem !important;
}
.stTextArea textarea {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    font-size: 0.88rem !important;
}
.stTextArea textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 1px #2563eb !important;
}
.stButton button {
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────
def init_session_state():
    if "texts" not in st.session_state:
        st.session_state.texts = ["", ""]
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "manual"
    if "file_results" not in st.session_state:
        st.session_state.file_results = []
    if "split_mode" not in st.session_state:
        st.session_state.split_mode = "file"


def reset_data():
    st.session_state.texts = ["", ""]
    st.session_state.analysis_done = False
    st.session_state.results = {}
    st.session_state.file_results = []


init_session_state()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-title">TEXT SIMILARITY<br>ANALYZER</div>
    <div class="sidebar-desc">
        Analisis kemiripan teks menggunakan Binary Representation dan Cosine Similarity.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-heading">METODE ANALISIS</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.78rem; font-weight:700; color:#0f172a; margin-top:0.4rem;">Preprocessing</div>
    <div class="sidebar-item">• Case Folding</div>
    <div class="sidebar-item">• Remove Punctuation</div>
    <div class="sidebar-item">• Tokenization</div>

    <div style="font-size:0.78rem; font-weight:700; color:#0f172a; margin-top:0.6rem;">Representation</div>
    <div class="sidebar-item">• Binary (0/1)</div>

    <div style="font-size:0.78rem; font-weight:700; color:#0f172a; margin-top:0.6rem;">Similarity</div>
    <div class="sidebar-item">• Cosine Similarity</div>
    """, unsafe_allow_html=True)

    # Statistik jika analisis sudah tersedia
    if st.session_state.analysis_done:
        summary = st.session_state.results["summary"]
        st.markdown('<div class="sidebar-heading">STATISTIK DATA</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sidebar-stat-box">
            <div class="sidebar-stat-row">
                <span style="color:#64748b;">Jumlah Status</span>
                <span style="font-weight:700; color:#0f172a;">{summary['jumlah_status']}</span>
            </div>
            <div class="sidebar-stat-row">
                <span style="color:#64748b;">Jumlah Vocabulary</span>
                <span style="font-weight:700; color:#0f172a;">{summary['jumlah_vocabulary']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Reset Data", use_container_width=True, type="secondary"):
        reset_data()
        st.rerun()


# ─────────────────────────────────────────────
# HEADER UTAMA
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>Text Similarity Analyzer</h1>
    <p>Analisis Kemiripan Teks Menggunakan Binary Representation dan Cosine Similarity</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS (CSV & DISPLAY)
# ─────────────────────────────────────────────
def df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def matrix_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True, encoding="utf-8-sig").encode("utf-8-sig")


def empty_state(icon: str, message: str):
    st.markdown(
        f'<div class="empty-state"><div class="es-icon">{icon}</div>'
        f'<div class="es-text">{message}</div></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# HEATMAP STYLING (MENGGUNAKAN .map() UNTUK PANDAS)
# ─────────────────────────────────────────────
def color_sim_cell(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v >= 0.80:
        return "background-color:#bfdbfe; color:#1e3a8a; text-align:center; font-weight:700;"
    elif v >= 0.60:
        return "background-color:#dbeafe; color:#1e40af; text-align:center; font-weight:600;"
    elif v >= 0.40:
        return "background-color:#eff6ff; color:#1d4ed8; text-align:center; font-weight:600;"
    elif v >= 0.20:
        return "background-color:#f8fafc; color:#475569; text-align:center; font-weight:500;"
    else:
        return "background-color:#ffffff; color:#94a3b8; text-align:center; font-weight:400;"


def color_interp_cell(val):
    return {
        "Sangat Tinggi": "background-color:#dbeafe; color:#1e40af; font-weight:600;",
        "Tinggi":        "background-color:#e0e7ff; color:#3730a3; font-weight:600;",
        "Sedang":        "background-color:#f1f5f9; color:#334155; font-weight:500;",
        "Rendah":        "background-color:#fef3c7; color:#92400e; font-weight:500;",
        "Sangat Rendah": "background-color:#f8fafc; color:#94a3b8; font-weight:400;",
    }.get(val, "")


# ─────────────────────────────────────────────
# FUNGSI ANALISIS UTAMA
# (Tetap memakai pipeline core/ tanpa mengubah algoritma)
# ─────────────────────────────────────────────
def run_analysis(texts: list) -> dict:
    valid_texts = [(i, t) for i, t in enumerate(texts) if t.strip()]
    indices, valid = zip(*valid_texts) if valid_texts else ([], [])
    status_ids = [f"S{i+1}" for i in range(len(valid))]

    preprocessed   = preprocess_all(list(valid))
    vocabulary     = build_vocabulary(preprocessed)
    binary_matrix  = build_binary_matrix(preprocessed, vocabulary)
    sim_matrix     = build_similarity_matrix(binary_matrix)
    pairs_df       = get_similarity_pairs(sim_matrix, status_ids)

    prep_rows = []
    for sid, res in zip(status_ids, preprocessed):
        prep_rows.append({
            "ID": sid,
            "Teks Asli": res["original"],
            "Case Folding": res["case_folded"],
            "Hapus Tanda Baca": res["no_punctuation"],
            "Token": ", ".join(res["tokens"]) if res["tokens"] else "(kosong)",
        })
    prep_df = pd.DataFrame(prep_rows)

    binary_df = pd.DataFrame(binary_matrix, columns=vocabulary)
    binary_df.insert(0, "Status", status_ids)

    sim_df = pd.DataFrame(np.round(sim_matrix, 4), index=status_ids, columns=status_ids)

    n = len(status_ids)
    # Struktur detail yang sederhana & akademis (kata unik & intersection)
    calculation_details = []
    for i in range(n):
        for j in range(i + 1, n):
            tokens_a = set(preprocessed[i]["tokens"])
            tokens_b = set(preprocessed[j]["tokens"])
            count_a = len(tokens_a)
            count_b = len(tokens_b)
            shared_words = sorted(list(tokens_a.intersection(tokens_b)))
            shared_count = len(shared_words)
            sim_val = float(sim_matrix[i][j])

            calculation_details.append({
                "id_a": status_ids[i],
                "id_b": status_ids[j],
                "count_a": count_a,
                "count_b": count_b,
                "shared_words": shared_words,
                "shared_count": shared_count,
                "similarity": sim_val,
            })

    all_similarities = [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    total_tokens = sum(len(r["tokens"]) for r in preprocessed)

    summary = {
        "jumlah_status": len(status_ids),
        "jumlah_vocabulary": len(vocabulary),
        "jumlah_token": total_tokens,
        "similarity_tertinggi": round(max(all_similarities), 4) if all_similarities else 0.0,
        "similarity_terendah": round(min(all_similarities), 4) if all_similarities else 0.0,
    }

    return {
        "status_ids": status_ids,
        "preprocessed": preprocessed,
        "vocabulary": vocabulary,
        "binary_matrix": binary_matrix,
        "sim_matrix": sim_matrix,
        "pairs_df": pairs_df,
        "prep_df": prep_df,
        "binary_df": binary_df,
        "sim_df": sim_df,
        "calculation_details": calculation_details,
        "summary": summary,
    }


# ─────────────────────────────────────────────
# TAB UTAMA
# ─────────────────────────────────────────────
tab_input, tab_prep, tab_binary, tab_sim, tab_detail = st.tabs([
    "Input",
    "Preprocessing",
    "Binary",
    "Similarity",
    "Detail",
])


# ══════════════════════════════════════════════
# TAB 1 — INPUT
# ══════════════════════════════════════════════
with tab_input:
    st.markdown('<div class="section-title">Sumber Data</div>', unsafe_allow_html=True)

    # Pilihan mode input
    col_mode1, col_mode2, col_mode_sp = st.columns([1.5, 1.8, 5])
    with col_mode1:
        if st.button(
            "✏️ Input Manual",
            use_container_width=True,
            type="primary" if st.session_state.input_mode == "manual" else "secondary",
        ):
            if st.session_state.input_mode != "manual":
                st.session_state.input_mode = "manual"
                st.session_state.analysis_done = False
                st.rerun()
    with col_mode2:
        if st.button(
            "📂 Upload Dokumen",
            use_container_width=True,
            type="primary" if st.session_state.input_mode == "upload" else "secondary",
        ):
            if st.session_state.input_mode != "upload":
                st.session_state.input_mode = "upload"
                st.session_state.analysis_done = False
                st.rerun()

    st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

    # ── MODE 1: INPUT MANUAL ──
    if st.session_state.input_mode == "manual":
        col_t1, col_t2, col_t_sp = st.columns([1.5, 2.2, 4.3])
        with col_t1:
            if st.button("➕ Tambah Status", use_container_width=True):
                st.session_state.texts.append("")
                st.session_state.analysis_done = False
                st.rerun()
        with col_t2:
            if st.button("📄 Gunakan Contoh Data", use_container_width=True, type="secondary"):
                st.session_state.texts = [
                    "Pelayanan sangat cepat dan petugas ramah",
                    "Pelayanan cepat dan petugas sangat ramah",
                    "Proses pelayanan mudah dan cepat",
                    "Petugas memberikan pelayanan dengan baik",
                    "Informasi pelayanan kurang jelas dan lambat",
                ]
                st.session_state.analysis_done = False
                st.rerun()

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

        texts_to_remove = []
        for idx in range(len(st.session_state.texts)):
            with st.container():
                st.markdown(f'<div class="status-badge">S{idx + 1}</div>', unsafe_allow_html=True)
                col_inp, col_del = st.columns([11, 1])
                with col_inp:
                    st.session_state.texts[idx] = st.text_area(
                        label=f"Status S{idx+1}",
                        value=st.session_state.texts[idx],
                        placeholder=f"Masukkan teks status S{idx+1}...",
                        key=f"text_area_{idx}",
                        height=68,
                        label_visibility="collapsed",
                    )
                with col_del:
                    st.markdown("<div style='height:0.25rem;'></div>", unsafe_allow_html=True)
                    if len(st.session_state.texts) > 2:
                        if st.button("🗑", key=f"del_{idx}", help=f"Hapus S{idx+1}", use_container_width=True):
                            texts_to_remove.append(idx)
                    else:
                        st.markdown("<div style='color:#cbd5e1; text-align:center; padding-top:0.4rem;'>✕</div>", unsafe_allow_html=True)

        if texts_to_remove:
            for idx in sorted(texts_to_remove, reverse=True):
                st.session_state.texts.pop(idx)
            st.session_state.analysis_done = False
            st.rerun()

        valid_count = sum(1 for t in st.session_state.texts if t.strip())
        if valid_count == 0:
            st.markdown('<div class="info-box-blue">💡 Masukkan teks status di atas atau klik <strong>Gunakan Contoh Data</strong>.</div>', unsafe_allow_html=True)
        elif valid_count == 1:
            st.markdown('<div class="warn-box">⚠️ Minimal diperlukan <strong>2 status</strong> yang tidak kosong untuk analisis.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="info-box">✅ <strong>{valid_count} status</strong> siap untuk dianalisis.</div>', unsafe_allow_html=True)

        texts_for_analysis = st.session_state.texts
        can_analyze = (valid_count >= 2)

    # ── MODE 2: UPLOAD DOKUMEN ──
    else:
        st.markdown("""
        <div class="info-box-blue">
        Upload satu atau beberapa dokumen dalam format <strong>PDF</strong>, <strong>DOCX</strong>, atau <strong>TXT</strong>.
        Teks akan diekstrak secara otomatis untuk kemudian dianalisis kemiripannya.
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload file dokumen",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.markdown("**Metode Pemisahan Teks:**")
            split_opt = st.radio(
                "Pemisahan teks",
                options=["Satu file = satu teks", "Satu baris = satu teks"],
                index=0 if st.session_state.split_mode == "file" else 1,
                horizontal=True,
                label_visibility="collapsed",
            )
            mode_choice = "file" if "file" in split_opt else "line"
            if mode_choice != st.session_state.split_mode:
                st.session_state.split_mode = mode_choice
                st.session_state.file_results = []
                st.session_state.analysis_done = False

            st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)
            if st.button("📥 Ekstrak Teks dari File", type="secondary"):
                with st.spinner("Mengekstrak teks..."):
                    st.session_state.file_results = process_uploaded_files(
                        uploaded_files, split_mode=st.session_state.split_mode
                    )
                    st.session_state.analysis_done = False

            if st.session_state.file_results:
                st.markdown('<div class="section-title">Hasil Ekstraksi Dokumen</div>', unsafe_allow_html=True)

                preview_data = []
                for fr in st.session_state.file_results:
                    preview_data.append({
                        "File": fr["filename"],
                        "Karakter": fr["char_count"] or "—",
                        "Baris": fr["line_count"] or "—",
                        "Unit Teks": f"{len(fr['units'])} teks" if fr["units"] else "—",
                        "Preview Teks": fr["preview"] or "(kosong)",
                    })
                st.dataframe(pd.DataFrame(preview_data), use_container_width=True, hide_index=True)

                for fr in st.session_state.file_results:
                    if fr["warning"]:
                        st.markdown(f'<div class="warn-box">⚠️ <strong>{fr["filename"]}</strong>: {fr["warning"]}</div>', unsafe_allow_html=True)

                all_units = collect_all_units(st.session_state.file_results)
                valid_units = [u for u in all_units if u.strip()]

                if len(valid_units) < 2:
                    st.markdown('<div class="warn-box">⚠️ Tidak cukup data. Minimal diperlukan <strong>2 teks</strong> untuk menghitung similarity.</div>', unsafe_allow_html=True)
                    can_analyze = False
                else:
                    st.markdown(f'<div class="info-box">✅ <strong>{len(valid_units)} unit teks</strong> berhasil diekstrak dan siap dianalisis.</div>', unsafe_allow_html=True)
                    can_analyze = True

                texts_for_analysis = valid_units

                # Tombol download extracted text
                if valid_units:
                    ext_rows = []
                    uid = 1
                    for fr in st.session_state.file_results:
                        for unit in fr["units"]:
                            if unit.strip():
                                ext_rows.append({"File": fr["filename"], "Text ID": f"S{uid}", "Extracted Text": unit})
                                uid += 1
                    st.download_button(
                        label="⬇️ Download Extracted Text (CSV)",
                        data=df_to_csv(pd.DataFrame(ext_rows)),
                        file_name="extracted_texts.csv",
                        mime="text/csv",
                    )
            else:
                texts_for_analysis = []
                can_analyze = False
        else:
            texts_for_analysis = []
            can_analyze = False

    # ── TOMBOL ANALISIS UTAMA (LEBIH MENONJOL) ──
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    col_btn, _ = st.columns([2.5, 4.5])
    with col_btn:
        analyze_clicked = st.button(
            "🔍 Analisis Kemiripan",
            type="primary",
            use_container_width=True,
            disabled=not can_analyze,
        )

    if analyze_clicked:
        with st.spinner("Sedang memproses analisis kemiripan..."):
            results = run_analysis(texts_for_analysis)
            st.session_state.results = results
            st.session_state.analysis_done = True
        st.success(f"Analisis selesai! {results['summary']['jumlah_status']} status berhasil dianalisis.")

    # ── RINGKASAN METRIK (SETELAH ANALISIS) ──
    if st.session_state.analysis_done:
        results = st.session_state.results
        summary = results["summary"]
        st.markdown('<div class="section-title">Ringkasan Hasil Analisis</div>', unsafe_allow_html=True)

        m_cols = st.columns(5)
        metrics = [
            (summary["jumlah_status"], "Jumlah Status"),
            (summary["jumlah_vocabulary"], "Jumlah Vocabulary"),
            (summary["jumlah_token"], "Total Token"),
            (f"{summary['similarity_tertinggi']:.4f}", "Similarity Tertinggi"),
            (f"{summary['similarity_terendah']:.4f}", "Similarity Terendah"),
        ]
        for col, (value, label) in zip(m_cols, metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="mv">{value}</div>
                    <div class="ml">{label}</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — PREPROCESSING
# ══════════════════════════════════════════════
with tab_prep:
    st.markdown('<div class="section-title">Tahap Preprocessing</div>', unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        empty_state("⚙️", "Silakan masukkan data dan klik <strong>🔍 Analisis Kemiripan</strong> pada tab Input.")
    else:
        results = st.session_state.results
        prep_df = results["prep_df"]

        st.markdown("""
        <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.75rem; align-items:center;">
            <div class="step-badge">1. Case Folding</div>
            <span style="color:#94a3b8; font-size:0.8rem;">→</span>
            <div class="step-badge">2. Hapus Tanda Baca</div>
            <span style="color:#94a3b8; font-size:0.8rem;">→</span>
            <div class="step-badge">3. Tokenisasi</div>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(prep_df, use_container_width=True, hide_index=True)

        with st.expander("📖 Keterangan Preprocessing"):
            st.markdown("""
            - **Case Folding**: Mengubah seluruh huruf menjadi huruf kecil.
            - **Hapus Tanda Baca**: Menghapus seluruh karakter tanda baca (*punctuation*).
            - **Tokenisasi**: Memecah kalimat menjadi daftar kata unik (*tokens*).
            """)

        st.download_button(
            "⬇️ Download Preprocessing (CSV)",
            data=df_to_csv(prep_df),
            file_name="preprocessing_result.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════
# TAB 3 — BINARY REPRESENTATION
# ══════════════════════════════════════════════
with tab_binary:
    st.markdown('<div class="section-title">Binary Representation (0/1)</div>', unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        empty_state("🔢", "Silakan masukkan data dan jalankan analisis terlebih dahulu pada tab Input.")
    else:
        results = st.session_state.results
        binary_df = results["binary_df"]
        vocabulary = results["vocabulary"]

        st.markdown(f"""
        <div class="info-box-blue">
        Membentuk representasi biner dari seluruh kata unik (<strong>{len(vocabulary)} kata</strong>):<br>
        <strong>1</strong> jika kata muncul pada status &nbsp;|&nbsp; <strong>0</strong> jika kata tidak muncul.
        </div>
        """, unsafe_allow_html=True)

        display_binary = binary_df.copy()
        word_cols = [c for c in display_binary.columns if c != "Status"]
        display_binary[word_cols] = display_binary[word_cols].astype(int)

        st.dataframe(display_binary, use_container_width=True, hide_index=True)

        with st.expander(f"📚 Daftar Vocabulary ({len(vocabulary)} kata)"):
            cols = st.columns(4)
            for i, w in enumerate(vocabulary):
                with cols[i % 4]:
                    st.markdown(f"- `{w}`")

        st.download_button(
            "⬇️ Download Binary Representation (CSV)",
            data=df_to_csv(display_binary),
            file_name="binary_representation.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════
# TAB 4 — SIMILARITY
# ══════════════════════════════════════════════
with tab_sim:
    st.markdown('<div class="section-title">Cosine Similarity</div>', unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        empty_state("📊", "Silakan masukkan data dan jalankan analisis terlebih dahulu pada tab Input.")
    else:
        results = st.session_state.results
        sim_df = results["sim_df"]
        pairs_df = results["pairs_df"]

        st.markdown("#### Matriks Cosine Similarity")
        st.markdown("""
        <div class="info-box-blue">
        Matriks bersifat simetris. Nilai diagonal selalu bernilai <strong>1.0000</strong> (identik dengan dirinya sendiri).
        </div>
        """, unsafe_allow_html=True)

        # Menggunakan .map() (kompatibel pandas terbaru)
        styled_sim = sim_df.style.map(color_sim_cell).format("{:.4f}")
        st.dataframe(styled_sim, use_container_width=True)

        # Legenda Kategori
        st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)
        leg_cols = st.columns(5)
        for col, (label, rng, bg, fg) in zip(leg_cols, [
            ("Sangat Tinggi", "0.80–1.00", "#bfdbfe", "#1e3a8a"),
            ("Tinggi", "0.60–0.79", "#dbeafe", "#1e40af"),
            ("Sedang", "0.40–0.59", "#eff6ff", "#1d4ed8"),
            ("Rendah", "0.20–0.39", "#f8fafc", "#475569"),
            ("Sangat Rendah", "0.00–0.19", "#ffffff", "#94a3b8"),
        ]):
            with col:
                st.markdown(
                    f"<div style='background:{bg}; color:{fg}; padding:4px 6px; border-radius:4px; "
                    f"border:1px solid #cbd5e1; font-size:0.75rem; text-align:center; font-weight:600;'>"
                    f"{label}<br><span style='font-weight:400;'>{rng}</span></div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")

        st.markdown("#### Tabel Pasangan Similarity")
        if not pairs_df.empty:
            styled_pairs = (
                pairs_df.style
                .map(color_interp_cell, subset=["Interpretasi"])
                .format({"Cosine Similarity": "{:.4f}"})
                .set_properties(subset=["Cosine Similarity"], **{"text-align": "center", "font-weight": "700"})
                .set_properties(subset=["Status 1", "Status 2"], **{"text-align": "center", "font-weight": "600"})
            )
            st.dataframe(styled_pairs, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Download Hasil")
        dcols = st.columns(4)
        with dcols[0]:
            st.download_button("⬇️ Preprocessing (CSV)", df_to_csv(results["prep_df"]), "preprocessing_result.csv", "text/csv", use_container_width=True)
        with dcols[1]:
            st.download_button("⬇️ Binary Repr. (CSV)", df_to_csv(results["binary_df"]), "binary_representation.csv", "text/csv", use_container_width=True)
        with dcols[2]:
            st.download_button("⬇️ Similarity Matrix (CSV)", matrix_to_csv(sim_df), "similarity_matrix.csv", "text/csv", use_container_width=True)
        with dcols[3]:
            st.download_button("⬇️ Pasangan Sim. (CSV)", df_to_csv(pairs_df), "similarity_pairs.csv", "text/csv", use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5 — DETAIL PERHITUNGAN
# (SEDERHANA, AKADEMIS, SESUAI CONTOH TUGAS)
# ══════════════════════════════════════════════
with tab_detail:
    st.markdown('<div class="section-title">Detail Perhitungan Cosine Similarity</div>', unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        empty_state("🧮", "Silakan masukkan data dan jalankan analisis terlebih dahulu pada tab Input.")
    else:
        results = st.session_state.results
        details = results["calculation_details"]

        st.markdown("""
        <div class="info-box-blue">
        Perhitungan Cosine Similarity untuk setiap pasangan status disajikan secara bertahap:<br>
        <strong>1. Jumlah kata pada masing-masing status</strong> &nbsp;→&nbsp;
        <strong>2. Kata yang sama</strong> &nbsp;→&nbsp;
        <strong>3. Substitusi rumus Cosine Similarity</strong> &nbsp;→&nbsp;
        <strong>4. Hasil akhir</strong>
        </div>
        """, unsafe_allow_html=True)

        for d in details:
            # Format similarity: 1.0000 jika identik, atau 4 desimal
            if abs(d["similarity"] - 1.0) < 1e-9:
                sim_str = "1.0000"
            elif d["similarity"] == 0.0:
                sim_str = "0"
            else:
                sim_str = f"{d['similarity']:.4f}"

            exp_label = f"{d['id_a']} vs {d['id_b']}  —  {sim_str}"

            with st.expander(exp_label):
                col_left, col_right = st.columns([1, 1.2])

                with col_left:
                    st.markdown(f"**Jumlah kata {d['id_a']} = {d['count_a']}**")
                    st.markdown(f"**Jumlah kata {d['id_b']} = {d['count_b']}**")
                    st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)

                    if d["shared_count"] > 0:
                        words_text = ", ".join(d["shared_words"])
                        st.markdown(f"**Kata yang sama ({d['shared_count']}):**")
                        st.markdown(f"<span style='color:#1e40af; font-weight:500;'>{words_text}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("**Kata yang sama (0):**")
                        st.markdown("<span style='color:#64748b;'>Tidak ada kata yang sama.</span>", unsafe_allow_html=True)

                with col_right:
                    st.markdown(f"**Cos({d['id_a']},{d['id_b']})**")

                    if d["shared_count"] == 0 or d["count_a"] == 0 or d["count_b"] == 0:
                        calc_text = f"Cos({d['id_a']},{d['id_b']}) = 0"
                    elif d["count_a"] == d["count_b"] == d["shared_count"]:
                        # Jika identik
                        sc = d["shared_count"]
                        calc_text = (
f"""       {sc}
──────────────
 √{sc} × √{sc}

= {sim_str}"""
                        )
                    else:
                        calc_text = (
f"""       {d['shared_count']}
──────────────
 √{d['count_a']} × √{d['count_b']}

= {sim_str}"""
                        )

                    st.markdown(f"""
                    <div class="calc-formula-box">{calc_text}</div>
                    """, unsafe_allow_html=True)
