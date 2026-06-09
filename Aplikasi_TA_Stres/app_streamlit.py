# =====================================================================
# APLIKASI KLASIFIKASI TINGKAT STRES MAHASISWA — STREAMLIT
# Berbasis PSS-10 | Universitas Krisnadwipayana Jakarta
# =====================================================================

import streamlit as st
import joblib
import numpy as np
import os

# =====================================================================
# KONFIGURASI HALAMAN
# =====================================================================
st.set_page_config(
    page_title="Deteksi Stres Mahasiswa UNKRIS",
    page_icon="🧠",
    layout="centered"
)

# =====================================================================
# LOAD MODEL & SCALER
# =====================================================================
@st.cache_resource
def load_model_dan_scaler():
    """
    Load model Random Forest + scaler.
    Pastikan file berikut ada di folder yang sama dengan app.py:
      - model_random_forest.pkl
      - scaler.pkl
    """
    try:
        model  = joblib.load('model_random_forest.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler, None
    except FileNotFoundError as e:
        return None, None, str(e)

model, scaler, error_msg = load_model_dan_scaler()

# =====================================================================
# HEADER APLIKASI
# =====================================================================
st.title("🧠 Aplikasi Klasifikasi Tingkat Stres Mahasiswa")
st.markdown("**Universitas Krisnadwipayana Jakarta**")
st.write("Isi kuesioner PSS-10 di bawah ini sesuai kondisi yang Anda rasakan **dalam satu bulan terakhir**.")

# Tampilkan error jika model tidak ditemukan
if error_msg:
    st.error(f"❌ File model tidak ditemukan: `{error_msg}`")
    st.info("Pastikan file `model_random_forest.pkl` dan `scaler.pkl` ada di folder yang sama dengan `app.py`.")
    st.stop()

st.write("---")

# =====================================================================
# PANDUAN SKALA
# =====================================================================
with st.expander("📖 Panduan Pengisian Skala"):
    st.markdown("""
    | Nilai | Keterangan |
    |-------|------------|
    | 0     | Tidak Pernah |
    | 1     | Hampir Tidak Pernah |
    | 2     | Kadang-kadang |
    | 3     | Cukup Sering |
    | 4     | Sangat Sering |
    """)

# =====================================================================
# FORM INPUT PSS-10
# =====================================================================
pertanyaan = {
    "PSS1" : "1. Dalam sebulan terakhir, seberapa sering Anda merasa kecewa karena sesuatu yang terjadi secara tidak terduga?",
    "PSS2" : "2. Dalam sebulan terakhir, seberapa sering Anda merasa tidak mampu mengendalikan hal-hal penting dalam hidup Anda?",
    "PSS3" : "3. Dalam sebulan terakhir, seberapa sering Anda merasa gelisah dan stres?",
    "PSS4" : "4. Dalam sebulan terakhir, seberapa sering Anda merasa percaya diri dengan kemampuan Anda untuk menyelesaikan masalah pribadi Anda?",
    "PSS5" : "5. Dalam sebulan terakhir, seberapa sering Anda merasa segala sesuatu berjalan sesuai keinginan Anda?",
    "PSS6" : "6. Dalam sebulan terakhir, seberapa sering Anda mengetahui bahwa Anda tidak bisa mengatasi hal-hal yang harus Anda lakukan?",
    "PSS7" : "7. Dalam sebulan terakhir, seberapa sering Anda mampu mengendalikan hal-hal yang menjengkelkan dalam hidup Anda?",
    "PSS8" : "8. Dalam sebulan terakhir, seberapa sering Anda merasa mampu mengendalikan permasalahan Anda?",
    "PSS9" : "9. Dalam sebulan terakhir, seberapa sering Anda marah karena hal-hal yang terjadi di luar kendali Anda?",
    "PSS10": "10. Dalam sebulan terakhir, seberapa sering Anda merasa tidak mampu menyelesaikan permasalahan-permasalahan yang menumpuk dalam hidup Anda?",
}

label_skala = {0: "0 – Tidak Pernah", 1: "1 – Hampir Tidak Pernah",
               2: "2 – Kadang-kadang", 3: "3 – Cukup Sering", 4: "4 – Sangat Sering"}

jawaban = {}
for kode, teks in pertanyaan.items():
    st.markdown(f"**{teks}**")
    jawaban[kode] = st.select_slider(
        label=kode,
        options=[0, 1, 2, 3, 4],
        value=2,
        format_func=lambda x: label_skala[x],
        label_visibility="collapsed"
    )
    st.write("")

st.write("---")

# =====================================================================
# TOMBOL PREDIKSI
# =====================================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    tombol = st.button("🔍 Cek Tingkat Stres Saya", type="primary", use_container_width=True)

if tombol:
    # --- Kumpulkan input ---
    input_raw = np.array([[jawaban[f"PSS{i}"] for i in range(1, 11)]])

    # --- Reverse scoring SEBELUM scaling ---
    # Item positif: PSS4(idx=3), PSS5(idx=4), PSS7(idx=6), PSS8(idx=7)
    input_reversed = input_raw.copy().astype(float)
    for idx in [3, 4, 6, 7]:
        input_reversed[0][idx] = 4 - input_reversed[0][idx]

    # --- Hitung skor manual untuk verifikasi ---
    total_skor = int(input_reversed[0].sum())

    # --- Normalisasi dengan scaler yang sama dari Colab ---
    input_scaled = scaler.transform(input_reversed)

    # --- Prediksi ---
    prediksi    = model.predict(input_scaled)[0]
    probabilitas = model.predict_proba(input_scaled)[0]

    # =====================================================================
    # TAMPILKAN HASIL
    # =====================================================================
    st.subheader("📊 Hasil Analisis Tingkat Stres")

    # Skor PSS-10
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Total Skor PSS-10", f"{total_skor} / 40")
    with col_b:
        kategori_skor = (
            "Rendah (0–13)" if total_skor <= 13
            else "Sedang (14–26)" if total_skor <= 26
            else "Tinggi (27–40)"
        )
        st.metric("Kategori Skor", kategori_skor)

    st.write("")

    # Hasil klasifikasi model
    if prediksi == 0:
        st.success("### ✅ Tingkat Stres Anda: RENDAH")
        st.info("""
        **Interpretasi:** Anda berada dalam kondisi psikologis yang baik.
        Tekanan yang Anda rasakan masih dalam batas yang wajar dan dapat dikelola.

        **Saran:**
        - Pertahankan kebiasaan positif yang sudah Anda lakukan
        - Tetap luangkan waktu untuk relaksasi di sela aktivitas perkuliahan
        - Jaga pola tidur dan olahraga rutin
        """)

    elif prediksi == 1:
        st.warning("### ⚠️ Tingkat Stres Anda: SEDANG")
        st.info("""
        **Interpretasi:** Anda mengalami tekanan yang cukup berarti dan perlu diperhatikan
        sebelum berkembang menjadi stres berat.

        **Saran:**
        - Kelola waktu pengerjaan skripsi dengan jadwal harian yang realistis
        - Ceritakan beban yang Anda rasakan kepada teman dekat atau keluarga
        - Batasi konsumsi kafein dan perbaiki kualitas tidur
        - Manfaatkan sesi bimbingan dengan dosen pembimbing secara rutin
        """)

    else:
        st.error("### 🚨 Tingkat Stres Anda: TINGGI")
        st.info("""
        **Interpretasi:** Anda mengalami tekanan yang signifikan dan memerlukan perhatian segera.

        **Saran:**
        - Segera hubungi layanan konseling atau psikolog kampus UNKRIS
        - Bicarakan kondisi Anda dengan dosen pembimbing atau dosen wali
        - Prioritaskan istirahat — produktivitas tidak akan optimal dalam kondisi stres tinggi
        - Jangan ragu untuk meminta bantuan; mencari bantuan adalah tanda keberanian
        """)

    # Probabilitas per kelas
    st.write("")
    st.markdown("**Distribusi Probabilitas Prediksi Model:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rendah", f"{probabilitas[0]*100:.1f}%")
    with col2:
        st.metric("Sedang", f"{probabilitas[1]*100:.1f}%")
    with col3:
        st.metric("Tinggi", f"{probabilitas[2]*100:.1f}%")

    # Catatan metodologis
    st.write("")
    with st.expander("ℹ️ Catatan Metodologis"):
        st.markdown(f"""
        - Instrumen yang digunakan: **PSS-10** (Cohen et al., 1983)
        - Adaptasi Bahasa Indonesia: **Hakim et al. (2024)**
        - Algoritma klasifikasi: **Random Forest** (model terbaik dari hasil perbandingan)
        - Reverse scoring diterapkan pada item: PSS4, PSS5, PSS7, PSS8
        - Input dinormalisasi menggunakan **Min-Max Scaler** yang sama dengan proses pelatihan
        - Aplikasi ini merupakan **Proof of Concept** — bukan pengganti diagnosis klinis profesional
        """)

# =====================================================================
# FOOTER
# =====================================================================
st.write("---")
st.caption("Aplikasi ini dikembangkan sebagai bagian dari penelitian skripsi | "
           "Ambrosia Woga Daso | Universitas Krisnadwipayana Jakarta | 2026")