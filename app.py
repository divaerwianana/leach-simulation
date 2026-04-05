import streamlit as st

# import semua modul
import abc_dinamis
import abc_statis
import randomsearch_dinamis
import randomsearch_statis

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Clustering Jaringan",
    page_icon="📡",
    layout="wide"
)

# =============================
# NAVBAR
# =============================
menu = st.sidebar.radio(
    "Pilih Metode",
    [
        "Home",
        "ABC Dinamis",
        "ABC Statis",
        "Random Dinamis",
        "Random Statis"
    ]
)

# =============================
# ROUTING
# =============================
if menu == "Home":
    st.title("📡 Simulasi Clustering Jaringan")

    st.markdown("""
    ### Penerapan Algoritma Artificial Bee Colony untuk Pemilihan Cluster Head Adaptif  
    pada Protokol LEACH di Jaringan MANET
    """)

    st.markdown("---")

    st.write("""
    Aplikasi ini digunakan untuk melakukan simulasi dan analisis performa algoritma clustering 
    pada jaringan berbasis Mobile Ad Hoc Network (MANET).

    Fokus utama penelitian ini adalah penerapan algoritma **Artificial Bee Colony (ABC)** 
    dalam proses pemilihan **Cluster Head (CH)** secara adaptif pada protokol **LEACH**, 
    serta membandingkannya dengan metode **Random Search**.
    """)

    st.markdown("### 🔍 Metode yang Tersedia")
    st.markdown("""
    - **ABC Dinamis** → Node bergerak (mobilitas jaringan)  
    - **ABC Statis** → Node tetap  
    - **Random Search Dinamis** → Metode pembanding dengan mobilitas  
    - **Random Search Statis** → Metode pembanding tanpa mobilitas  
    """)

    st.markdown("### ⚙️ Parameter Pengujian")
    st.markdown("""
    - Jumlah Node  
    - Luas Area Jaringan  
    - Energi Awal Node  
    - Letak Base Station  
    """)

    st.markdown("### 📊 Metrik Evaluasi")
    st.markdown("""
    - **FND (First Node Dies)** → Waktu ketika node pertama mati  
    - **HND (Half Node Dies)** → Waktu ketika 50% node mati  
    - **LND (Last Node Dies)** → Waktu ketika seluruh node mati  
    - **PDR (Packet Delivery Ratio)** → Rasio paket berhasil diterima  
    - **Total Energy** → Sisa energi dalam jaringan  
    - **Packet Loss** → Paket yang gagal dikirim  
    """)

    st.markdown("---")

    st.info("Gunakan menu di sidebar untuk memilih metode simulasi.")
    
# =============================
# HALAMAN LAIN
# =============================
elif menu == "ABC Dinamis":
    abc_dinamis.run()

elif menu == "ABC Statis":
    abc_statis.run()

elif menu == "Random Dinamis":
    randomsearch_dinamis.run()

elif menu == "Random Statis":
    randomsearch_statis.run()
