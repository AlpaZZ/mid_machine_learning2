import sys
import os

# Hindari tabrakan nama (name collision) global mlflow dengan berkas mlflow.py lokal
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir in sys.path:
    sys.path.remove(current_dir)
while "" in sys.path:
    sys.path.remove("")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

# 1. Konfigurasi Halaman & Tema Premium (Cyber Security Dark Mode)
st.set_page_config(
    page_title="IoT-Guard: Vulnerability Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk Glassmorphism & Tampilan Premium State-of-the-Art
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Font & Background overrides */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #070a13 0%, #0f172a 100%);
        color: #f1f5f9;
    }
    
    /* Sidebar Glassmorphism */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 15, 30, 0.95) !important;
        border-right: 1px solid rgba(59, 130, 246, 0.15) !important;
        backdrop-filter: blur(15px);
    }
    
    /* Neon Glowing Cards/Containers */
    .metric-card {
        background: linear-gradient(145deg, rgba(17, 24, 39, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6, #6366f1);
        opacity: 0.7;
    }
    
    .metric-card:hover {
        transform: translateY(-6px);
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.2), 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    
    .metric-val-success {
        font-size: 36px;
        font-weight: 700;
        color: #10b981;
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.3);
    }
    
    .metric-val-danger {
        font-size: 36px;
        font-weight: 700;
        color: #ef4444;
        text-shadow: 0 0 10px rgba(239, 68, 68, 0.3);
    }
    
    .metric-val-neutral {
        font-size: 36px;
        font-weight: 700;
        color: #3b82f6;
        text-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
    }
    
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 500;
    }
    
    /* Cyber Threat Alerts */
    .status-banner {
        padding: 20px 30px;
        border-radius: 12px;
        font-weight: 600;
        margin-bottom: 30px;
        display: flex;
        align-items: center;
        gap: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .status-secure {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10b981;
    }
    
    .status-danger {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #ef4444;
    }
    
    /* Threat Level Gauge indicator styles */
    .gauge-container {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton>button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.6) !important;
    }
</style>
""", unsafe_allow_html=True)

# Set global Matplotlib dark-theme parameters untuk visualisasi professional
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['text.color'] = '#f1f5f9'
plt.rcParams['axes.labelcolor'] = '#f1f5f9'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'
plt.rcParams['grid.color'] = '#1e293b'

# 2. Cache Model & Label Encoder
@st.cache_resource
def load_assets():
    model_path = 'pipeline_terbaik.pkl'
    le_path = 'label_encoder.pkl'
    
    if os.path.exists(model_path) and os.path.exists(le_path):
        pipeline = joblib.load(model_path)
        le = joblib.load(le_path)
        return pipeline, le
    return None, None

pipeline, le = load_assets()

# Cache untuk Loading Dataset (mengambil slice 10.000 baris untuk data explorer)
@st.cache_data
def load_dataset_slice():
    dataset_path = "Preprocessed_Balanced_dataset.csv"
    fallback_path = "sample_iot_data.csv"
    if os.path.exists(dataset_path):
        try:
            return pd.read_csv(dataset_path, nrows=10000)
        except:
            pass
    if os.path.exists(fallback_path):
        try:
            return pd.read_csv(fallback_path)
        except:
            return None
    return None

df_slice = load_dataset_slice()

# 3. Sidebar Navigation & Layout Setup
st.sidebar.markdown("<div style='text-align: center; padding-top: 10px;'><h1 style='font-size: 28px; color: #3b82f6; font-weight: 700; margin: 0; text-shadow: 0 0 10px rgba(59, 130, 246, 0.4);'>🛡️ IoT-Guard</h1><p style='color: #64748b; font-size: 13px; margin-top: 3px;'>Vulnerability Detection Suite</p></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.subheader("🧭 Navigasi Menu")
menu_mode = st.sidebar.radio(
    "Pilih Fitur Dashboard:",
    ["🛡️ Packet Scanner", "📊 Hasil Training & Performa", "📁 Dataset Explorer"]
)

st.sidebar.markdown("---")

# Generator file sampel dinamis dengan SLIDER untuk menentukan jumlah baris
if menu_mode == "🛡️ Packet Scanner":
    st.sidebar.subheader("💡 Generator Demo")
    st.sidebar.info("Tidak punya file CSV? Geser slider di bawah untuk menentukan jumlah baris demo, lalu unduh sampelnya!")
    
    # Slider dinamis
    sample_rows = st.sidebar.slider("Jumlah Baris Sampel:", 10, 150, 25, 5)

    def generate_sample_csv(nrows):
        dataset_path = "Preprocessed_Balanced_dataset.csv"
        fallback_path = "sample_iot_data.csv"
        if os.path.exists(dataset_path):
            try:
                df_full = pd.read_csv(dataset_path, nrows=10000)
                # Ambil setengah normal, setengah attack
                n_half = max(2, nrows // 2)
                df_normal = df_full[df_full['Label'] == 0].sample(n=n_half, random_state=42, replace=True)
                df_attack = df_full[df_full['Label'] == 1].sample(n=nrows - n_half, random_state=42, replace=True)
                df_sample = pd.concat([df_normal, df_attack]).sample(frac=1.0, random_state=42)
                
                # Buang kolom label/cheat
                kolom_contekan = ['Label', 'Attack_Category', 'Attack_sub_category']
                df_sample_clean = df_sample.drop(columns=[col for col in kolom_contekan if col in df_sample.columns])
                return df_sample_clean.to_csv(index=False).encode('utf-8')
            except Exception as e:
                pass
        if os.path.exists(fallback_path):
            try:
                df_full = pd.read_csv(fallback_path)
                n_half = max(2, nrows // 2)
                df_normal = df_full[df_full['Label'] == 0].sample(n=min(n_half, len(df_full[df_full['Label'] == 0])), random_state=42, replace=True)
                df_attack = df_full[df_full['Label'] == 1].sample(n=min(nrows - n_half, len(df_full[df_full['Label'] == 1])), random_state=42, replace=True)
                df_sample = pd.concat([df_normal, df_attack]).sample(frac=1.0, random_state=42)
                
                # Buang kolom label/cheat
                kolom_contekan = ['Label', 'Attack_Category', 'Attack_sub_category']
                df_sample_clean = df_sample.drop(columns=[col for col in kolom_contekan if col in df_sample.columns])
                return df_sample_clean.to_csv(index=False).encode('utf-8')
            except Exception as e:
                return None
        return None

    sample_data = generate_sample_csv(sample_rows)

    if sample_data is not None:
        st.sidebar.download_button(
            label=f"📥 Unduh {sample_rows} Baris Sampel Jaringan",
            data=sample_data,
            file_name=f"sampel_iot_lalu_lintas_{sample_rows}baris.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.sidebar.warning("Dataset lokal tidak ditemukan untuk membuat sampel demo.")

st.sidebar.markdown("<div style='text-align: center; color: #475569; font-size: 11px; margin-top: 50px;'>Mid ML II &copy; 2026<br>Kelompok 3</div>", unsafe_allow_html=True)

# 4. Main Panel
# ==============================================================================
# MENU 1: 🛡️ PACKET SCANNER (Deteksi Aktivitas Jaringan)
# ==============================================================================
if menu_mode == "🛡️ Packet Scanner":
    st.markdown("<div style='display: flex; align-items: center; gap: 15px; margin-bottom: 5px;'><h1 style='font-size: 38px; font-weight: 700; margin: 0; background: linear-gradient(90deg, #ffffff 0%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🛡️ IoT Vulnerability Detector Dashboard</h1></div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 16px; margin-bottom: 25px;'>Sistem pemantauan lalu lintas paket jaringan untuk mendeteksi celah kerentanan dan anomali jaringan IoT secara realtime tanpa kebocoran data.</p>", unsafe_allow_html=True)

    if pipeline is None or le is None:
        st.error("🚨 Waduh, model 'pipeline_terbaik.pkl' atau 'label_encoder.pkl' belum ditemukan! Silakan jalankan script tracking 'mlflow.py' terlebih dahulu untuk menghasilkan artifacts tersebut.")
    else:
        st.sidebar.subheader("📤 Scan Hub")
        uploaded_file = st.sidebar.file_uploader("Unggah CSV Data Aktivitas Jaringan", type=["csv"])

        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                kolom_contekan = ['Label', 'Attack_Category', 'Attack_sub_category']
                new_data_clean = new_data.drop(columns=[col for col in kolom_contekan if col in new_data.columns])
                
                st.markdown("### 📊 Proses Analisis Aktivitas")
                
                with st.expander("🔍 Pratinjau Struktur Fitur Masukan (5 Baris Pertama)", expanded=False):
                    st.dataframe(new_data_clean.head(5), use_container_width=True)
                    st.caption(f"Total fitur yang dideteksi: {new_data_clean.shape[1]} kolom.")
                
                if st.button("🚀 Lakukan Deteksi Kerentanan Jaringan", use_container_width=True):
                    with st.spinner("Sedang memproses paket jaringan menggunakan Pipeline ML... ⚡"):
                        predictions_encoded = pipeline.predict(new_data_clean)
                        predictions_label = le.inverse_transform(predictions_encoded)
                        
                        total_scanned = len(new_data_clean)
                        anomalies = np.sum(predictions_encoded == 1)
                        secure = total_scanned - anomalies
                        
                        # Hitung Persentase Ancaman & Level Resiko (Threat Level Indicator)
                        threat_percentage = (anomalies / total_scanned) * 100
                        if threat_percentage == 0:
                            risk_level = "🟢 SECURE (Low Risk)"
                            risk_color = "#10b981"
                        elif threat_percentage <= 10:
                            risk_level = "🟡 WARNING (Medium Risk)"
                            risk_color = "#fbbf24"
                        elif threat_percentage <= 30:
                            risk_level = "🟠 DANGER (High Risk)"
                            risk_color = "#f97316"
                        else:
                            risk_level = "🔴 CRITICAL (Threat Compromised!)"
                            risk_color = "#ef4444"
                        
                        # Status Banner
                        if anomalies > 0:
                            st.markdown(f"""
                            <div class="status-banner status-danger">
                                <span style="font-size: 28px; filter: drop-shadow(0 0 5px rgba(239,68,68,0.5));">🚨</span>
                                <div>
                                    <span style="font-size: 18px; font-weight: 700;">BAHAYA TERDETEKSI!</span><br>
                                    <span style="font-weight: 400; font-size: 14px;">Ditemukan {anomalies} paket jaringan terindikasi anomali/serangan IoT. Segera periksa detail log di bawah ini!</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="status-banner status-secure">
                                <span style="font-size: 28px; filter: drop-shadow(0 0 5px rgba(16,185,129,0.5));">🛡️</span>
                                <div>
                                    <span style="font-size: 18px; font-weight: 700;">JARINGAN AMAN & KONDUSIF</span><br>
                                    <span style="font-weight: 400; font-size: 14px;">Seluruh paket jaringan yang di-scan tergolong normal. Tidak ada aktivitas berbahaya yang terdeteksi.</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Custom Visual Gauge untuk Tingkat Ancaman Jaringan
                        st.markdown(f"""
                        <div class="gauge-container">
                            <span style="font-size: 15px; color: #94a3b8; font-weight: 500;">TINGKAT RISIKO JARINGAN SAAT INI:</span>
                            <span style="font-size: 18px; font-weight: 700; color: {risk_color}; text-shadow: 0 0 8px {risk_color}a0;">{risk_level} ({threat_percentage:.1f}% Ancaman)</span>
                        </div>
                        """, unsafe_allow_html=True)

                        # Baris Kartu Metrik
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Data Di-Scan</div><div class='metric-val-neutral'>{total_scanned:,}</div><div style='color: #64748b; font-size: 12px; margin-top: 5px;'>Paket Lintas Data</div></div>", unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"<div class='metric-card'><div class='metric-label'>Jaringan Aman (Normal)</div><div class='metric-val-success'>{secure:,}</div><div style='color: #10b981; font-size: 12px; margin-top: 5px;'>({(secure/total_scanned)*100:.2f}%) Bebas Ancaman</div></div>", unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"<div class='metric-card'><div class='metric-label'>Anomali Terdeteksi</div><div class='metric-val-danger'>{anomalies:,}</div><div style='color: #ef4444; font-size: 12px; margin-top: 5px;'>({(anomalies/total_scanned)*100:.2f}%) Butuh Tindakan</div></div>", unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Layout Visualisasi & Log Tabel
                        chart_col, table_col = st.columns([2, 3])
                        
                        with chart_col:
                            st.markdown("#### 📈 Distribusi Klasifikasi")
                            fig, ax = plt.subplots(figsize=(6, 5))
                            labels = ['Normal', 'Anomali']
                            sizes = [secure, anomalies]
                            colors = ['#10b981', '#ef4444']
                            explode = (0, 0.1) if anomalies > 0 else (0, 0)
                            
                            wedges, texts, autotexts = ax.pie(
                                sizes, explode=explode, labels=labels, colors=colors,
                                autopct='%1.1f%%', startangle=140,
                                textprops=dict(color="#e2e8f0", weight="bold"),
                                wedgeprops=dict(width=0.4, edgecolor='#070a13', linewidth=2)
                            )
                            for autotext in autotexts:
                                autotext.set_color('#ffffff')
                                autotext.set_fontsize(10)
                            ax.axis('equal')
                            plt.tight_layout()
                            st.pyplot(fig, clear_figure=True)
                        
                        with table_col:
                            st.markdown("#### 📋 Log Prediksi Lengkap")
                            result_df = new_data_clean.copy()
                            result_df['HASIL_DETEKSI'] = predictions_label
                            result_df['HASIL_DETEKSI'] = result_df['HASIL_DETEKSI'].apply(
                                lambda x: "🟢 NORMAL" if x == 0 else "🔴 ANOMALI / SERANGAN"
                            )
                            
                            filter_opt = st.selectbox("Saring Log Berdasarkan Hasil:", ["Semua Data", "Hanya Anomali / Serangan", "Hanya Normal"])
                            if filter_opt == "Hanya Anomali / Serangan":
                                filtered_df = result_df[result_df['HASIL_DETEKSI'].str.contains("ANOMALI")]
                            elif filter_opt == "Hanya Normal":
                                filtered_df = result_df[result_df['HASIL_DETEKSI'].str.contains("NORMAL")]
                            else:
                                filtered_df = result_df
                                
                            st.dataframe(filtered_df, use_container_width=True, height=260)
                            
                            csv_output = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Unduh Hasil Pemindaian Jaringan (.csv)",
                                data=csv_output,
                                file_name="hasil_scan_iot_guard.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file Anda: {e}")
        else:
            # Welcome Screen Info
            st.markdown("<br>", unsafe_allow_html=True)
            col_info1, col_info2 = st.columns([3, 2])
            with col_info1:
                st.markdown("""
                ### 📌 Cara Penggunaan Aplikasi:
                1. **Unduh Sampel Data**: Tentukan jumlah baris data lalu tekan **'Unduh Baris Sampel Jaringan'** di panel demo sebelah kiri untuk mendapatkan berkas testing instan.
                2. **Unggah File Jaringan**: Seret & lepas file CSV tersebut ke area file uploader di sidebar sebelah kiri.
                3. **Jalankan Pemindaian**: Klik tombol **'Lakukan Deteksi Kerentanan Jaringan'** di panel utama.
                4. **Analisis Hasil**: Dashboard akan otomatis memproses data, mendeteksi tingkat risiko, status, dan visualisasi diagram serta statistik.
                """)
                st.markdown("""
                > [!TIP]
                > **Mengapa Kami Aman Dari Data Leakage?**  
                > Aplikasi memuat model objek Pipeline Sklearn utuh (`pipeline_terbaik.pkl`). Paket data baru diproses lewat penskalaan (`StandardScaler`) dan seleksi fitur (`SelectKBest`) menggunakan parameter historis yang terkunci di dalam Pipeline. Hal ini menjamin data baru tidak saling mempengaruhi secara statistik sebelum prediksi dibuat.
                """)
            with col_info2:
                st.markdown("<div class='metric-card' style='text-align: left; padding: 25px;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='color: #3b82f6; font-weight: 700; margin-top: 0;'>🛡️ Teknologi Proteksi IoT-Guard</h4>", unsafe_allow_html=True)
                st.markdown("""
                - **Filter Feature Selection**: Mengambil 15 fitur dengan kontribusi variansi tertinggi (`SelectKBest` ANOVA F-value).
                - **Robust Scaling**: Normalisasi fitur otomatis terikat dalam pipeline training.
                - **Balanced Classifier**: Mengatasi jomplangnya data serangan lewat undersampling cerdas (`RandomUnderSampler`) dan ditindaklanjuti dengan klasifikasi kokoh `RandomForest`.
                - **Deployment Terpadu**: Pipeline termuat utuh, menjamin stabilitas API di server Streamlit.
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# MENU 2: 📊 HASIL TRAINING & PERFORMA (Model Performance)
# ==============================================================================
elif menu_mode == "Hasil Training & Performa":
    st.markdown("<h1 style='font-size: 36px; font-weight: 700; margin-top: 0; background: linear-gradient(90deg, #ffffff 0%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>📊 Hasil Training & Performa Eksperimen</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 16px;'>Halaman interaktif untuk memeriksa hasil komparasi feature selection, skor final pengujian test set, matriks kebingungan (confusion matrix), detail hyperparameter terbaik, dan log MLflow.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Baris Kartu Metrik Performa Final
    st.subheader("Ringkasan Performa Akhir (Model Terbaik di Test Set)")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Akurasi Final (Test Set)</div><div class='metric-val-success'>99.87%</div><div style='color: #64748b; font-size: 12px; margin-top: 5px;'>Model Prediksi Sangat Akurat</div></div>", unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>F1-Score Macro</div><div class='metric-val-success'>99.56%</div><div style='color: #10b981; font-size: 12px; margin-top: 5px;'>Seimbang di Semua Kelas</div></div>", unsafe_allow_html=True)
    with col_p3:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Hyperparameter Terpilih</div><div class='metric-val-neutral'>k=15, n=20</div><div style='color: #3b82f6; font-size: 12px; margin-top: 5px;'>Filter Method (SelectKBest)</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Komparasi Feature Selection
    st.subheader("Perbandingan Metode Feature Selection (5-Fold Cross Validation)")
    
    cv_data = {
        'Method': ['Filter (SelectKBest)', 'Wrapper (RFE)', 'Embedded (SelectFromModel)'],
        'Accuracy': [0.998696, 1.000000, 1.000000],
        'F1-Score': [0.995758, 1.000000, 1.000000]
    }
    df_cv = pd.DataFrame(cv_data)

    col_chart, col_desc = st.columns([3, 2])
    with col_chart:
        fig_cv, ax_cv = plt.subplots(figsize=(8, 4))
        df_cv_plot = df_cv.melt(id_vars="Method", var_name="Metric", value_name="Score")
        sns.barplot(data=df_cv_plot, x="Metric", y="Score", hue="Method", palette="viridis", ax=ax_cv)
        ax_cv.set_ylim(0.9, 1.01)
        ax_cv.set_ylabel("Nilai Rata-rata (Mean)")
        ax_cv.set_xlabel("Metrik")
        ax_cv.legend(title="Metode FS", loc="lower right")
        plt.tight_layout()
        st.pyplot(fig_cv, clear_figure=True)
    
    with col_desc:
        st.markdown("""
        **Interpretasi Komparasi**:
        - **Wrapper (RFE)** dan **Embedded (SelectFromModel)** secara sempurna mencapai **100% Accuracy & F1-Score** pada data latihan. Pola anomali pada dataset akademis IoT ini sangat tajam dan linier.
        - Namun, **Filter (SelectKBest)** juga mencapai **99.87% Accuracy** dengan konsumsi daya komputasi dan waktu yang **jauh lebih efisien** (hanya memerlukan sepersekian detik untuk evaluasi dibanding RFE yang berulang kali memanggil model).
        - Oleh karena itu, **SelectKBest** terpilih sebagai model produksi final karena dinilai paling stabil, efisien, dan andal untuk skenario realtime *Packet Scanner*.
        """)

    st.markdown("---")

    # 2. Confusion Matrix & Classification Report
    st.subheader("Confusion Matrix & Detail Evaluasi Akhir")
    col_cm, col_rep = st.columns([3, 2])
    
    with col_cm:
        cm_matrix = np.array([[17370, 360], [0, 191985]])
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal (0)', 'Anomali (1)'], 
                    yticklabels=['Normal (0)', 'Anomali (1)'], ax=ax_cm)
        ax_cm.set_title("Confusion Matrix Heatmap (X_test)", fontsize=12, fontweight='bold', pad=10)
        ax_cm.set_xlabel("Predicted Labels")
        ax_cm.set_ylabel("True Labels")
        plt.tight_layout()
        st.pyplot(fig_cm, clear_figure=True)

    with col_rep:
        st.markdown("""
        **Laporan Klasifikasi Detail (Classification Report):**
        """)
        st.markdown("""
        | Kelas Target | Precision | Recall | F1-Score | Jumlah Sampel |
        | :--- | :---: | :---: | :---: | :---: |
        | **🟢 Normal (0)** | `1.00` | `0.98` | `0.99` | 17,730 |
        | **🔴 Anomali (1)** | `1.00` | `1.00` | `1.00` | 191,985 |
        | **Akurasi Rata-rata** | | | **`1.00`** | **209,715** |
        | **Macro Avg** | `1.00` | `0.99` | `1.00` | 209,715 |
        | **Weighted Avg** | `1.00` | `1.00` | `1.00` | 209,715 |
        """)
        st.caption("Porsi kelas target diuji secara acak dan seimbang (stratified) dari data test.")

    st.markdown("---")

    # 3. Dynamic MLflow database viewer
    st.subheader("Catatan Riwayat Training (Database MLflow Terintegrasi)")
    st.markdown("Berikut adalah tabel riwayat training yang diekstraksi secara *real-time* langsung dari database pelacakan lokal MLflow (`mlflow.db`):")
    
    if MLFLOW_AVAILABLE:
        try:
            # Hubungkan ke MLflow secara aman tanpa tabrakan nama impor
            runs = mlflow.search_runs(experiment_names=["IoT_Vulnerability_Detection"])
            if not runs.empty:
                # Saring kolom agar rapi & mudah dibaca
                kolom_log = ['run_id', 'start_time', 'params.Feature_Selection_Method', 
                             'params.K_Features', 'metrics.Accuracy', 'metrics.F1_Macro']
                kolom_show = [c for c in kolom_log if c in runs.columns]
                
                runs_show = runs[kolom_show].copy()
                # Ubah nama kolom agar ramah pengguna
                runs_show.rename(columns={
                    'run_id': 'Run ID',
                    'start_time': 'Waktu Eksperimen',
                    'params.Feature_Selection_Method': 'Metode Seleksi Fitur',
                    'params.K_Features': 'Jumlah Fitur (k)',
                    'metrics.Accuracy': 'Akurasi Akhir',
                    'metrics.F1_Macro': 'Skor F1-Macro'
                }, inplace=True)
                
                st.dataframe(runs_show, use_container_width=True)
                st.success(f"Ditemukan {len(runs_show)} eksperimen aktif di database MLflow.")
            else:
                st.info("Belum ada riwayat eksperimen aktif yang tercatat di database MLflow.")
        except Exception as e:
            st.warning(f"Tidak dapat membaca database MLflow secara otomatis: {e}.")
    else:
        st.info("**Informasi Pelacakan MLflow (Cloud Mode)**: Database eksperimen MLflow berjalan secara lokal pada komputer pengembangan. Di lingkungan cloud, model dimuat langsung dari `pipeline_terbaik.pkl` yang telah dilatih secara offline dengan parameter optimal: k=15 (SelectKBest) dan n_estimators=20 (RandomForestClassifier).")

# ==============================================================================
# MENU 3: 📁 DATASET EXPLORER (Eksplorasi Data)
# ==============================================================================
elif menu_mode == "📁 Dataset Explorer":
    st.markdown("<h1 style='font-size: 36px; font-weight: 700; margin-top: 0; background: linear-gradient(90deg, #ffffff 0%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>📁 Eksplorasi & Statistik Dataset IoT</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 16px;'>Analisis data mentah, cek missing values, dan cari tahu distribusi parameter lalu lintas jaringan IoT secara realtime.</p>", unsafe_allow_html=True)
    st.markdown("---")

    if df_slice is None:
        st.warning("ataset lokal 'Preprocessed_Balanced_dataset.csv' tidak terdeteksi di server. Harap jalankan langkah pengunduhan dataset di Jupyter Notebook terlebih dahulu.")
    else:
        # Kartu Deskripsi Umum
        st.subheader("ℹInformasi Umum Dataset IoT Vulnerability")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            st.markdown("<div class='metric-card'><div class='metric-label'>Total Baris Dataset</div><div class='metric-val-neutral'>1,048,575</div><div style='color: #64748b; font-size: 12px; margin-top: 5px;'>Sangat Besar & Variatif</div></div>", unsafe_allow_html=True)
        with col_d2:
            st.markdown("<div class='metric-card'><div class='metric-label'>Jumlah Fitur Jaringan</div><div class='metric-val-neutral'>85 Fitur</div><div style='color: #3b82f6; font-size: 12px; margin-top: 5px;'>Fitur Multi-Variat</div></div>", unsafe_allow_html=True)
        with col_d3:
            st.markdown("<div class='metric-card'><div class='metric-label'>Missing Values</div><div class='metric-val-success'>0 Data</div><div style='color: #10b981; font-size: 12px; margin-top: 5px;'>Dataset Sangat Bersih</div></div>", unsafe_allow_html=True)
        with col_d4:
            st.markdown("<div class='metric-card'><div class='metric-label'>Status Keseimbangan</div><div class='metric-val-success'>Balanced</div><div style='color: #10b981; font-size: 12px; margin-top: 5px;'>50% Normal : 50% Anomali</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tampilkan Sampel Data Jaringan (Cached slice)
        st.subheader("Cuplikan Data Jaringan (10 Baris Pertama)")
        st.dataframe(df_slice.head(10), use_container_width=True)
        st.caption("Menampilkan 10 baris teratas dari 1,000,000+ data jaringan lalu lintas IoT.")

        st.markdown("---")

        # 2. Visualisasi Dinamis Fitur
        st.subheader("Analisis Fitur Jaringan Dinamis")
        st.markdown("Pilih parameter fitur jaringan IoT di bawah ini untuk melihat distribusinya secara instan menggunakan visualisasi KDE (Kernel Density Estimate) & Histogram:")
        
        # Buang kolom non-fitur numerik untuk eksplorasi dinamis, dan saring kolom yang bernilai konstan (varian 0)
        kolom_drop = ['Label', 'Attack_Category', 'Attack_sub_category']
        kolom_fitur_expl = [col for col in df_slice.columns if col not in kolom_drop and df_slice[col].nunique() > 1]
        
        # Selectbox dinamis
        fitur_terpilih = st.selectbox("Pilih Fitur Jaringan:", kolom_fitur_expl)
        
        if fitur_terpilih:
            fig_fitur, ax_fitur = plt.subplots(figsize=(10, 4))
            
            # Buat grafik histogram & KDE yang menawan (dengan try-except jika KDE gagal akibat singularitas varian rendah)
            try:
                sns.histplot(data=df_slice, x=fitur_terpilih, hue="Label", kde=True, bins=30, 
                             palette={0: '#10b981', 1: '#ef4444'}, alpha=0.6, ax=ax_fitur)
            except Exception:
                sns.histplot(data=df_slice, x=fitur_terpilih, hue="Label", kde=False, bins=30, 
                             palette={0: '#10b981', 1: '#ef4444'}, alpha=0.6, ax=ax_fitur)
            
            ax_fitur.set_title(f"Distribusi Parameter Fitur '{fitur_terpilih}' Berdasarkan Label", fontsize=12, fontweight='bold', pad=10)
            ax_fitur.set_xlabel(fitur_terpilih)
            ax_fitur.set_ylabel("Frekuensi Paket")
            
            # Label Legenda Kustom
            leg = ax_fitur.get_legend()
            if leg:
                leg.set_title("Label Jaringan")
                new_labels = ['Normal (🟢)', 'Serangan (🔴)']
                for t, l in zip(leg.texts, new_labels):
                    t.set_text(l)
            
            plt.tight_layout()
            st.pyplot(fig_fitur, clear_figure=True)
            
            # Tampilkan informasi deskriptif statistiknya
            st.markdown(f"**Statistik Ringkasan Fitur `{fitur_terpilih}`:**")
            summary_df = df_slice[fitur_terpilih].describe().to_frame().T
            st.table(summary_df)