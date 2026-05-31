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
import matplotlib.patches as mpatches
import seaborn as sns

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="IoT-Guard | Vulnerability Detection Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. PREMIUM CSS — Cyberpunk Dark Mode dengan Animasi
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Base Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── App Background ── */
    .stApp {
        background: radial-gradient(ellipse at 0% 0%, rgba(30,58,138,0.25) 0%, transparent 60%),
                    radial-gradient(ellipse at 100% 100%, rgba(88,28,135,0.15) 0%, transparent 60%),
                    linear-gradient(160deg, #03070f 0%, #060d1a 50%, #060b17 100%);
        min-height: 100vh;
        color: #e2e8f0;
    }

    /* ── Hide Streamlit default chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 1400px; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060d1a 0%, #080f1f 100%) !important;
        border-right: 1px solid rgba(59,130,246,0.12) !important;
        backdrop-filter: blur(20px);
    }
    section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

    /* ── Sidebar Logo Area ── */
    .sidebar-logo {
        padding: 28px 20px 20px;
        text-align: center;
        border-bottom: 1px solid rgba(59,130,246,0.1);
        margin-bottom: 16px;
    }
    .sidebar-logo .logo-icon {
        font-size: 48px;
        display: block;
        margin-bottom: 8px;
        filter: drop-shadow(0 0 16px rgba(59,130,246,0.6));
        animation: pulse-icon 3s ease-in-out infinite;
    }
    @keyframes pulse-icon {
        0%, 100% { filter: drop-shadow(0 0 12px rgba(59,130,246,0.5)); transform: scale(1); }
        50% { filter: drop-shadow(0 0 24px rgba(99,102,241,0.8)); transform: scale(1.05); }
    }
    .sidebar-logo .logo-title {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #60a5fa, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-logo .logo-sub {
        font-size: 11px;
        color: #475569;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 2px;
    }

    /* ── Navigation Pills ── */
    div[data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        padding: 10px 14px !important;
        border-radius: 10px !important;
        margin: 3px 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        color: #94a3b8 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        border: 1px solid transparent !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(59,130,246,0.1) !important;
        color: #93c5fd !important;
        border-color: rgba(59,130,246,0.2) !important;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30,58,138,0.4) 0%, rgba(67,56,202,0.2) 50%, rgba(88,28,135,0.2) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 20px;
        padding: 36px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #3b82f6, #6366f1, #8b5cf6, transparent);
        animation: shimmer 4s ease-in-out infinite;
    }
    @keyframes shimmer {
        0% { opacity: 0.4; }
        50% { opacity: 1; }
        100% { opacity: 0.4; }
    }
    .hero-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #f8fafc 0%, #94a3b8 70%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px;
        line-height: 1.1;
    }
    .hero-sub {
        color: #64748b;
        font-size: 15px;
        line-height: 1.6;
        max-width: 700px;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.25);
        color: #34d399;
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 14px;
    }

    /* ── Section Header ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 28px 0 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .section-icon {
        width: 38px; height: 38px;
        background: linear-gradient(135deg, #1e3a8a, #3730a3);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
        flex-shrink: 0;
    }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
    }
    .section-desc {
        font-size: 13px;
        color: #64748b;
        margin: 0;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.9) 0%, rgba(23,37,64,0.8) 100%);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(12px);
    }
    .metric-card::after {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, transparent 100%);
        pointer-events: none;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99,102,241,0.35);
        box-shadow: 0 0 30px rgba(99,102,241,0.15), 0 20px 40px -10px rgba(0,0,0,0.5);
    }
    .metric-card .accent-bar {
        position: absolute;
        top: 0; left: 0;
        width: 100%;
        height: 3px;
    }
    .accent-blue { background: linear-gradient(90deg, #2563eb, #4f46e5); }
    .accent-green { background: linear-gradient(90deg, #059669, #10b981); }
    .accent-red { background: linear-gradient(90deg, #dc2626, #ef4444); }
    .accent-purple { background: linear-gradient(90deg, #7c3aed, #6366f1); }

    .metric-label {
        font-size: 11px;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .metric-val-success {
        font-size: 34px;
        font-weight: 800;
        color: #10b981;
        text-shadow: 0 0 20px rgba(16,185,129,0.4);
        font-feature-settings: 'tnum';
    }
    .metric-val-danger {
        font-size: 34px;
        font-weight: 800;
        color: #ef4444;
        text-shadow: 0 0 20px rgba(239,68,68,0.4);
        font-feature-settings: 'tnum';
    }
    .metric-val-neutral {
        font-size: 34px;
        font-weight: 800;
        color: #60a5fa;
        text-shadow: 0 0 20px rgba(96,165,250,0.4);
        font-feature-settings: 'tnum';
    }
    .metric-val-purple {
        font-size: 34px;
        font-weight: 800;
        color: #a78bfa;
        text-shadow: 0 0 20px rgba(167,139,250,0.4);
        font-feature-settings: 'tnum';
    }
    .metric-hint {
        font-size: 11px;
        color: #334155;
        margin-top: 6px;
    }

    /* ── Status Alerts ── */
    .alert-danger {
        background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(220,38,38,0.06) 100%);
        border: 1px solid rgba(239,68,68,0.3);
        border-left: 4px solid #ef4444;
        border-radius: 12px;
        padding: 18px 24px;
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 24px;
        animation: alert-pulse 2s ease-in-out infinite;
    }
    .alert-secure {
        background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(5,150,105,0.05) 100%);
        border: 1px solid rgba(16,185,129,0.25);
        border-left: 4px solid #10b981;
        border-radius: 12px;
        padding: 18px 24px;
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 24px;
    }
    @keyframes alert-pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
        50% { box-shadow: 0 0 20px rgba(239,68,68,0.15); }
    }
    .alert-icon { font-size: 32px; flex-shrink: 0; }
    .alert-title { font-size: 16px; font-weight: 700; margin: 0 0 3px; }
    .alert-body { font-size: 13px; color: #94a3b8; margin: 0; }

    /* ── Threat Gauge Bar ── */
    .gauge-wrap {
        background: rgba(15,23,42,0.8);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 24px;
    }
    .gauge-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #475569;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .gauge-track {
        background: rgba(255,255,255,0.06);
        border-radius: 999px;
        height: 10px;
        overflow: hidden;
        margin-bottom: 10px;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 1s ease;
    }
    .gauge-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .gauge-pct { font-size: 13px; color: #64748b; }

    /* ── Info Card ── */
    .info-card {
        background: rgba(15,23,42,0.7);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 22px 24px;
        backdrop-filter: blur(8px);
    }
    .info-card-title {
        font-size: 14px;
        font-weight: 700;
        color: #60a5fa;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .info-card ul {
        padding-left: 18px;
        margin: 0;
        color: #94a3b8;
        font-size: 13px;
        line-height: 1.8;
    }
    .info-card ul li strong { color: #e2e8f0; }

    /* ── Table Styling ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(90deg, #2563eb 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        border-radius: 10px !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 15px rgba(59,130,246,0.35) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ── Download Button ── */
    .stDownloadButton > button {
        background: linear-gradient(90deg, rgba(16,185,129,0.15) 0%, rgba(5,150,105,0.1) 100%) !important;
        color: #10b981 !important;
        border: 1px solid rgba(16,185,129,0.3) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(16,185,129,0.25) !important;
        border-color: rgba(16,185,129,0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* ── Selectbox / Slider ── */
    .stSelectbox label, .stSlider label, .stFileUploader label {
        color: #64748b !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 0.8px !important;
        text-transform: uppercase !important;
    }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.06) !important; margin: 28px 0 !important; }

    /* ── Stats Badge (for tables) ── */
    .stat-badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-green { background: rgba(16,185,129,0.15); color: #34d399; }
    .badge-red { background: rgba(239,68,68,0.15); color: #f87171; }

    /* ── Footer ── */
    .sidebar-footer {
        text-align: center;
        color: #1e293b;
        font-size: 11px;
        padding: 20px;
        border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 30px;
        letter-spacing: 0.5px;
    }
    .sidebar-footer span { color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Matplotlib Dark Theme (global)
# ============================================================
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor': '#0d1829',
    'axes.edgecolor': '#1e293b',
    'text.color': '#cbd5e1',
    'axes.labelcolor': '#94a3b8',
    'xtick.color': '#64748b',
    'ytick.color': '#64748b',
    'grid.color': '#1e293b',
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'legend.facecolor': '#0f172a',
    'legend.edgecolor': '#1e293b',
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ============================================================
# 3. LOAD ASSETS (cached)
# ============================================================
@st.cache_resource
def load_assets():
    model_path = 'pipeline_terbaik.pkl'
    le_path    = 'label_encoder.pkl'
    if os.path.exists(model_path) and os.path.exists(le_path):
        return joblib.load(model_path), joblib.load(le_path)
    return None, None

@st.cache_data
def load_dataset_slice():
    for path in ["Preprocessed_Balanced_dataset.csv", "sample_iot_data.csv"]:
        if os.path.exists(path):
            try:
                return pd.read_csv(path, nrows=10000)
            except Exception:
                continue
    return None

pipeline, le = load_assets()
df_slice     = load_dataset_slice()

# ============================================================
# 4. SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🛡️</span>
        <div class="logo-title">IoT-Guard</div>
        <div class="logo-sub">Vulnerability Detection Suite</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<p style='color:#475569;font-size:11px;font-weight:600;letter-spacing:1px;"
        "text-transform:uppercase;padding:0 8px;margin-bottom:6px;'>Navigasi</p>",
        unsafe_allow_html=True
    )

    menu_mode = st.radio(
        "menu",
        ["🛡️  Packet Scanner", "📊  Model Performance", "📁  Dataset Explorer"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # ── Demo Data Generator (only on Scanner page) ──
    if "Packet Scanner" in menu_mode:
        st.markdown(
            "<p style='color:#475569;font-size:11px;font-weight:600;letter-spacing:1px;"
            "text-transform:uppercase;padding:0 8px;margin-bottom:6px;'>Generator Data Demo</p>",
            unsafe_allow_html=True
        )

        sample_rows = st.slider("Jumlah Baris Sampel", 10, 150, 25, 5)

        def generate_sample_csv(nrows):
            for path in ["Preprocessed_Balanced_dataset.csv", "sample_iot_data.csv"]:
                if not os.path.exists(path):
                    continue
                try:
                    df_full = pd.read_csv(path, nrows=10000)
                    n_half  = max(2, nrows // 2)
                    avail0  = df_full[df_full['Label'] == 0]
                    avail1  = df_full[df_full['Label'] == 1]
                    n0 = min(n_half, len(avail0))
                    n1 = min(nrows - n_half, len(avail1))
                    df_sample = pd.concat([
                        avail0.sample(n=n0, random_state=42, replace=(n0 > len(avail0))),
                        avail1.sample(n=n1, random_state=42, replace=(n1 > len(avail1)))
                    ]).sample(frac=1.0, random_state=42)
                    drop_cols = ['Label', 'Attack_Category', 'Attack_sub_category']
                    df_clean  = df_sample.drop(columns=[c for c in drop_cols if c in df_sample.columns])
                    return df_clean.to_csv(index=False).encode('utf-8')
                except Exception:
                    continue
            return None

        sample_data = generate_sample_csv(sample_rows)
        if sample_data is not None:
            st.download_button(
                label=f"⬇️  Unduh {sample_rows} Baris Data",
                data=sample_data,
                file_name=f"sampel_iot_{sample_rows}baris.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.caption("Dataset lokal tidak tersedia untuk demo.")

    # ── File Uploader (only on Scanner page) ──
    if "Packet Scanner" in menu_mode and pipeline is not None:
        st.markdown("---")
        st.markdown(
            "<p style='color:#475569;font-size:11px;font-weight:600;letter-spacing:1px;"
            "text-transform:uppercase;padding:0 8px;margin-bottom:6px;'>Scan Hub</p>",
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader("Unggah CSV Aktivitas Jaringan", type=["csv"])
    else:
        uploaded_file = None

    st.markdown("""
    <div class="sidebar-footer">
        Mid ML II &copy; 2026<br>
        <span>Kelompok 3</span> · IoT Vulnerability
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ============================================================
#  PAGE: PACKET SCANNER
# ============================================================
# ============================================================
if "Packet Scanner" in menu_mode:

    # Hero Banner
    model_status_badge = (
        '<span class="hero-badge">● Model Aktif</span>' if pipeline is not None
        else '<span class="hero-badge" style="background:rgba(239,68,68,.12);border-color:rgba(239,68,68,.25);color:#f87171;">● Model Tidak Tersedia</span>'
    )
    st.markdown(f"""
    <div class="hero-banner">
        {model_status_badge}
        <div class="hero-title">IoT Packet Scanner</div>
        <div class="hero-sub">
            Sistem deteksi anomali lalu lintas jaringan IoT berbasis Machine Learning.
            Unggah data aktivitas jaringan Anda untuk mengidentifikasi potensi serangan secara <em>real-time</em>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if pipeline is None or le is None:
        st.markdown("""
        <div class="alert-danger">
            <span class="alert-icon">⚠️</span>
            <div>
                <div class="alert-title" style="color:#f87171;">Model Belum Tersedia</div>
                <div class="alert-body">File <code>pipeline_terbaik.pkl</code> atau <code>label_encoder.pkl</code>
                tidak ditemukan. Jalankan <code>python mlflow.py</code> terlebih dahulu untuk menghasilkan artefak model.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                drop_cols  = ['Label', 'Attack_Category', 'Attack_sub_category']
                new_data_clean = new_data.drop(columns=[c for c in drop_cols if c in new_data.columns])

                with st.expander("🔍  Pratinjau Data Masukan", expanded=False):
                    st.dataframe(new_data_clean.head(5), use_container_width=True)
                    st.caption(f"Terdeteksi {new_data_clean.shape[1]} kolom fitur, {len(new_data_clean):,} baris.")

                if st.button("🚀  Lakukan Deteksi Kerentanan Jaringan", use_container_width=True):
                    with st.spinner("Memproses paket jaringan…"):
                        preds_enc   = pipeline.predict(new_data_clean)
                        preds_label = le.inverse_transform(preds_enc)
                        total    = len(new_data_clean)
                        anomali  = int(np.sum(preds_enc == 1))
                        normal   = total - anomali
                        threat_pct = (anomali / total) * 100

                    # Risk level
                    if threat_pct == 0:
                        rl, rc = "SECURE — Jaringan Aman", "#10b981"
                        gauge_grad = "linear-gradient(90deg,#059669,#10b981)"
                    elif threat_pct <= 10:
                        rl, rc = "WARNING — Risiko Menengah", "#fbbf24"
                        gauge_grad = "linear-gradient(90deg,#d97706,#fbbf24)"
                    elif threat_pct <= 30:
                        rl, rc = "DANGER — Risiko Tinggi", "#f97316"
                        gauge_grad = "linear-gradient(90deg,#ea580c,#f97316)"
                    else:
                        rl, rc = "CRITICAL — Sistem Terancam!", "#ef4444"
                        gauge_grad = "linear-gradient(90deg,#dc2626,#ef4444)"

                    # Status Banner
                    if anomali > 0:
                        st.markdown(f"""
                        <div class="alert-danger">
                            <span class="alert-icon">🚨</span>
                            <div>
                                <div class="alert-title" style="color:#f87171;">BAHAYA TERDETEKSI!</div>
                                <div class="alert-body">Ditemukan <strong style="color:#fca5a5;">{anomali:,} paket</strong>
                                terindikasi anomali/serangan. Tinjau log di bawah untuk tindak lanjut segera.</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="alert-secure">
                            <span class="alert-icon">🛡️</span>
                            <div>
                                <div class="alert-title" style="color:#34d399;">JARINGAN AMAN</div>
                                <div class="alert-body">Seluruh paket yang di-scan tergolong normal. Tidak ada aktivitas mencurigakan.</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Threat Gauge
                    st.markdown(f"""
                    <div class="gauge-wrap">
                        <div class="gauge-label">Tingkat Ancaman Jaringan Real-Time</div>
                        <div class="gauge-track">
                            <div class="gauge-fill" style="width:{min(threat_pct,100):.1f}%;background:{gauge_grad};"></div>
                        </div>
                        <div class="gauge-row">
                            <span style="font-size:14px;font-weight:700;color:{rc};">{rl}</span>
                            <span class="gauge-pct">{threat_pct:.1f}% Terindikasi Ancaman</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Metric Cards
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="accent-bar accent-blue"></div>
                            <div class="metric-label">Total Paket Di-Scan</div>
                            <div class="metric-val-neutral">{total:,}</div>
                            <div class="metric-hint">Paket lalu lintas jaringan</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="accent-bar accent-green"></div>
                            <div class="metric-label">Paket Normal</div>
                            <div class="metric-val-success">{normal:,}</div>
                            <div class="metric-hint">{(normal/total)*100:.1f}% bebas ancaman</div>
                        </div>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="accent-bar accent-red"></div>
                            <div class="metric-label">Anomali Terdeteksi</div>
                            <div class="metric-val-danger">{anomali:,}</div>
                            <div class="metric-hint">{(anomali/total)*100:.1f}% perlu tindakan</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Visualization + Log
                    chart_col, table_col = st.columns([2, 3])

                    with chart_col:
                        st.markdown("<div class='section-header'><div class='section-icon'>📈</div><div><div class='section-title'>Distribusi Klasifikasi</div><div class='section-desc'>Komposisi paket hasil scan</div></div></div>", unsafe_allow_html=True)

                        fig, ax = plt.subplots(figsize=(6, 5))
                        sizes  = [normal, anomali]
                        clrs   = ['#10b981', '#ef4444']
                        explode = (0, 0.08) if anomali > 0 else (0, 0)
                        wedges, texts, autotexts = ax.pie(
                            sizes, explode=explode, labels=['Normal', 'Anomali'],
                            colors=clrs, autopct='%1.1f%%', startangle=130,
                            textprops=dict(color='#cbd5e1', fontsize=12, fontweight='bold'),
                            wedgeprops=dict(width=0.45, edgecolor='#060d1a', linewidth=2.5)
                        )
                        for at in autotexts:
                            at.set_color('#fff')
                            at.set_fontsize(11)
                        ax.axis('equal')
                        plt.tight_layout()
                        st.pyplot(fig, clear_figure=True)

                    with table_col:
                        st.markdown("<div class='section-header'><div class='section-icon'>📋</div><div><div class='section-title'>Log Prediksi Detail</div><div class='section-desc'>Hasil deteksi per paket</div></div></div>", unsafe_allow_html=True)

                        result_df = new_data_clean.copy()
                        result_df['STATUS'] = [
                            "🟢 NORMAL" if x == 0 else "🔴 ANOMALI"
                            for x in preds_enc
                        ]

                        filter_opt = st.selectbox(
                            "Filter Log:",
                            ["Semua Paket", "Hanya Anomali", "Hanya Normal"],
                            label_visibility="visible"
                        )
                        mask = {
                            "Hanya Anomali": result_df['STATUS'].str.contains("ANOMALI"),
                            "Hanya Normal":  result_df['STATUS'].str.contains("NORMAL"),
                        }.get(filter_opt, pd.Series([True]*len(result_df)))

                        st.dataframe(result_df[mask], use_container_width=True, height=280)

                        st.download_button(
                            label="⬇️  Ekspor Hasil Scan (.csv)",
                            data=result_df.to_csv(index=False).encode('utf-8'),
                            file_name="hasil_scan_iot_guard.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")

        else:
            # Welcome / How-to screen
            st.markdown("<br>", unsafe_allow_html=True)
            col_how, col_tech = st.columns([3, 2], gap="large")

            with col_how:
                st.markdown("""
                <div class="section-header">
                    <div class="section-icon">📌</div>
                    <div>
                        <div class="section-title">Cara Menggunakan Packet Scanner</div>
                        <div class="section-desc">Ikuti langkah di bawah ini untuk mulai memindai jaringan</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                steps = [
                    ("1", "#3b82f6", "Unduh Data Sampel", "Geser slider di sidebar untuk memilih jumlah baris, lalu klik tombol <b>Unduh Baris Data</b> untuk mendapat file testing instan."),
                    ("2", "#6366f1", "Unggah File CSV", "Seret &amp; lepas file CSV aktivitas jaringan ke area <b>Scan Hub</b> di panel sidebar kiri."),
                    ("3", "#8b5cf6", "Jalankan Pemindaian", "Klik tombol <b>Lakukan Deteksi Kerentanan</b> di panel utama untuk memproses paket."),
                    ("4", "#a78bfa", "Analisis Hasil", "Dashboard menampilkan status risiko, gauge ancaman, distribusi visual, dan log prediksi lengkap."),
                ]
                for num, color, title, desc in steps:
                    st.markdown(f"""
                    <div style="display:flex;gap:16px;align-items:flex-start;margin-bottom:20px;">
                        <div style="width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,{color}44,{color}22);
                            border:1px solid {color}55;display:flex;align-items:center;justify-content:center;
                            font-size:16px;font-weight:800;color:{color};flex-shrink:0;">{num}</div>
                        <div>
                            <div style="font-size:14px;font-weight:700;color:#e2e8f0;margin-bottom:4px;">{title}</div>
                            <div style="font-size:13px;color:#64748b;line-height:1.6;">{desc}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_tech:
                st.markdown("""
                <div class="info-card">
                    <div class="info-card-title">🔧 Teknologi Proteksi IoT-Guard</div>
                    <ul>
                        <li><strong>Filter Feature Selection</strong><br>
                            Memilih 15 fitur terpenting via SelectKBest (ANOVA F-value).</li>
                        <li><strong>Standard Scaling</strong><br>
                            Normalisasi otomatis terkunci di dalam pipeline.</li>
                        <li><strong>RandomUnderSampler</strong><br>
                            Menyeimbangkan distribusi kelas secara cerdas.</li>
                        <li><strong>Random Forest Classifier</strong><br>
                            Ensemble 20 pohon keputusan yang kokoh dan stabil.</li>
                        <li><strong>Zero Data Leakage</strong><br>
                            Pipeline utuh memastikan data baru diproses secara independen.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)


# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
elif "Model Performance" in menu_mode:

    st.markdown("""
    <div class="hero-banner">
        <span class="hero-badge">📊 Hasil Eksperimen</span>
        <div class="hero-title">Performa & Evaluasi Model</div>
        <div class="hero-sub">
            Komparasi metode feature selection, confusion matrix, classification report,
            dan riwayat logging MLflow dari eksperimen terbaik.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top Metric Cards ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-card">
            <div class="accent-bar accent-green"></div>
            <div class="metric-label">Akurasi Final</div>
            <div class="metric-val-success">99.87%</div>
            <div class="metric-hint">Pada data uji independen</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card">
            <div class="accent-bar accent-green"></div>
            <div class="metric-label">F1-Score Macro</div>
            <div class="metric-val-success">99.56%</div>
            <div class="metric-hint">Seimbang di semua kelas</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card">
            <div class="accent-bar accent-blue"></div>
            <div class="metric-label">Fitur Terpilih (k)</div>
            <div class="metric-val-neutral">15</div>
            <div class="metric-hint">SelectKBest · Filter Method</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card">
            <div class="accent-bar accent-purple"></div>
            <div class="metric-label">Pohon Keputusan</div>
            <div class="metric-val-purple">20</div>
            <div class="metric-hint">n_estimators · Random Forest</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 1: Feature Selection Comparison ──
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🔬</div>
        <div>
            <div class="section-title">Perbandingan Metode Feature Selection</div>
            <div class="section-desc">5-Fold Stratified Cross Validation pada data latihan</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cv_data = {
        'Metode': ['Filter\n(SelectKBest)', 'Wrapper\n(RFE)', 'Embedded\n(SelectFromModel)'],
        'Accuracy': [0.998696, 1.000000, 1.000000],
        'F1-Score': [0.995758, 1.000000, 1.000000],
    }
    df_cv = pd.DataFrame(cv_data)

    col_chart, col_desc = st.columns([3, 2], gap="large")

    with col_chart:
        fig_cv, ax_cv = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(df_cv['Metode']))
        w = 0.35
        bars1 = ax_cv.bar(x - w/2, df_cv['Accuracy'], w, label='Accuracy',
                          color=['#3b82f6','#4f46e5','#7c3aed'], alpha=0.9,
                          edgecolor='#0f172a', linewidth=1.2)
        bars2 = ax_cv.bar(x + w/2, df_cv['F1-Score'], w, label='F1-Score',
                          color=['#10b981','#059669','#047857'], alpha=0.9,
                          edgecolor='#0f172a', linewidth=1.2)

        # Value labels on bars
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax_cv.text(bar.get_x() + bar.get_width()/2., h + 0.0005,
                       f'{h:.4f}', ha='center', va='bottom',
                       fontsize=9, color='#94a3b8', fontweight='bold')

        ax_cv.set_ylim(0.985, 1.005)
        ax_cv.set_xticks(x)
        ax_cv.set_xticklabels(df_cv['Metode'], fontsize=11)
        ax_cv.set_ylabel("Nilai Rata-rata CV", fontsize=11)
        ax_cv.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.3f}'))
        ax_cv.legend(loc='lower right', fontsize=10)
        ax_cv.grid(axis='y', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_cv, clear_figure=True)

    with col_desc:
        st.markdown("""
        <div class="info-card" style="margin-top:8px;">
            <div class="info-card-title">📝 Interpretasi Komparasi</div>
            <ul>
                <li><strong>Wrapper (RFE)</strong> &amp; <strong>Embedded (SelectFromModel)</strong>
                    mencapai <em>100% Accuracy &amp; F1-Score</em> pada data latihan.</li>
                <li><strong>Filter (SelectKBest)</strong> mencapai <em>99.87% Accuracy</em>
                    namun dengan waktu komputasi jauh lebih singkat — tanpa memanggil model berulang kali.</li>
                <li>Oleh karena itu, <strong>SelectKBest terpilih</strong> sebagai strategi produksi
                    final: paling stabil, efisien, dan andal untuk skenario real-time.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Section 2: Confusion Matrix ──
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🎯</div>
        <div>
            <div class="section-title">Confusion Matrix & Classification Report</div>
            <div class="section-desc">Evaluasi detail pada data uji (X_test)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cm, col_rep = st.columns([3, 2], gap="large")

    with col_cm:
        cm = np.array([[17370, 360], [0, 191985]])
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))

        # Custom color palette
        cmap = sns.color_palette(["#0f172a", "#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa"], as_cmap=True)
        sns.heatmap(cm, annot=True, fmt='d',
                    cmap='YlOrRd',
                    xticklabels=['Normal', 'Anomali'],
                    yticklabels=['Normal', 'Anomali'],
                    ax=ax_cm, linewidths=2, linecolor='#0f172a',
                    annot_kws={'size': 16, 'weight': 'bold', 'color': 'white'})
        ax_cm.set_title("Confusion Matrix · Data Uji", fontsize=13, fontweight='bold', pad=14, color='#e2e8f0')
        ax_cm.set_xlabel("Prediksi", fontsize=11)
        ax_cm.set_ylabel("Aktual", fontsize=11)
        ax_cm.tick_params(labelsize=11, colors='#94a3b8')
        plt.tight_layout()
        st.pyplot(fig_cm, clear_figure=True)

    with col_rep:
        st.markdown("**Laporan Klasifikasi Detail (Test Set)**", unsafe_allow_html=False)
        st.markdown("""
        | Kelas | Precision | Recall | F1 | Support |
        |:---|:---:|:---:|:---:|:---:|
        | 🟢 Normal | `1.00` | `0.98` | `0.99` | 17,730 |
        | 🔴 Anomali | `1.00` | `1.00` | `1.00` | 191,985 |
        | **Macro Avg** | `1.00` | `0.99` | `1.00` | 209,715 |
        | **Weighted Avg** | `1.00` | `1.00` | `1.00` | 209,715 |
        """)
        st.caption("Data uji diambil secara stratified 20% dari keseluruhan dataset.")

        st.markdown("<br>", unsafe_allow_html=True)
        # Mini score bars
        for label, score, color in [("Akurasi", 0.9987, "#10b981"), ("F1 Macro", 0.9956, "#3b82f6"), ("Precision", 1.000, "#a78bfa")]:
            pct = int(score * 100)
            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-size:12px;color:#94a3b8;font-weight:600;">{label}</span>
                    <span style="font-size:12px;color:{color};font-weight:700;">{score:.2%}</span>
                </div>
                <div style="background:rgba(255,255,255,0.06);border-radius:999px;height:6px;overflow:hidden;">
                    <div style="width:{pct}%;height:100%;background:{color};border-radius:999px;
                        box-shadow:0 0 8px {color}88;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Section 3: MLflow Log ──
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🔍</div>
        <div>
            <div class="section-title">Riwayat Eksperimen MLflow</div>
            <div class="section-desc">Catatan run yang tersimpan di database SQLite lokal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if MLFLOW_AVAILABLE:
        try:
            runs = mlflow.search_runs(experiment_names=["IoT_Vulnerability_Detection"])
            if not runs.empty:
                col_log = ['run_id', 'start_time', 'params.Feature_Selection_Method',
                           'params.K_Features', 'metrics.Accuracy', 'metrics.F1_Macro']
                runs_show = runs[[c for c in col_log if c in runs.columns]].copy()
                runs_show.rename(columns={
                    'run_id': 'Run ID',
                    'start_time': 'Waktu Run',
                    'params.Feature_Selection_Method': 'Metode Seleksi',
                    'params.K_Features': 'k Fitur',
                    'metrics.Accuracy': 'Akurasi',
                    'metrics.F1_Macro': 'F1 Macro'
                }, inplace=True)
                st.dataframe(runs_show, use_container_width=True)
                st.success(f"✅  {len(runs_show)} eksperimen tercatat di MLflow database.")
            else:
                st.info("Belum ada riwayat eksperimen. Jalankan `python mlflow.py` untuk mencatat run pertama.")
        except Exception as e:
            st.warning(f"Tidak dapat membaca database MLflow: {e}")
    else:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">💡 Cloud Mode — MLflow Tidak Tersedia</div>
            <ul>
                <li>Database eksperimen MLflow berjalan secara lokal pada komputer pengembangan.</li>
                <li>Di lingkungan cloud, model dimuat langsung dari <strong>pipeline_terbaik.pkl</strong>
                    yang telah dilatih offline.</li>
                <li>Parameter final: <strong>k=15 (SelectKBest)</strong>,
                    <strong>n_estimators=20 (RandomForestClassifier)</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE: DATASET EXPLORER
# ============================================================
elif "Dataset Explorer" in menu_mode:

    st.markdown("""
    <div class="hero-banner">
        <span class="hero-badge">📁 Eksplorasi Data</span>
        <div class="hero-title">Dataset Explorer</div>
        <div class="hero-sub">
            Jelajahi statistik, distribusi fitur, dan karakteristik lalu lintas jaringan IoT
            secara interaktif untuk memahami pola tersembunyi di balik data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df_slice is None:
        st.markdown("""
        <div class="alert-danger">
            <span class="alert-icon">⚠️</span>
            <div>
                <div class="alert-title" style="color:#f87171;">Dataset Tidak Ditemukan</div>
                <div class="alert-body">File <code>Preprocessed_Balanced_dataset.csv</code> atau
                <code>sample_iot_data.csv</code> tidak tersedia. Pastikan salah satunya ada di direktori proyek.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        is_sample = not os.path.exists("Preprocessed_Balanced_dataset.csv")
        if is_sample:
            st.info("📌  Menampilkan data dari file sampel `sample_iot_data.csv` (200 baris representatif).")

        # Dataset Overview Cards
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown("""
            <div class="metric-card">
                <div class="accent-bar accent-blue"></div>
                <div class="metric-label">Total Baris Dataset</div>
                <div class="metric-val-neutral">1,048,575</div>
                <div class="metric-hint">Sangat besar &amp; variatif</div>
            </div>""", unsafe_allow_html=True)
        with d2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="accent-bar accent-blue"></div>
                <div class="metric-label">Jumlah Fitur</div>
                <div class="metric-val-neutral">{len(df_slice.columns)}</div>
                <div class="metric-hint">Parameter jaringan multi-variat</div>
            </div>""", unsafe_allow_html=True)
        with d3:
            st.markdown("""
            <div class="metric-card">
                <div class="accent-bar accent-green"></div>
                <div class="metric-label">Missing Values</div>
                <div class="metric-val-success">0</div>
                <div class="metric-hint">Dataset sangat bersih</div>
            </div>""", unsafe_allow_html=True)
        with d4:
            st.markdown("""
            <div class="metric-card">
                <div class="accent-bar accent-green"></div>
                <div class="metric-label">Status Keseimbangan</div>
                <div class="metric-val-success">Balanced</div>
                <div class="metric-hint">Normal ≈ Anomali (undersampled)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Data Snapshot
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📋</div>
            <div>
                <div class="section-title">Cuplikan Data Jaringan</div>
                <div class="section-desc">10 baris pertama dari dataset lalu lintas IoT</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_slice.head(10), use_container_width=True)
        st.caption(f"Ditampilkan 10 dari {len(df_slice):,} baris yang dimuat. Dataset asli: 1,000,000+ baris.")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Dynamic Feature Analysis
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📊</div>
            <div>
                <div class="section-title">Analisis Fitur Dinamis</div>
                <div class="section-desc">Visualisasi distribusi KDE + Histogram per fitur jaringan berdasarkan label</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        drop_cols   = ['Label', 'Attack_Category', 'Attack_sub_category']
        valid_cols  = [c for c in df_slice.columns
                       if c not in drop_cols and df_slice[c].nunique() > 1
                       and pd.api.types.is_numeric_dtype(df_slice[c])]

        sel_feat = st.selectbox("Pilih fitur jaringan untuk dieksplorasi:", valid_cols)

        if sel_feat and 'Label' in df_slice.columns:
            fig_f, ax_f = plt.subplots(figsize=(11, 4.5))
            try:
                sns.histplot(data=df_slice, x=sel_feat, hue='Label', kde=True, bins=35,
                             palette={0: '#10b981', 1: '#ef4444'}, alpha=0.55, ax=ax_f)
            except Exception:
                sns.histplot(data=df_slice, x=sel_feat, hue='Label', kde=False, bins=35,
                             palette={0: '#10b981', 1: '#ef4444'}, alpha=0.55, ax=ax_f)

            ax_f.set_title(f"Distribusi Fitur  ·  {sel_feat}", fontsize=13, fontweight='bold', pad=12, color='#e2e8f0')
            ax_f.set_xlabel(sel_feat, fontsize=11)
            ax_f.set_ylabel("Frekuensi Paket", fontsize=11)
            ax_f.grid(axis='y', alpha=0.3)

            leg = ax_f.get_legend()
            if leg:
                leg.set_title("Label", prop={'size': 10})
                for t, lbl in zip(leg.texts, ['Normal (0)', 'Anomali (1)']):
                    t.set_text(lbl)

            plt.tight_layout()
            st.pyplot(fig_f, clear_figure=True)

            # Summary statistics
            st.markdown(f"**Statistik Ringkasan — `{sel_feat}`**")
            st.table(df_slice[sel_feat].describe().to_frame().T.style.format("{:.4f}"))