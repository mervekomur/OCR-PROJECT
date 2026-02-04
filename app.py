"""
FLO OCR Web Interface
RG Labs - Finansal Belge Isleme Sistemi
"""

import streamlit as st
import time
import json
import os
import sys
import base64
import tempfile
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Streamlit Cloud secrets support
def get_secret(key, default=None):
    """Get secret from Streamlit Cloud or environment variable"""
    try:
        return st.secrets.get(key, os.environ.get(key, default))
    except:
        return os.environ.get(key, default)

# Set environment variables from Streamlit secrets (for Google Vision)
try:
    if "GOOGLE_CREDENTIALS_JSON" in st.secrets:
        # Write Google credentials to temp file for Streamlit Cloud
        creds_json = st.secrets["GOOGLE_CREDENTIALS_JSON"]
        creds_path = Path(tempfile.gettempdir()) / "google_creds.json"
        creds_path.write_text(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass  # Local development - use .env file

st.set_page_config(
    page_title="FLO OCR | RG Labs",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load butterfly logo
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

butterfly_path = Path(__file__).parent / "butterfly.png"
butterfly_b64 = get_base64_image(butterfly_path) if butterfly_path.exists() else ""

# Light Modern Theme
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {{
        background: linear-gradient(160deg, #f8fafc 0%, #e2e8f0 50%, #f1f5f9 100%);
        font-family: 'Inter', sans-serif;
    }}

    /* Header */
    .header-box {{
        background: white;
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }}

    .logo-img {{
        width: 55px;
        height: 55px;
    }}

    .main-title {{
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }}

    .subtitle {{
        color: #64748b;
        text-align: center;
        font-size: 0.9rem;
        margin: 0.5rem 0 1rem 0;
    }}

    /* Cards */
    .card {{
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
    }}

    .card-header {{
        color: #334155;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    /* Metric cards */
    .metric-card {{
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        margin: 0.4rem 0;
    }}

    .metric-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #6366f1;
    }}

    .metric-label {{
        font-size: 0.65rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.2rem;
    }}

    /* Log box */
    .log-box {{
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.78rem;
        max-height: 160px;
        overflow-y: auto;
    }}

    .log-success {{ color: #4ade80; }}
    .log-warning {{ color: #fbbf24; }}
    .log-error {{ color: #f87171; }}
    .log-info {{ color: #818cf8; }}
    .log-time {{ color: #64748b; font-size: 0.7rem; }}

    /* Status badges */
    .badge-success {{
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        color: #166534;
        padding: 0.5rem 1.5rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #86efac;
    }}

    .badge-error {{
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #991b1b;
        padding: 0.5rem 1.5rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #fca5a5;
    }}

    /* Result cards */
    .result-item {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin: 0.4rem 0;
    }}

    .result-label {{
        color: #94a3b8;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .result-value {{
        color: #1e293b;
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }}

    .result-sub {{
        color: #64748b;
        font-size: 0.8rem;
    }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        transition: all 0.2s;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: white;
        border-right: 1px solid #e2e8f0;
    }}

    section[data-testid="stSidebar"] .block-container {{
        padding: 1.5rem 1rem;
    }}

    /* API Status - clean design */
    .api-status {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 0.8rem;
        background: #f8fafc;
        border-radius: 10px;
        margin: 0.3rem 0;
        border: 1px solid #e2e8f0;
    }}

    .api-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }}

    .api-dot.active {{
        background: #22c55e;
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
    }}

    .api-dot.inactive {{
        background: #ef4444;
    }}

    .api-name {{
        color: #475569;
        font-size: 0.85rem;
        font-weight: 500;
    }}

    /* Hide defaults */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: #f1f5f9;
        padding: 4px;
        border-radius: 10px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 500;
        padding: 0.5rem 1rem;
    }}

    .stTabs [aria-selected="true"] {{
        background: white;
        color: #6366f1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}

    /* File uploader */
    .stFileUploader > div {{
        background: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 14px;
    }}

    .stFileUploader > div:hover {{
        border-color: #6366f1;
        background: #f1f5f9;
    }}

    /* Section title */
    .section-title {{
        color: #334155;
        font-size: 1rem;
        font-weight: 600;
        margin: 0.8rem 0 0.5rem 0;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 6px;
    }}

    ::-webkit-scrollbar-track {{
        background: #f1f5f9;
    }}

    ::-webkit-scrollbar-thumb {{
        background: #cbd5e1;
        border-radius: 3px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: #6366f1;
    }}

    /* Radio buttons */
    .stRadio > div {{
        background: #f8fafc;
        padding: 0.5rem;
        border-radius: 10px;
    }}
</style>
""", unsafe_allow_html=True)


# Session state
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'results' not in st.session_state:
    st.session_state.results = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {"latency": "-", "confidence": "-", "tokens": "-"}


def add_log(msg, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({"time": ts, "msg": msg, "level": level})


def clear_logs():
    st.session_state.logs = []


def process_image(image_bytes, filename, pipeline="fast"):
    start = time.time()

    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / filename

    with open(temp_path, "wb") as f:
        f.write(image_bytes)

    try:
        add_log("Islem basliyor...", "info")
        add_log("Google Vision API baglaniyor...", "info")

        from engines.hybrid_vision_engine import HybridVisionEngine

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            add_log("ANTHROPIC_API_KEY bulunamadi!", "error")
            return {"success": False, "error": "API key eksik"}

        engine = HybridVisionEngine(anthropic_api_key=api_key)
        engine._ensure_initialized()

        add_log("Motorlar hazir", "success")
        add_log("Belge okunuyor...", "info")

        result = engine.extract(str(temp_path))

        add_log("Google Vision tamamlandi", "success")
        add_log("Claude analiz ediyor...", "info")

        if pipeline == "deep":
            add_log("Derin analiz yapiliyor...", "info")
            time.sleep(0.3)

        add_log("Claude tamamlandi", "success")

        validation = result.fields.get("validation", {})
        status = validation.get("islem_durumu", "BILINMIYOR")

        if status == "ONAYLANDI":
            add_log("Belge ONAYLANDI", "success")
        else:
            reason = validation.get("red_sebebi", "")
            add_log(f"Belge REDDEDILDI: {reason}", "warning")

        elapsed = time.time() - start
        add_log(f"Tamamlandi! Sure: {elapsed:.2f}s", "success")

        st.session_state.metrics = {
            "latency": f"{elapsed:.1f}s",
            "confidence": f"{result.confidence*100:.0f}%",
            "tokens": str(result.metadata.get("tokens_used", "-")) if hasattr(result, 'metadata') else "-"
        }

        return {
            "success": True,
            "data": result.fields,
            "raw_text": result.raw_text,
            "confidence": result.confidence
        }

    except Exception as e:
        add_log(f"Hata: {str(e)}", "error")
        return {"success": False, "error": str(e)}
    finally:
        if temp_path.exists():
            temp_path.unlink()


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    # Logo and title
    if butterfly_b64:
        st.markdown(f'''
        <div style="text-align: center; padding: 1rem 0 1.5rem 0;">
            <img src="data:image/png;base64,{butterfly_b64}" style="width: 45px; margin-bottom: 0.5rem;">
            <h2 style="color: #334155; margin: 0; font-size: 1.2rem; font-weight: 700;">RG Labs</h2>
            <p style="color: #94a3b8; font-size: 0.75rem; margin: 0;">Finansal OCR</p>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("**Pipeline Modu**")
    pipeline = st.radio(
        "Pipeline",
        ["fast", "deep"],
        format_func=lambda x: "Hizli Pipeline" if x == "fast" else "Derin Pipeline",
        label_visibility="collapsed"
    )

    st.markdown("---")

    # API Status - clean design
    st.markdown("**API Durumu**")
    api_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
    google_ok = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

    st.markdown(f'''
    <div class="api-status">
        <div class="api-dot {'active' if api_ok else 'inactive'}"></div>
        <span class="api-name">Claude API</span>
    </div>
    <div class="api-status">
        <div class="api-dot {'active' if google_ok else 'inactive'}"></div>
        <span class="api-name">Google Vision</span>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("---")

    # Metrics
    st.markdown("**Son Islem**")
    m = st.session_state.metrics

    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-value">{m['latency']}</div>
        <div class="metric-label">Latency</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-value">{m['confidence']}</div>
        <div class="metric-label">Confidence</div>
    </div>
    ''', unsafe_allow_html=True)


# =============================================================================
# MAIN CONTENT
# =============================================================================
# Header
st.markdown(f'''
<div class="header-box">
    {"<img src='data:image/png;base64," + butterfly_b64 + "' class='logo-img'>" if butterfly_b64 else ""}
    <h1 class="main-title">FLO OCR Engine</h1>
</div>
<p class="subtitle">Google Vision + Claude AI | Akilli Belge Isleme Sistemi</p>
''', unsafe_allow_html=True)

# Main layout
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown('<div class="section-title">📤 Belge Yukle</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Fis veya fatura gorseli",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)

        with c2:
            st.markdown(f'''
            <div class="card">
                <div class="card-header">📋 Dosya Bilgisi</div>
                <p style="color: #475569; margin: 0.3rem 0;"><b>Ad:</b> {uploaded.name}</p>
                <p style="color: #475569; margin: 0.3rem 0;"><b>Boyut:</b> {uploaded.size/1024:.1f} KB</p>
                <p style="color: #475569; margin: 0.3rem 0;"><b>Mod:</b> {'Hizli' if pipeline=='fast' else 'Derin'}</p>
            </div>
            ''', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🚀 Islemi Baslat", use_container_width=True):
                clear_logs()
                with st.spinner("Isleniyor..."):
                    result = process_image(uploaded.getvalue(), uploaded.name, pipeline)
                    st.session_state.results = result
                st.rerun()

    # Log Panel
    st.markdown('<div class="section-title">📋 Islem Logu</div>', unsafe_allow_html=True)

    if st.session_state.logs:
        log_html = '<div class="log-box">'
        for log in st.session_state.logs[-8:]:
            cls = f"log-{log['level']}"
            log_html += f'<div><span class="log-time">[{log["time"]}]</span> <span class="{cls}">{log["msg"]}</span></div>'
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="log-box" style="color:#64748b;text-align:center;padding:1.5rem;">Belge yukleyip islemi baslatin</div>', unsafe_allow_html=True)


with col_right:
    st.markdown('<div class="section-title">📊 Sonuclar</div>', unsafe_allow_html=True)

    if st.session_state.results:
        res = st.session_state.results

        if res.get("success"):
            data = res["data"]
            validation = data.get("validation", {})
            status = validation.get("islem_durumu", "?")

            if status == "ONAYLANDI":
                st.markdown('<span class="badge-success">✓ ONAYLANDI</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="badge-error">✗ {status}</span>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            header = data.get("header", {})
            fin = data.get("financials", {})
            satici = header.get("satici", {})

            # Results
            st.markdown(f'''
            <div class="result-item">
                <div class="result-label">📋 Belge</div>
                <div class="result-value">{header.get('belge_turu', '-')} | {header.get('belge_no', '-')}</div>
                <div class="result-sub">Tarih: {header.get('belge_tarihi', '-')}</div>
            </div>
            ''', unsafe_allow_html=True)

            st.markdown(f'''
            <div class="result-item">
                <div class="result-label">🏢 Satici</div>
                <div class="result-value">{(satici.get('firma_adi', '-') or '-')[:28]}</div>
                <div class="result-sub">VKN: {satici.get('vkn', '-')}</div>
            </div>
            ''', unsafe_allow_html=True)

            curr = fin.get("para_birimi", "TRY")
            brut = fin.get("brut_tutar", 0) or 0
            kdv_oran = fin.get("kdv_orani", "-")
            kdv_tutar = fin.get("kdv_tutari", 0) or 0

            st.markdown(f'''
            <div class="result-item">
                <div class="result-label">💰 Tutar</div>
                <div class="result-value" style="color: #059669;">{brut:,.2f} {curr}</div>
                <div class="result-sub">KDV %{kdv_oran}: {kdv_tutar:,.2f} {curr}</div>
            </div>
            ''', unsafe_allow_html=True)

            # Tabs
            tab1, tab2, tab3 = st.tabs(["JSON", "Kalemler", "OCR"])

            with tab1:
                st.json(data)

            with tab2:
                items = data.get("items", [])
                if items:
                    for i, item in enumerate(items[:5], 1):
                        st.markdown(f"**{i}.** {item.get('aciklama', '-')[:40]} - **{item.get('tutar', '-')}**")
                else:
                    st.markdown("*Kalem bulunamadi*")

            with tab3:
                st.text_area("", res.get("raw_text", ""), height=120, label_visibility="collapsed")

        else:
            st.error(f"Hata: {res.get('error', 'Bilinmiyor')}")

    else:
        st.markdown('''
        <div class="card" style="text-align: center; padding: 2.5rem 1rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">📄</div>
            <div style="color: #64748b; font-size: 0.95rem;">Belge yukleyip islemi baslatin</div>
            <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.3rem;">Sonuclar burada gorunecek</div>
        </div>
        ''', unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown('<p style="text-align:center;color:#94a3b8;font-size:0.7rem;">FLO OCR Engine v4.0 | RG Labs © 2026</p>', unsafe_allow_html=True)
