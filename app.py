import os
import re
import sys
import time
import math
import pathlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Ensure repository root is in sys.path
BASE_DIR = pathlib.Path(__file__).parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import Hybrid Pipeline
try:
    from models.ensemble.predict import HybridInferencePipeline
    HYBRID_AVAILABLE = True
except Exception as e:
    HYBRID_AVAILABLE = False
    HYBRID_ERROR = str(e)

# Set Streamlit page config
st.set_page_config(
    page_title="Hybrid AI Text Detector | Dashboard",
    page_icon=":material/security:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS: injected once at startup ─────────────────────────────────────────────
def _inject_css():
    """Inject core theme CSS. Called once at render time."""
    bg         = "#090d16"
    surface    = "#111827"
    surface2   = "#0b1120"
    border_col = "rgba(255,255,255,0.07)"
    text_pri   = "#f1f5f9"
    text_sec   = "#94a3b8"
    text_card  = "#cbd5e1"
    sidebar_bg = "#0b1120"
    sidebar_br = "rgba(255,255,255,0.07)"
    track_bg   = "#1e293b"
    input_bg   = "#0b1120"
    input_br   = "#1e293b"
    sub_info   = "#64748b"

    st.markdown(f"""
<style>
/* ── 0. Google Fonts ──────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── 1. Core Theme ───────────────────────────────────────────── */
.stApp {{
    background-color: {bg} !important;
    color: {text_pri} !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

/* ── 2. Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    min-width: 280px !important;
    max-width: 290px !important;
    background-color: {sidebar_bg} !important;
    border-right: 1px solid {sidebar_br} !important;
}}
[data-testid="stSidebar"] .block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 1.5rem !important;
}}
/* Style the built-in expander to blend with sidebar */
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background: #0f172a !important;
    border: 1px solid {input_br} !important;
    border-radius: 8px !important;
    margin-top: 8px !important;
}}

/* ── 3. Main content padding ─────────────────────────────────── */
.main .block-container {{
    padding-top: 1.0rem !important;
    padding-bottom: 2rem !important;
    max-width: 1440px !important;
}}

/* ── 4. Dashboard Header ─────────────────────────────────────── */
.dash-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.0rem 1.5rem;
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 60%, #312e81 100%);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 20px -4px rgba(15,23,42,0.6);
    margin-bottom: 1.2rem;
}}
.dash-title {{
    font-size: 1.45rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin: 0;
}}
.dash-subtitle {{ font-size: 0.85rem; color: #94a3b8; margin-top: 2px; }}
.dash-tags    {{ font-size: 0.76rem; color: #818cf8; font-weight: 600; margin-top: 4px; letter-spacing: 0.02em; }}

/* Status pill */
.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 13px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}}
.status-online  {{ background: rgba(16,185,129,0.12); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }}
.status-offline {{ background: rgba(244,63,94,0.12);  color: #f43f5e; border: 1px solid rgba(244,63,94,0.3); }}
.status-dot {{ width:7px; height:7px; border-radius:50%; background-color:currentColor; }}

/* ── 5. Cards & Section Header Boxes ─────────────────────────── */
.card-header-box {{
    background: {surface};
    border: 1px solid {border_col};
    border-radius: 10px;
    padding: 0.7rem 1.1rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3),
                0 8px 16px -4px rgba(0,0,0,0.35);
    display: flex;
    align-items: center;
}}
.card-header-title {{
    font-size: 0.82rem;
    font-weight: 700;
    color: {text_card};
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0;
    line-height: 1;
    display: flex;
    align-items: center;
}}
.card-panel {{
    background: {surface};
    border: 1px solid {border_col};
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    /* elevated shadow */
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3),
                0 10px 24px -4px rgba(0,0,0,0.35);
    transition: box-shadow 0.2s ease;
}}
.card-panel:hover {{
    box-shadow: 0 6px 12px -2px rgba(0,0,0,0.4),
                0 16px 32px -6px rgba(0,0,0,0.45);
}}
.card-title {{
    font-size: 0.82rem;
    font-weight: 700;
    color: {text_card};
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.85rem;
}}

/* ── 6. Textarea & focus glow ────────────────────────── */
.stTextArea textarea {{
    background-color: {input_bg} !important;
    color: {text_pri} !important;
    border: 1px solid {input_br} !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    padding: 0.8rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}}
.stTextArea textarea:focus {{
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.25), 0 0 14px rgba(99,102,241,0.15) !important;
    outline: none !important;
}}

/* ── 7. Status chip / stats bar ─────────────────────── */
.input-stats-bar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-around;
    background: {surface2};
    border: 1px solid {input_br};
    border-radius: 20px;
    padding: 6px 16px;
    margin-top: 8px;
    margin-bottom: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    color: {text_sec};
    gap: 8px;
}}
.stat-item  {{ display: flex; align-items: center; gap: 5px; }}
.stat-value {{ color: {text_pri}; font-weight: 700; }}

/* ── 8. Buttons ─────────────────────────────────────── */
.stButton > button {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border-radius: 8px !important;
    height: 38px !important;
    transition: transform 0.12s ease, box-shadow 0.15s ease, opacity 0.15s ease !important;
}}
.stButton > button:hover  {{ transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important; }}
.stButton > button:active {{ transform: translateY(0px) !important; opacity: 0.88 !important; }}

/* ── 9. Result Banner ─────────────────────── */
.banner-human {{
    background: linear-gradient(135deg, rgba(6,78,59,0.45) 0%, rgba(4,120,87,0.22) 100%);
    border: 1.5px solid #10b981;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(16,185,129,0.12);
}}
.banner-ai {{
    background: linear-gradient(135deg, rgba(136,19,55,0.45) 0%, rgba(190,18,60,0.22) 100%);
    border: 1.5px solid #f43f5e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(244,63,94,0.12);
}}
.res-hdr-flex {{ display: flex; align-items: center; justify-content: space-between; }}

/* Verdict badge */
.res-badge-human {{
    color: #34d399;
    font-size: 1.55rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    line-height: 1.1;
    display: flex;
    align-items: center;
}}
.res-badge-ai {{
    color: #fb7185;
    font-size: 1.55rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    line-height: 1.1;
    display: flex;
    align-items: center;
}}
.res-badge-sub {{
    font-size: 0.76rem;
    color: {sub_info};
    font-weight: 500;
    margin-top: 3px;
    letter-spacing: 0.01em;
}}

/* Confidence % */
.res-score-huge {{
    font-size: 2.1rem;
    font-weight: 900;
    color: #ffffff;
    line-height: 1;
    letter-spacing: -0.03em;
}}
.res-score-sub {{
    font-size: 0.72rem;
    color: {sub_info};
    font-weight: 500;
    text-align: right;
    letter-spacing: 0.02em;
    margin-top: 3px;
}}

/* Gauge */
.gauge-wrapper   {{ margin-top: 1rem; opacity: 0.92; }}
.gauge-labels    {{ display: flex; justify-content: space-between; font-size: 0.70rem; font-weight: 600; color: {sub_info}; margin-bottom: 5px; }}
.gauge-track     {{ position: relative; height: 10px; background: {track_bg}; border-radius: 5px; overflow: hidden; }}
.gauge-fill-ai   {{ height: 100%; background: linear-gradient(90deg,#f43f5e,#fb7185); border-radius: 5px; }}
.gauge-fill-human{{ height: 100%; background: linear-gradient(90deg,#10b981,#34d399); border-radius: 5px; }}
.gauge-threshold-line {{ position:absolute; top:0; bottom:0; width:2px; background:#f59e0b; box-shadow:0 0 6px #f59e0b; z-index:2; }}
.gauge-sub-info  {{ display:flex; justify-content:space-between; font-size:0.70rem; color:{sub_info}; margin-top:4px; }}

/* ── 10. Model Cards ────── */
.model-row-card {{
    background: {surface2};
    border: 1px solid {input_br};
    border-radius: 8px;
    padding: 9px 12px;
    margin-bottom: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    transition: box-shadow 0.15s ease, transform 0.12s ease;
}}
.model-row-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.3); transform: translateY(-1px); }}
.model-row-header {{ display:flex; justify-content:space-between; align-items:center; font-size:0.82rem; font-weight:700; margin-bottom:5px; }}
.model-name    {{ color: {text_pri}; }}
.model-probs   {{ font-size: 0.78rem; font-weight: 600; }}
.prob-ai-tag   {{ color: #fb7185; }}
.prob-hu-tag   {{ color: #34d399; }}

.mini-track {{ height:6px; background:{track_bg}; border-radius:3px; overflow:hidden; }}
.mini-fill-safe   {{ height:100%; border-radius:3px; background: linear-gradient(90deg,#10b981,#34d399); }}
.mini-fill-warn   {{ height:100%; border-radius:3px; background: linear-gradient(90deg,#f59e0b,#fbbf24); }}
.mini-fill-danger {{ height:100%; border-radius:3px; background: linear-gradient(90deg,#f43f5e,#fb7185); }}

/* ── 11. Consensus Panel ─────────────────────────────────────── */
.consensus-badge-unanimous {{
    background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3);
    color: #34d399; border-radius: 8px; padding: 8px 12px;
    font-size: 0.84rem; font-weight: 700; margin-bottom: 10px;
    display: flex; align-items: center;
}}
.consensus-badge-split {{
    background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3);
    color: #fbbf24; border-radius: 8px; padding: 8px 12px;
    font-size: 0.84rem; font-weight: 700; margin-bottom: 10px;
    display: flex; align-items: center;
}}
.vote-item {{ display:flex; justify-content:space-between; align-items:center; font-size:0.78rem; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.04); }}
.vote-item:last-child {{ border-bottom:none; }}
.vote-model-lbl {{ color:{text_sec}; font-weight:600; }}
.vote-result-ai  {{ color:#fb7185; font-weight:700; }}
.vote-result-hu  {{ color:#34d399; font-weight:700; }}

/* ── 12. Stylometric Feature Grid ─────── */
.metric-grid-6 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }}
@media (max-width: 520px) {{
    .metric-grid-6 {{ grid-template-columns: repeat(2,1fr); }}
}}
.feat-card {{
    background: {surface2};
    border: 1px solid {input_br};
    border-radius: 8px;
    padding: 12px 8px 10px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.18);
    transition: box-shadow 0.18s ease, transform 0.14s ease, border-color 0.18s ease;
    cursor: default;
}}
.feat-card:hover {{
    box-shadow: 0 6px 18px rgba(56,189,248,0.18);
    transform: translateY(-2px);
    border-color: rgba(56,189,248,0.35);
}}
.feat-icon {{ font-size: 1.05rem; margin-bottom: 4px; line-height: 1; display: flex; justify-content: center; align-items: center; }}
.feat-val  {{ font-size:1.2rem; font-weight:800; color:#38bdf8; line-height:1; }}
.feat-lbl  {{ font-size:0.68rem; color:{text_sec}; font-weight:600; margin-top:4px; text-transform:uppercase; letter-spacing:0.03em; }}

/* ── 13. Sidebar Status Box ──────────────────────────────────── */
.sb-status-box  {{ background:#0f172a; border:1px solid {input_br}; border-radius:8px; padding:10px 12px; margin-top:4px; }}
.sb-status-title{{ font-size:0.76rem; font-weight:700; color:{text_card}; margin-bottom:6px; letter-spacing:0.04em; }}
.sb-item {{ font-size:0.73rem; color:{text_sec}; padding:2.5px 0; display:flex; align-items:center; justify-content:space-between; }}
.sb-active {{ color:#34d399; font-weight:700; }}

/* ── 14. Footer ──────────────────────────────────────────────── */
.app-footer {{ text-align:center; padding:1.2rem 0 0.5rem; font-size:0.74rem; color:#475569; border-top:1px solid rgba(255,255,255,0.05); margin-top:1.5rem; }}
</style>
""", unsafe_allow_html=True)

_inject_css()

# ----------------------------------------------------
# Paths & Cached Models
# ----------------------------------------------------

TFIDF_PATH = BASE_DIR / "outputs" / "tfidf" / "tfidf_vectorizer.pkl"
MODELS_DIR = BASE_DIR / "outputs" / "models"
MIN_WORDS = 200

# Cache Hybrid Inference Pipeline
@st.cache_resource(show_spinner="Loading Hybrid Transformer Models & Ensembles...")
def get_hybrid_pipeline():
    if HYBRID_AVAILABLE:
        try:
            return HybridInferencePipeline()
        except Exception as e:
            st.error(f"Error initializing Hybrid Inference Pipeline: {e}")
            return None
    return None

@st.cache_resource
def load_tfidf_vectorizer():
    if TFIDF_PATH.exists():
        try:
            return joblib.load(TFIDF_PATH)
        except Exception:
            return None
    return None

@st.cache_resource
def load_baseline_models():
    models = {}
    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Linear SVM": "linear_svm.joblib",
        "Multinomial Naive Bayes": "multinomial_naive_bayes.joblib",
        "Random Forest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib"
    }
    for name, filename in model_files.items():
        path = MODELS_DIR / filename
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception:
                pass
    return models

def count_words(text: str) -> int:
    """
    Consistently count words in text.
    Handles spaces, multiple spaces, newlines, tabs, and punctuation naturally.
    """
    if not text or not text.strip():
        return 0
    return len(text.strip().split())

def format_prediction_time(seconds) -> str:
    """Format duration into a human-readable prediction time string."""
    if seconds is None:
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1000.0:.0f} ms"
    return f"{seconds:.1f} sec"

def compute_text_metrics(text: str):
    """Compute linguistic features for text analysis UI."""
    word_count = count_words(text)
    words = re.findall(r'\b\w+\b', text)
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    chars = len(text)
    sentence_count = len(sentences) if len(sentences) > 0 else (1 if word_count > 0 else 0)

    avg_word_len = sum(len(w) for w in words) / len(words) if len(words) > 0 else 0
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0

    unique_words = set(w.lower() for w in words)
    ttr = len(unique_words) / word_count if word_count > 0 else 0

    return {
        "word_count": word_count,
        "char_count": chars,
        "sentence_count": sentence_count,
        "avg_word_len": avg_word_len,
        "avg_sentence_len": avg_sentence_len,
        "ttr": ttr
    }

# Load cached resources
hybrid_pipeline = get_hybrid_pipeline()
vectorizer = load_tfidf_vectorizer()
baseline_models = load_baseline_models()

# Streamlit Header
st.markdown("""
<div class="dash-header">
    <div>
        <div class="dash-title">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align:-3px;"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="15" x2="23" y2="15"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="15" x2="4" y2="15"/></svg>Hybrid AI Text Detector
        </div>
        <div class="dash-subtitle">Advanced Multi-Model AI Detection & Stylometric Analysis System</div>
        <div class="dash-tags">RoBERTa &bull; DeBERTa &bull; DistilBERT &bull; XGBoost &bull; Stacking Meta-Classifier</div>
    </div>
    <div>
        <span class="status-pill status-online">
            <span class="status-dot"></span> SYSTEM ONLINE
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Compact Sidebar Configuration
with st.sidebar:
    st.markdown("### Detection Engine")

    model_options = []
    if hybrid_pipeline is not None:
        model_options.append("Hybrid Stacking Meta-Classifier (Recommended)")
        model_options.append("Hybrid Weighted Ensemble")
        model_options.append("RoBERTa-base (Transformer)")
        model_options.append("DeBERTa-v3 (Transformer)")
        model_options.append("DistilBERT (Transformer)")

    for name in baseline_models.keys():
        model_options.append(f"{name} (TF-IDF)")

    if not model_options:
        model_options = ["Heuristic Analyzer"]

    selected_engine = st.selectbox(
        "Select Engine",
        options=model_options,
        index=0,
        help="Choose the model or ensemble architecture to evaluate text."
    )

    # Architecture Status & Diagnostic Model Signature
    with st.expander("Architecture Status & Signature", expanded=True):
        if hybrid_pipeline is not None:
            sig = getattr(hybrid_pipeline, "get_runtime_signature", lambda: {})()
            st.markdown(f"""
            <div class="sb-item"><span class="sb-active">● Hybrid Pipeline Active</span></div>
            <div class="sb-item"><span>Device:</span> <code>{hybrid_pipeline.device}</code></div>
            <div class="sb-item"><span>RoBERTa V2.1:</span> <code style="font-size:0.68rem;">{sig.get('roberta_path', 'Active')}</code></div>
            <div class="sb-item"><span>DeBERTa V2.1:</span> <code style="font-size:0.68rem;">{sig.get('deberta_path', 'Active')}</code></div>
            <div class="sb-item"><span>DistilBERT V2.1:</span> <code style="font-size:0.68rem;">{sig.get('distilbert_path', 'Active')}</code></div>
            <div class="sb-item"><span>XGBoost:</span> <code style="font-size:0.68rem;">{sig.get('xgboost_path', 'Active')}</code></div>
            <div class="sb-item"><span>Meta-Classifier:</span> <code style="font-size:0.68rem;">{sig.get('meta_classifier_path', 'Active')}</code></div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""<div class="sb-item"><span style="color:#f43f5e;">Hybrid Pipeline Offline</span></div>""", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="sb-item"><span>TF-IDF Vectorizer</span> <span>{'Active' if vectorizer else 'Offline'}</span></div>
            <div class="sb-item"><span>Baseline Models</span> <span>{len(baseline_models)} loaded</span></div>
        """, unsafe_allow_html=True)

# Main Dashboard Grid
col_left, col_right = st.columns([1.05, 1.0], gap="medium")

with col_left:
    st.markdown("""
    <div class="card-header-box">
        <div class="card-header-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#cbd5e1" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align:-1px;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>TEXT ANALYSIS
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "main_text_input" not in st.session_state:
        st.session_state["main_text_input"] = ""

    input_text = st.text_area(
        label="Paste academic text, essay, or generated content below:",
        height=270,
        placeholder=f"Paste text here to inspect for AI generation (minimum {MIN_WORDS} words required)...",
        key="main_text_input"
    )

    word_count = count_words(input_text)
    metrics = compute_text_metrics(input_text)
    is_valid_input = word_count >= MIN_WORDS

    # Check for text & engine modifications to prevent stale results
    current_matches_last = (
        "last_analyzed_text" in st.session_state and
        st.session_state.last_analyzed_text == input_text and
        "last_selected_engine" in st.session_state and
        st.session_state.last_selected_engine == selected_engine and
        is_valid_input
    )

    if not current_matches_last and "last_analysis_results" in st.session_state:
        del st.session_state["last_analysis_results"]

    col_btn1, col_btn2 = st.columns([1.2, 1])
    with col_btn1:
        analyze_clicked = st.button("Analyze Text", type="primary", icon=":material/analytics:", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("Clear Text", icon=":material/delete_outline:", use_container_width=True)
        if clear_clicked:
            st.session_state["main_text_input"] = ""
            if "last_analyzed_text" in st.session_state:
                del st.session_state["last_analyzed_text"]
            if "last_selected_engine" in st.session_state:
                del st.session_state["last_selected_engine"]
            if "last_analysis_results" in st.session_state:
                del st.session_state["last_analysis_results"]
            st.rerun()

    if analyze_clicked and not is_valid_input:
        st.warning(f"Please enter at least {MIN_WORDS} words for reliable detection.", icon=":material/warning:")

    # Execute inference immediately if requested
    should_run_inference = analyze_clicked and is_valid_input

    if should_run_inference:
        with st.spinner("Running Hybrid Detection Engine..."):
            t_start = time.perf_counter()
            ai_prob = 0.5
            hybrid_res = None
            model_breakdown = {}

            # Execute Backend Hybrid Pipeline
            if hybrid_pipeline is not None and ("Hybrid" in selected_engine or "RoBERTa" in selected_engine or "DeBERTa" in selected_engine or "DistilBERT" in selected_engine):
                hybrid_res = hybrid_pipeline.predict(input_text)

                # Auto-clear Streamlit cache if pipeline instance in memory is from an older version
                if not isinstance(hybrid_res, dict) or "model_breakdown" not in hybrid_res or "ai_probability" not in hybrid_res:
                    st.cache_resource.clear()
                    hybrid_pipeline = get_hybrid_pipeline()
                    hybrid_res = hybrid_pipeline.predict(input_text)

                # Extract model breakdown mapping safely
                if "model_breakdown" in hybrid_res:
                    mb = hybrid_res["model_breakdown"]
                    r_ai = float(mb.get("roberta", mb.get("RoBERTa-base", {})).get("ai_probability", 0.5))
                    d_ai = float(mb.get("deberta", mb.get("DeBERTa-v3", {})).get("ai_probability", 0.5))
                    db_ai = float(mb.get("distilbert", mb.get("DistilBERT", {})).get("ai_probability", 0.5))
                    xgb_ai = float(mb.get("xgboost", mb.get("XGBoost (Stylometrics)", {})).get("ai_probability", 0.5))
                elif "base_probs" in hybrid_res:
                    bp = hybrid_res["base_probs"]
                    r_ai = float(bp["roberta"][1])
                    d_ai = float(bp["deberta"][1])
                    db_ai = float(bp["distilbert"][1])
                    xgb_ai = float(bp["xgboost"][1])
                else:
                    r_ai = d_ai = db_ai = xgb_ai = 0.5

                model_breakdown["RoBERTa-base"] = r_ai
                model_breakdown["DeBERTa-v3"] = d_ai
                model_breakdown["DistilBERT"] = db_ai
                model_breakdown["XGBoost (Stylometrics)"] = xgb_ai

                if "Stacking Meta-Classifier" in selected_engine or "Hybrid Stacking" in selected_engine or "Hybrid" in selected_engine:
                    ai_prob = float(hybrid_res.get("ai_probability", hybrid_res.get("final_probs", [0.5, 0.5])[1]))
                    human_prob = float(hybrid_res.get("human_probability", 1.0 - ai_prob))
                    strategy_lbl = hybrid_res.get("strategy_used", "Stacking Meta-Classifier")
                elif "Weighted Ensemble" in selected_engine:
                    ai_prob = float(hybrid_res.get("weighted_voting_probs", [0.5, 0.5])[1])
                    human_prob = float(hybrid_res.get("weighted_voting_probs", [0.5, 0.5])[0])
                    strategy_lbl = "Weighted Voting Ensemble"
                elif "RoBERTa" in selected_engine:
                    ai_prob = r_ai
                    human_prob = 1.0 - r_ai
                    strategy_lbl = "RoBERTa-base Sequence Classifier"
                elif "DeBERTa" in selected_engine:
                    ai_prob = d_ai
                    human_prob = 1.0 - d_ai
                    strategy_lbl = "DeBERTa-v3 Sequence Classifier"
                elif "DistilBERT" in selected_engine:
                    ai_prob = db_ai
                    human_prob = 1.0 - db_ai
                    strategy_lbl = "DistilBERT Sequence Classifier"

            elif vectorizer is not None and baseline_models:
                vec = vectorizer.transform([input_text])
                clean_engine_name = selected_engine.replace(" (TF-IDF)", "")

                if clean_engine_name in baseline_models:
                    m = baseline_models[clean_engine_name]
                    if hasattr(m, "predict_proba"):
                        ai_prob = float(m.predict_proba(vec)[0][1])
                    elif hasattr(m, "decision_function"):
                        score = float(m.decision_function(vec)[0])
                        ai_prob = 1.0 / (1.0 + math.exp(-score))
                    strategy_lbl = f"{clean_engine_name} (TF-IDF)"

                for bname, bm in baseline_models.items():
                    try:
                        if hasattr(bm, "predict_proba"):
                            model_breakdown[bname] = float(bm.predict_proba(vec)[0][1])
                    except Exception:
                        pass
            else:
                strategy_lbl = "Heuristic Fallback"
                ai_prob = 0.5 + (0.15 * (1.0 - metrics['ttr'])) - (0.05 * (metrics['avg_sentence_len'] / 20))
                ai_prob = min(max(ai_prob, 0.05), 0.95)

            elapsed_sec = time.perf_counter() - t_start
            elapsed_ms = elapsed_sec * 1000.0
            pred_time_str = format_prediction_time(elapsed_sec)

            # Cache results in session state
            st.session_state.last_analyzed_text = input_text
            st.session_state.last_selected_engine = selected_engine
            st.session_state.last_analysis_results = {
                "ai_prob": ai_prob,
                "human_prob": 1.0 - ai_prob,
                "hybrid_res": hybrid_res,
                "model_breakdown": model_breakdown,
                "strategy_lbl": strategy_lbl,
                "elapsed_ms": elapsed_ms,
                "elapsed_sec": elapsed_sec,
                "pred_time_str": pred_time_str,
                "metrics": metrics
            }
            current_matches_last = True

    # Extract canonical prediction time display for rendering both panels
    if current_matches_last and "last_analysis_results" in st.session_state:
        res_data = st.session_state.last_analysis_results
        pred_time_display = res_data.get("pred_time_str", format_prediction_time(res_data.get("elapsed_sec")))
    else:
        pred_time_display = "—"

    # Render Live Word Counter & Stats Bar
    if is_valid_input:
        wc_html = f'<span style="color: #10b981; font-weight: 800;">Word count: {word_count} / {MIN_WORDS}</span>'
        bar_border = "1px solid rgba(16, 185, 129, 0.3)"
        bar_bg = "rgba(16, 185, 129, 0.08)"
    else:
        wc_html = f'<span style="color: #f59e0b; font-weight: 800;">Word count: {word_count} / {MIN_WORDS}</span>'
        bar_border = "1px solid rgba(245, 158, 11, 0.3)"
        bar_bg = "rgba(245, 158, 11, 0.08)"

    st.markdown(f"""
    <div class="input-stats-bar" style="border: {bar_border}; background: {bar_bg}; justify-content: space-between; padding: 9px 16px;">
        <div class="stat-item" style="font-size: 0.88rem;">{wc_html}</div>
        <div class="stat-item"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-1px; margin-right:4px;"><polyline points="4 7 4 4 20 4 20 7"/><line x1="12" y1="4" x2="12" y2="20"/></svg>Chars: <span class="stat-value">{metrics['char_count']}</span></div>
        <div class="stat-item"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-1px; margin-right:4px;"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/></svg>Sentences: <span class="stat-value">{metrics['sentence_count']}</span></div>
        <div class="stat-item"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-1px; margin-right:4px;"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>Prediction: <span class="stat-value">{pred_time_display}</span></div>
    </div>
    """, unsafe_allow_html=True)


# Right Column: Detection Results & Analytics
with col_right:
    if current_matches_last and "last_analysis_results" in st.session_state:
        # Retrieve cached results
        res = st.session_state.last_analysis_results
        ai_prob = res["ai_prob"]
        human_prob = res.get("human_prob", 1.0 - ai_prob)
        hybrid_res = res["hybrid_res"]
        model_breakdown = res["model_breakdown"]
        strategy_lbl = res["strategy_lbl"]
        elapsed_ms = res["elapsed_ms"]

        if hybrid_res and "prediction" in hybrid_res:
            is_ai = (hybrid_res["prediction"] == "AI Generated")
        else:
            is_ai = (ai_prob >= 0.50)

        # Render Primary Prediction Summary Card
        banner_class = "banner-ai" if is_ai else "banner-human"
        badge_class = "res-badge-ai" if is_ai else "res-badge-human"
        badge_text = "AI-GENERATED" if is_ai else "HUMAN-WRITTEN"
        primary_pct = f"{ai_prob * 100:.1f}%" if is_ai else f"{human_prob * 100:.1f}%"
        primary_label = "AI Probability" if is_ai else "Human Confidence"

        gauge_fill_class = "gauge-fill-ai" if is_ai else "gauge-fill-human"
        gauge_width_pct = min(max(ai_prob * 100, 0), 100)

        st.markdown(f"""
        <div class="{banner_class}">
            <div class="res-hdr-flex">
                <div>
                    <div class="{badge_class}">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>{badge_text}
                    </div>
                    <div class="res-badge-sub">{strategy_lbl}</div>
                </div>
                <div style="text-align:right;">
                    <div class="res-score-huge">{primary_pct}</div>
                    <div class="res-score-sub">{primary_label}</div>
                </div>
            </div>
            <div class="gauge-wrapper">
                <div class="gauge-labels">
                    <span>AI (0%)</span>
                    <span>50%</span>
                    <span>Human (100%)</span>
                </div>
                <div class="gauge-track">
                    <div class="{gauge_fill_class}" style="width:{gauge_width_pct:.1f}%;"></div>
                </div>
                <div class="gauge-sub-info">
                    <span>AI Prob: {ai_prob * 100:.1f}%</span>
                    <span>Human Prob: {human_prob * 100:.1f}%</span>
                    <span>Latency: {elapsed_ms:.1f} ms</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Breakdown & Consensus Side-by-Side
        sub_col1, sub_col2 = st.columns([1.1, 0.9])

        with sub_col1:
            st.markdown("""
            <div class="card-header-box">
                <div class="card-header-title">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#cbd5e1" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align:-1px;"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>MODEL BREAKDOWN
                </div>
            </div>
            """, unsafe_allow_html=True)

            if model_breakdown:
                for m_name, p_val in model_breakdown.items():
                    p_ai_pct = p_val * 100
                    p_hu_pct = (1.0 - p_val) * 100
                    if p_ai_pct < 30:
                        bar_cls = "mini-fill-safe"
                    elif p_ai_pct < 65:
                        bar_cls = "mini-fill-warn"
                    else:
                        bar_cls = "mini-fill-danger"
                    st.markdown(f"""
                    <div class="model-row-card">
                        <div class="model-row-header">
                            <span class="model-name">{m_name}</span>
                            <span class="model-probs">
                                <span class="prob-ai-tag">AI {p_ai_pct:.1f}%</span> | <span class="prob-hu-tag">Hu {p_hu_pct:.1f}%</span>
                            </span>
                        </div>
                        <div class="mini-track">
                            <div class="{bar_cls}" style="width:{p_ai_pct:.1f}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with sub_col2:
            st.markdown("""
            <div class="card-header-box">
                <div class="card-header-title">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#cbd5e1" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align:-1px;"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>MODEL CONSENSUS
                </div>
            </div>
            """, unsafe_allow_html=True)

            is_hybrid = ("Hybrid" in selected_engine or "Stacking" in selected_engine or "Weighted" in selected_engine)

            if is_hybrid and hybrid_res and "consensus" in hybrid_res:
                cons_data = hybrid_res["consensus"]
                cons_state = cons_data["state"]
                votes = cons_data["votes"]

                if "UNANIMOUS" in cons_state:
                    st.markdown(f"""<div class="consensus-badge-unanimous"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2.5" style="vertical-align:-2px; margin-right:5px;"><polyline points="20 6 9 17 4 12"/></svg> {cons_state}</div>""", unsafe_allow_html=True)
                elif "TIED" in cons_state:
                    st.markdown(f"""<div class="consensus-badge-split" style="background:rgba(245,158,11,0.1); border-color:rgba(245,158,11,0.3); color:#fbbf24;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2.5" style="vertical-align:-2px; margin-right:5px;"><circle cx="12" cy="12" r="10"/><line x1="8" y1="12" x2="16" y2="12"/></svg> {cons_state}</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="consensus-badge-split"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2.5" style="vertical-align:-2px; margin-right:5px;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg> {cons_state}</div>""", unsafe_allow_html=True)

                display_names = {
                    "roberta": "RoBERTa",
                    "deberta": "DeBERTa",
                    "distilbert": "DistilBERT",
                    "xgboost": "XGBoost"
                }

                st.markdown('<div style="margin-top: 6px;">', unsafe_allow_html=True)
                for m_key, m_disp in display_names.items():
                    v = votes.get(m_key, "Human")
                    v_class = "vote-result-ai" if v == "AI" else "vote-result-hu"
                    st.markdown(f"""
                    <div class="vote-item">
                        <span class="vote-model-lbl">{m_disp}</span>
                        <span class="{v_class}">{"AI" if v == "AI" else "Human"}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                clean_name = selected_engine.split("(")[0].strip()
                v_single = "AI" if is_ai else "Human"
                v_class_single = "vote-result-ai" if is_ai else "vote-result-hu"
                st.markdown(f"""<div class="consensus-badge-unanimous" style="background:rgba(99,102,241,0.1); border-color:rgba(99,102,241,0.3); color:#818cf8;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2.5" style="vertical-align:-2px; margin-right:5px;"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg> SINGLE MODEL RESULT (1/1 MODEL)</div>""", unsafe_allow_html=True)
                st.markdown('<div style="margin-top: 6px;">', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="vote-item">
                    <span class="vote-model-lbl">{clean_name}</span>
                    <span class="{v_class_single}">{v_single}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Stylometric Features Panel
        st.markdown("""
        <div class="card-header-box">
            <div class="card-header-title">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#cbd5e1" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align:-1px;"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>STYLOMETRIC & LINGUISTIC FEATURES
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-grid-6">
            <div class="feat-card">
                <div class="feat-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg></div>
                <div class="feat-val">{metrics['ttr']:.2f}</div>
                <div class="feat-lbl">Lexical Diversity</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 6H3"/><path d="M15 12H3"/><path d="M17 18H3"/></svg></div>
                <div class="feat-val">{metrics['avg_sentence_len']:.1f}</div>
                <div class="feat-lbl">Avg Sentence Len</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 7 4 4 20 4 20 7"/><line x1="9" y1="20" x2="15" y2="20"/><line x1="12" y1="4" x2="12" y2="20"/></svg></div>
                <div class="feat-val">{metrics['avg_word_len']:.1f}</div>
                <div class="feat-lbl">Avg Word Len</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg></div>
                <div class="feat-val">{word_count}</div>
                <div class="feat-lbl">Word Count</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg></div>
                <div class="feat-val">{metrics['sentence_count']}</div>
                <div class="feat-lbl">Sentences</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg></div>
                <div class="feat-val">{pred_time_display}</div>
                <div class="feat-lbl">Prediction Time</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if not is_valid_input:
            st.markdown(f"""
            <div class="card-panel" style="text-align: center; padding: 2.5rem 1.5rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem; color: #f59e0b; display: flex; justify-content: center;">
                    <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                </div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #f59e0b; margin-bottom: 0.4rem;">Minimum {MIN_WORDS} Words Required</div>
                <div style="font-size: 0.85rem; color: #94a3b8; max-width: 380px; margin: 0 auto;">Current input contains <b>{word_count}</b> / {MIN_WORDS} words. Please enter at least {MIN_WORDS} words to enable AI detection.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-panel" style="text-align: center; padding: 2.5rem 1.5rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem; color: #6366f1; display: flex; justify-content: center;">
                    <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
                </div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.4rem;">Ready for Analysis ({word_count} Words)</div>
                <div style="font-size: 0.85rem; color: #94a3b8; max-width: 380px; margin: 0 auto;">Input requirement satisfied. Click <b>Analyze Text</b> on the left panel to run the Hybrid Detection Engine.</div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="app-footer">
    <strong>Hybrid AI Text Detector</strong> &bull; Multi-Model Ensemble Detection & Stylometric System<br>
    RoBERTa &bull; DeBERTa &bull; DistilBERT &bull; XGBoost &bull; Stacking Meta-Classifier
</div>
""", unsafe_allow_html=True)

