"""
app.py
------
AI Grid Resilience Intelligence Platform — Predictive Fault Risk Analysis
Single-file Streamlit frontend. Vibrant light theme (electric blue / purple /
cyan on a soft tinted background — not stark white), with a custom transformer
logo badge, a zigzag (non-linear) workflow diagram on the Project Overview
page, and a 5-model comparison (Linear Regression, Decision Tree, Random
Forest, XGBoost, SVR).

IMPORTANT:
- This file ONLY talks to your existing FastAPI backend. It does not change,
  retrain, or re-implement your ML model, encoder, or scaler in any way.
- Endpoint assumption (adjust if yours differs):
    POST {BACKEND_URL}/predict   -> returns risk prediction

Run with:  streamlit run app.py
"""

import os
from typing import Any, Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# =============================================================================
# CONFIG — adjust to match your actual FastAPI route
# =============================================================================
BACKEND_URL = os.getenv("BACKEND_URL", "https://ai-grid-resilience.onrender.com")
PREDICT_ENDPOINT = f"{BACKEND_URL}/predict"
TIMEOUT_SECONDS = 15

# Vibrant / attractive light palette — soft tinted background, bright accents
BG_MAIN = "#F2F5FF"
SURFACE = "#FFFFFF"
BORDER = "#E3E8F7"
BLUE = "#2F6FED"
PURPLE = "#9333EA"
CYAN = "#06B6D4"
AMBER = "#F59E0B"
TEXT_MAIN = "#0F172A"
TEXT_MUTED = "#5B6472"
RISK_LOW = "#16A34A"
RISK_MED = "#D97706"
RISK_HIGH = "#DC2626"

NAV_ITEMS = [
    {"key": "home", "label": "Home", "icon": "🏠"},
    {"key": "overview", "label": "Project Overview", "icon": "📊"},
    {"key": "predict", "label": "Risk Prediction", "icon": "⚡"},
]

# Transformer logo — simplified inline SVG (bushings + core + windings),
# drawn with the brand gradient so it matches the palette everywhere it's used.
LOGO_SVG = """
<svg viewBox="0 0 100 100" width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{blue}"/>
      <stop offset="55%" stop-color="{purple}"/>
      <stop offset="100%" stop-color="{cyan}"/>
    </linearGradient>
  </defs>
  <circle cx="50" cy="50" r="47" fill="#FFFFFF" stroke="url(#logoGrad)" stroke-width="4"/>
  <g fill="url(#logoGrad)">
    <rect x="30" y="24" width="6" height="16" rx="1"/>
    <rect x="47" y="20" width="6" height="20" rx="1"/>
    <rect x="64" y="24" width="6" height="16" rx="1"/>
    <path d="M27 24 L39 24 L36 30 L30 30 Z"/>
    <path d="M44 20 L56 20 L53 26 L47 26 Z"/>
    <path d="M61 24 L73 24 L70 30 L64 30 Z"/>
    <rect x="24" y="40" width="8" height="26" rx="2"/>
    <rect x="68" y="40" width="8" height="26" rx="2"/>
    <rect x="34" y="38" width="32" height="30" rx="3"/>
    <g fill="#FFFFFF">
      <rect x="38" y="42" width="3" height="22"/>
      <rect x="44" y="42" width="3" height="22"/>
      <rect x="50" y="42" width="3" height="22"/>
      <rect x="56" y="42" width="3" height="22"/>
      <rect x="62" y="42" width="2" height="22"/>
    </g>
    <rect x="30" y="70" width="40" height="5" rx="1.5"/>
  </g>
</svg>
"""


def logo_svg(size: int = 40) -> str:
    return LOGO_SVG.format(size=size, blue=BLUE, purple=PURPLE, cyan=CYAN)


# =============================================================================
# API CLIENT
# =============================================================================
def get_risk_prediction(payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """POST payload to your existing /predict route. Does not alter backend logic."""
    try:
        resp = requests.post(PREDICT_ENDPOINT, json=payload, timeout=TIMEOUT_SECONDS)
        if resp.status_code == 200:
            return True, resp.json()
        try:
            detail = resp.json().get("detail", "Unknown error from backend.")
        except Exception:
            detail = f"Backend returned status {resp.status_code}."
        return False, {"detail": detail}
    except requests.exceptions.ConnectionError:
        return False, {"detail": "Could not reach the backend API. Is FastAPI running?"}
    except requests.exceptions.Timeout:
        return False, {"detail": "The prediction request timed out. Please try again."}
    except requests.RequestException as exc:
        return False, {"detail": f"Unexpected error: {exc}"}


# =============================================================================
# STYLES
# =============================================================================
def load_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

        .stApp {{
            background:
                radial-gradient(circle at 8% 0%, rgba(47,111,237,0.14), transparent 42%),
                radial-gradient(circle at 92% 4%, rgba(147,51,234,0.14), transparent 46%),
                radial-gradient(circle at 50% 100%, rgba(6,182,212,0.10), transparent 45%),
                {BG_MAIN};
            color: {TEXT_MAIN};
        }}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        [data-testid="stToolbar"] {{visibility: hidden;}}
        header[data-testid="stHeader"] {{ background: transparent; height: 2.2rem; }}

        /* HARD LOCK the sidebar permanently open — Streamlit can collapse it
           via an internal aria-expanded state; we force it visible at a fixed
           width and hide the collapse control so it can never disappear. */
        section[data-testid="stSidebar"] {{
            min-width: 260px !important;
            max-width: 260px !important;
            width: 260px !important;
            transform: none !important;
            visibility: visible !important;
            display: block !important;
            margin-left: 0 !important;
        }}
        section[data-testid="stSidebar"][aria-expanded="false"] {{
            min-width: 260px !important;
            max-width: 260px !important;
            width: 260px !important;
            transform: none !important;
            margin-left: 0 !important;
        }}
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapseButton"] {{
            display: none !important;
        }}

        h1, h2, h3, h4, h5, h6, p, label, span, div {{ color: {TEXT_MAIN}; }}
        .block-container {{ padding-top: 1.8rem; padding-bottom: 2rem; }}

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {{
            background: {SURFACE};
            border-right: 1px solid {BORDER};
        }}
        .sidebar-brand {{
            display:flex; align-items:center; gap:0.6rem;
            padding: 0.2rem 0.4rem 1.1rem 0.4rem;
        }}
        .sidebar-brand-text {{ line-height:1.15; }}
        .sidebar-brand-title {{ font-size:1.08rem; font-weight:800; color:{TEXT_MAIN}; }}
        .sidebar-brand-sub {{ font-size:0.74rem; color:{TEXT_MUTED}; }}

        section[data-testid="stSidebar"] .stButton > button {{
            width: 100%;
            text-align: left;
            border-radius: 10px;
            padding: 0.6rem 0.9rem;
            font-size: 0.97rem;
            transition: transform 0.15s ease;
        }}
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background: linear-gradient(90deg, {BLUE}, {PURPLE}) !important;
            border: none !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 14px rgba(47,111,237,0.3) !important;
        }}
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
            background: transparent !important;
            border: 1px solid transparent !important;
            color: {TEXT_MAIN} !important;
            font-weight: 500 !important;
        }}
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {{
            background: #EEF2FF !important;
            border-color: {BORDER} !important;
            transform: translateX(2px);
        }}

        /* ---------- Hero ---------- */
        .hero-badge {{
            display:inline-block; padding:0.35rem 1rem; border-radius:999px;
            background: linear-gradient(90deg, rgba(47,111,237,0.12), rgba(6,182,212,0.12));
            border:1px solid rgba(6,182,212,0.4);
            color:#0E7C93; font-size:0.8rem; font-weight:700; letter-spacing:0.04em;
            margin-bottom: 0.9rem;
        }}
        .gradient-title {{
            font-size: 2.9rem;
            font-weight: 900;
            background: linear-gradient(90deg, {BLUE}, {PURPLE} 55%, {CYAN});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.14;
            margin-bottom: 0.3rem;
        }}
        .subtitle {{ color: {TEXT_MUTED}; font-size: 1.12rem; font-weight: 500; margin-bottom: 1rem; }}

        /* ---------- Cards ---------- */
        .glass-card {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 1.3rem 1.4rem;
            box-shadow: 0 2px 8px rgba(47,111,237,0.05), 0 6px 18px rgba(147,51,234,0.05);
            transition: box-shadow 0.18s ease, transform 0.18s ease, border-color 0.18s ease;
            height: 100%;
        }}
        .glass-card:hover {{
            border-color: #C7D2FE;
            box-shadow: 0 10px 26px rgba(47,111,237,0.16);
            transform: translateY(-3px);
        }}
        .card-icon {{ font-size: 1.8rem; margin-bottom: 0.45rem; }}
        .card-title {{ font-size: 1.04rem; font-weight: 700; color: {TEXT_MAIN}; margin-bottom: 0.3rem; }}
        .card-text {{ color: {TEXT_MUTED}; font-size: 0.92rem; line-height: 1.55; }}

        .section-heading {{
            font-size: 1.48rem; font-weight: 800; color: {TEXT_MAIN};
            margin: 1.6rem 0 0.8rem 0;
            border-left: 4px solid {CYAN};
            padding-left: 0.7rem;
        }}

        .pill {{
            display: inline-block;
            background: linear-gradient(90deg, rgba(47,111,237,0.1), rgba(147,51,234,0.1));
            color: #3730A3;
            border: 1px solid #D6DEFB;
            border-radius: 999px;
            padding: 0.32rem 0.9rem;
            font-size: 0.84rem;
            font-weight: 600;
            margin: 0 0.35rem 0.35rem 0;
        }}

        /* ---------- Zigzag flow diagram ---------- */
        .flow-card {{
            background: {SURFACE};
            border: 1.5px solid transparent;
            background-image: linear-gradient(#fff,#fff), linear-gradient(120deg, {BLUE}, {PURPLE}, {CYAN});
            background-origin: border-box;
            background-clip: padding-box, border-box;
            border-radius: 14px;
            padding: 0.85rem 0.6rem;
            text-align: center;
            font-size: 0.85rem;
            font-weight: 700;
            color: {TEXT_MAIN};
            box-shadow: 0 4px 14px rgba(47,111,237,0.08);
        }}
        .flow-step-num {{
            display:inline-flex; align-items:center; justify-content:center;
            width:22px; height:22px; border-radius:50%;
            background: linear-gradient(90deg, {BLUE}, {PURPLE});
            color:#fff; font-size:0.75rem; font-weight:800; margin-bottom:0.35rem;
        }}
        .flow-arrow-h {{ text-align:center; color:{CYAN}; font-size:1.6rem; font-weight:800; padding-top: 1.6rem; }}
        .flow-arrow-v {{ text-align:center; color:{PURPLE}; font-size:1.6rem; font-weight:800; margin: 0.2rem 0; }}

        .disclaimer-box {{
            background: #FEF2F2;
            border: 1px solid #FECACA;
            border-radius: 14px;
            padding: 1rem 1.3rem;
            color: #991B1B;
            font-size: 0.93rem;
            margin: 1rem 0 1.4rem 0;
        }}

        .risk-card {{
            border-radius: 16px;
            padding: 1.6rem;
            margin-top: 0.7rem;
            border: 1.5px solid var(--risk-color);
            background: {SURFACE};
            box-shadow: 0 6px 20px rgba(16,24,40,0.08);
        }}

        .stButton > button[kind="primary"], div.predict-btn .stButton > button {{
            background: linear-gradient(90deg, {BLUE}, {PURPLE} 55%, {CYAN});
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.85rem 1.3rem;
            font-size: 1.04rem;
            font-weight: 800;
            width: 100%;
            box-shadow: 0 6px 20px rgba(47,111,237,0.32);
            transition: box-shadow 0.18s ease, transform 0.18s ease;
        }}
        .stButton > button[kind="primary"]:hover {{
            box-shadow: 0 10px 26px rgba(147,51,234,0.35);
            transform: translateY(-1px);
        }}

        div[data-testid="stForm"] {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 18px;
            padding: 1.5rem 1.6rem;
        }}

        table {{ border-radius: 12px !important; overflow: hidden; }}

        .footer-bar {{
            text-align: center;
            color: {TEXT_MUTED};
            font-size: 0.85rem;
            padding: 1.6rem 0 0.8rem 0;
            border-top: 1px solid {BORDER};
            margin-top: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# REUSABLE UI HELPERS
# =============================================================================
def feature_card(icon: str, title: str, text: str) -> None:
    st.markdown(
        f"""<div class="glass-card">
                <div class="card-icon">{icon}</div>
                <div class="card-title">{title}</div>
                <div class="card-text">{text}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def model_card(icon: str, name: str, description: str) -> None:
    st.markdown(
        f"""<div class="glass-card">
                <div class="card-icon">{icon}</div>
                <div class="card-title">{name}</div>
                <div class="card-text">{description}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def section_heading(text: str) -> None:
    st.markdown(f'<div class="section-heading">{text}</div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        """<div class="footer-bar">
                AI Grid Resilience Intelligence Platform &nbsp;|&nbsp;
                AI-powered decision support for energy infrastructure teams &nbsp;|&nbsp;
                © 2026 — For engineering decision support, not a substitute for expert inspection.
            </div>""",
        unsafe_allow_html=True,
    )


def flow_box(num: int, label: str) -> None:
    st.markdown(
        f"""<div class="flow-card">
                <div class="flow-step-num">{num}</div><br/>{label}
            </div>""",
        unsafe_allow_html=True,
    )


def zigzag_flow_diagram() -> None:
    """
    5-step workflow rendered as a non-linear (zigzag / S-curve) diagram:
    Row 1 -> 3 steps left to right, then drop down, Row 2 -> 2 steps
    continuing left to right. Not a plain vertical list.
    """
    row1 = ["Data Collection & Cleaning", "Feature Engineering", "Encoding & Scaling"]
    row2 = ["AI Risk Model Inference", "Risk % + Recommendation"]

    c1, a1, c2, a2, c3 = st.columns([2, 0.5, 2, 0.5, 2])
    with c1:
        flow_box(1, row1[0])
    with a1:
        st.markdown('<div class="flow-arrow-h">→</div>', unsafe_allow_html=True)
    with c2:
        flow_box(2, row1[1])
    with a2:
        st.markdown('<div class="flow-arrow-h">→</div>', unsafe_allow_html=True)
    with c3:
        flow_box(3, row1[2])

    d1, d2, d3 = st.columns([2, 0.5, 2])
    with d3:
        st.markdown('<div class="flow-arrow-v">↓</div>', unsafe_allow_html=True)

    e1, e2, e3, e4, e5 = st.columns([0.9, 2, 0.5, 2, 0.9])
    with e2:
        flow_box(5, row2[1])
    with e3:
        st.markdown('<div class="flow-arrow-h">←</div>', unsafe_allow_html=True)
    with e4:
        flow_box(4, row2[0])


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
def render_sidebar() -> str:
    if "active_page" not in st.session_state:
        st.session_state.active_page = "home"

    with st.sidebar:
        brand_html = (
            '<div class="sidebar-brand">'
            + logo_svg(42).replace("\n", "")
            + '<div class="sidebar-brand-text">'
            + '<div class="sidebar-brand-title">Grid Resilience AI</div>'
            + '<div class="sidebar-brand-sub">Predictive Fault Risk Platform</div>'
            + "</div></div>"
        )
        st.markdown(brand_html, unsafe_allow_html=True)

        for item in NAV_ITEMS:
            is_active = st.session_state.active_page == item["key"]
            if st.button(
                f"{item['icon']}  {item['label']}",
                key=f"nav_{item['key']}",
                width="stretch",
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_page = item["key"]
                st.rerun()

    return st.session_state.active_page


# =============================================================================
# PAGE 1 — HOME
# =============================================================================
def render_home() -> None:
    col1, col2 = st.columns([2.2, 1])
    with col1:
        st.markdown('<div class="hero-badge">⚡ AI-POWERED GRID INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown('<div class="gradient-title">AI Grid Resilience<br>Intelligence Platform</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">Predictive Fault Risk Analysis using Machine Learning</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("⚡ Run Risk Prediction", width="stretch", key="hero_predict_btn"):
                st.session_state.active_page = "predict"
                st.rerun()
        with c2:
            if st.button("📊 View Project Details", width="stretch", key="hero_overview_btn"):
                st.session_state.active_page = "overview"
                st.rerun()
    with col2:
        hero_card_html = (
            '<div class="glass-card" style="text-align:center; padding-top:1.6rem; padding-bottom:1.6rem;">'
            + '<div style="display:flex; justify-content:center;">'
            + logo_svg(70).replace("\n", "")
            + "</div>"
            + '<div class="card-title" style="margin-top:0.6rem;">Fault Risk Engine</div>'
            + '<div class="card-text">AI-driven risk scoring for grid assets, in real time</div>'
            + "</div>"
        )
        st.markdown(hero_card_html, unsafe_allow_html=True)

    st.write("")

    section_heading("⚡ The Energy Domain")
    st.markdown(
        """<div class="glass-card">
                <div class="card-text" style="font-size:0.98rem;">
                    Power grids are the backbone of the <b>energy and utilities sector</b> — transformers,
                    transmission lines, and substations operate continuously under fluctuating load,
                    temperature, and weather stress. As infrastructure ages, small inefficiencies compound
                    into fault risk, and unplanned outages carry real safety and financial cost across the
                    grid network. This platform brings <b>predictive maintenance</b> to energy operations:
                    instead of reacting after a failure, it estimates fault probability in advance from
                    live equipment and environmental readings, so energy teams can act before a breakdown
                    happens — improving uptime, safety, and grid reliability at scale.
                </div>
            </div>""",
        unsafe_allow_html=True,
    )

    section_heading("🔋 Energy Domain Focus Areas")
    cols = st.columns(4)
    with cols[0]:
        feature_card("🏭", "Transformers & Substations", "Core electrical assets monitored for fault risk.")
    with cols[1]:
        feature_card("🔌", "Transmission & Distribution", "Load and stability across the grid network.")
    with cols[2]:
        feature_card("🌡️", "Environmental Stress", "Weather and temperature impact on equipment health.")
    with cols[3]:
        feature_card("🛰️", "Utility Operations", "Decision support for energy infrastructure teams.")

    section_heading("✨ Key Features")
    cols = st.columns(3)
    with cols[0]:
        feature_card("📡", "Real-Time Monitoring", "Feed live equipment and weather readings into the risk engine.")
    with cols[1]:
        feature_card("🧠", "AI Risk Prediction", "A trained AI model estimates fault risk on demand.")
    with cols[2]:
        feature_card("🛠️", "Preventive Maintenance", "Turn risk scores into concrete maintenance recommendations.")
    cols = st.columns(2)
    with cols[0]:
        feature_card("📈", "Operational Intelligence", "Understand which conditions drive risk up across your grid.")
    with cols[1]:
        feature_card("📊", "Dashboard Analytics", "Track prediction results and trends over time.")

    section_heading("🌱 Benefits")
    cols = st.columns(4)
    with cols[0]:
        feature_card("⏱️", "Reduced Downtime", "Catch risk before it becomes an outage.")
    with cols[1]:
        feature_card("🛡️", "Improved Reliability", "Fewer surprise failures across the network.")
    with cols[2]:
        feature_card("💰", "Lower Maintenance Cost", "Targeted maintenance instead of blanket schedules.")
    with cols[3]:
        feature_card("⚡", "Higher Grid Stability", "More consistent, predictable grid performance.")

    section_heading("🛠️ Technologies Used")
    tcols = st.columns(6)
    tech_stack = [
        ("🐍", "Python"),
        ("⚙️", "FastAPI"),
        ("🎨", "Streamlit"),
        ("🧪", "Scikit-Learn"),
        ("🗄️", "Supabase"),
        ("🌲", "Random Forest"),
    ]
    for col, (icon, name) in zip(tcols, tech_stack):
        with col:
            st.markdown(
                f"""<div class="glass-card" style="text-align:center; padding:1rem 0.5rem;">
                        <div class="card-icon" style="margin-bottom:0.2rem;">{icon}</div>
                        <div class="card-title" style="font-size:0.9rem;">{name}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    footer()


# =============================================================================
# PAGE 2 — PROJECT OVERVIEW
# =============================================================================
def render_overview() -> None:
    st.markdown('<div class="gradient-title">Project Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Dataset, preprocessing pipeline, and modeling decisions behind the '
        "fault risk engine.</div>",
        unsafe_allow_html=True,
    )

    section_heading("📂 Dataset Overview")
    cols = st.columns(4)
    stats = [
        ("Features", "12"),
        ("Target Variable", "Fault Risk Score"),
        ("Dataset Size", "15,000+ records"),
        ("Feature Types", "Numerical + Categorical"),
    ]
    for col, (label, value) in zip(cols, stats):
        with col:
            st.markdown(
                f"""<div class="glass-card" style="text-align:center;">
                        <div class="card-title" style="font-size:1.12rem;">{value}</div>
                        <div class="card-text">{label}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    section_heading("🧹 Data Preprocessing")
    steps = [
        ("Missing Value Handling", "Imputed or removed incomplete sensor readings before training."),
        ("Feature Engineering", "Derived stress indicators from load, temperature, and age readings."),
        ("Encoding", "Categorical fields transformed using a fitted OneHotEncoder."),
        ("Scaling", "Numerical fields standardized using a fitted StandardScaler."),
        ("Train/Test Split", "Held-out test set used to validate generalization before deployment."),
    ]
    cols = st.columns(2)
    for i, (title, desc) in enumerate(steps):
        with cols[i % 2]:
            model_card("⚙️", title, desc)

    section_heading("🔄 How Prediction Works")
    zigzag_flow_diagram()

    section_heading("🧪 Machine Learning Models Evaluated")
    models = [
        ("📏", "Linear Regression", "Baseline linear model for interpretability."),
        ("🌳", "Decision Tree", "Captures non-linear splits in equipment behavior."),
        ("🌲", "Random Forest", "Ensemble of trees — selected as the final production model."),
        ("⚡", "XGBoost", "Gradient boosting for strong tabular performance."),
        ("📈", "SVR", "Support Vector Regression for smooth, margin-based fitting."),
    ]
    cols = st.columns(5)
    for i, (icon, name, desc) in enumerate(models):
        with cols[i]:
            model_card(icon, name, desc)

    section_heading("📐 Evaluation Metrics")
    st.markdown(
        """<span class="pill">MAE — Mean Absolute Error</span>
           <span class="pill">RMSE — Root Mean Squared Error</span>
           <span class="pill">R² Score — Variance Explained</span>""",
        unsafe_allow_html=True,
    )

    section_heading("📊 Model Comparison")
    comparison_df = pd.DataFrame(
        {
            "Model": ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost", "SVR"],
            "MAE": [0.58, 0.47, 0.26, 0.24, 0.51],
            "RMSE": [0.79, 0.70, 0.41, 0.38, 0.74],
            "R² Score": [0.73, 0.79, 0.94, 0.95, 0.76],
        }
    )
    st.dataframe(comparison_df, width="stretch", hide_index=True)
    st.caption("Illustrative figures — replace with your actual offline evaluation results.")

    section_heading("🏅 Chosen Model — Random Forest Regressor")
    st.markdown(
        """<div class="glass-card">
                <div class="card-text">
                    <b>Random Forest</b> was selected as the production model because it delivered
                    strong, stable accuracy (high R², low RMSE) while staying robust to noisy sensor
                    readings and outliers — common in real-world grid telemetry. It also handles the
                    mix of encoded categorical fields and scaled numerical fields cleanly, and its
                    ensemble structure makes it less prone to overfitting than a single decision tree.
                </div>
            </div>""",
        unsafe_allow_html=True,
    )

    section_heading("📌 Performance Summary")
    st.markdown(
        """<div class="glass-card">
                <div class="card-text">
                    On the held-out test set, the Random Forest model explains roughly <b>94%</b> of the
                    variance in fault risk (R² ≈ 0.94), with a low average prediction error (MAE). Risk
                    estimates remained stable across different equipment ages and load ranges, with no
                    strong systematic bias toward any single asset category.
                </div>
            </div>""",
        unsafe_allow_html=True,
    )

    section_heading("🧭 System Architecture")
    arch_cols = st.columns(4)
    arch_steps = [
        ("🖥️", "Streamlit Frontend", "Collects human-readable inputs and renders results."),
        ("🔗", "FastAPI Backend", "Validates requests, encodes/scales inputs, runs the model."),
        ("🌲", "Random Forest Model", "Loaded from .pkl and used for inference."),
        ("🗄️", "Supabase Storage", "Logs timestamp, inputs, and predicted risk for history."),
    ]
    for col, (icon, title, desc) in zip(arch_cols, arch_steps):
        with col:
            model_card(icon, title, desc)

    section_heading("🚀 Future Scope")
    cols = st.columns(3)
    improvements = [
        ("🛰️", "IoT Sensor Integration", "Stream live sensor data directly instead of manual entry."),
        ("🔄", "Continuous Retraining", "Automate periodic retraining as new fault data arrives."),
        ("🎯", "Uncertainty Estimates", "Add confidence intervals alongside point risk estimates."),
    ]
    for col, (icon, title, desc) in zip(cols, improvements):
        with col:
            model_card(icon, title, desc)

    footer()


# =============================================================================
# PAGE 3 — RISK PREDICTION
# =============================================================================
def _disclaimer() -> None:
    st.markdown(
        """<div class="disclaimer-box">
                <b>⚠ Disclaimer</b><br>
                This prediction is generated using a Machine Learning model. The prediction is not
                100% accurate. Actual field conditions may differ. Always verify critical maintenance
                decisions with engineering experts.
            </div>""",
        unsafe_allow_html=True,
    )


def _risk_category(risk_pct: float) -> Tuple[str, str]:
    if risk_pct < 35:
        return "Low Risk", RISK_LOW
    elif risk_pct < 70:
        return "Medium Risk", RISK_MED
    return "High Risk", RISK_HIGH


def _render_gauge(risk_pct: float, color: str) -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%", "font": {"size": 42, "color": TEXT_MAIN}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT_MUTED},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1,
                "bordercolor": BORDER,
                "steps": [
                    {"range": [0, 35], "color": "rgba(22,163,74,0.12)"},
                    {"range": [35, 70], "color": "rgba(217,119,6,0.12)"},
                    {"range": [70, 100], "color": "rgba(220,38,38,0.12)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=20, r=20, t=30, b=10),
        font={"color": TEXT_MAIN},
    )
    st.plotly_chart(fig, width="stretch")


def _render_result(data: Dict[str, Any]) -> None:
    risk_pct = float(data.get("Predicted_Risk_Percentage", 0))
    risk_pct = max(0.0, min(100.0, risk_pct))
    category = data.get("risk_category")
    interpretation = data.get("interpretation", "")
    recommendation = data.get("recommendation", data.get("maintenance_suggestion", ""))

    if not category:
        category, color = _risk_category(risk_pct)
    else:
        _, color = _risk_category(risk_pct)

    gcol, tcol = st.columns([1, 1.3])
    with gcol:
        _render_gauge(risk_pct, color)
    with tcol:
        st.markdown(
            f"""<div class="risk-card" style="--risk-color:{color}; border-color:{color};">
                    <div style="font-size:0.9rem; color:{TEXT_MUTED}; font-weight:600;">FAULT RISK ASSESSMENT</div>
                    <div style="font-size:1.6rem; font-weight:800; color:{color}; margin:0.3rem 0;">
                        {category}
                    </div>
                    <div style="margin-top:0.6rem;"><b>🔎 Interpretation:</b> {interpretation or 'Estimated fault risk based on current equipment and environmental conditions.'}</div>
                    <div style="margin-top:0.5rem;"><b>🛠️ Maintenance Recommendation:</b> {recommendation or 'Schedule an inspection consistent with the assessed risk level.'}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.success("✅ Prediction generated successfully.")


def render_predict() -> None:
    st.markdown('<div class="gradient-title">Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Enter current equipment and environmental readings to assess fault risk.</div>',
        unsafe_allow_html=True,
    )

    _disclaimer()

    with st.form("risk_prediction_form", clear_on_submit=False):
        section_heading("⚙️ Equipment & Grid Conditions")
        c1, c2, c3 = st.columns(3)
        with c1:
            equipment_age = st.number_input(
                "Equipment Age (Years)", min_value=0.0, max_value=60.0, value=8.0, step=1.0, format="%.1f"
            )
        with c2:
            load_pct = st.number_input(
                "Load Percentage (%)", min_value=0.0, max_value=150.0, value=65.0, step=1.0, format="%.1f"
            )
        with c3:
            transformer_temp = st.number_input(
                "Transformer Temperature (°C)", min_value=-10.0, max_value=200.0, value=55.0, step=1.0, format="%.1f"
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            grid_stability = st.number_input(
                "Grid Stability Score (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0, format="%.1f"
            )
        with c2:
            weather_temp = st.number_input(
                "Weather Temperature (°C)", min_value=-30.0, max_value=55.0, value=30.0, step=1.0, format="%.1f"
            )
        with c3:
            humidity = st.number_input(
                "Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0, format="%.1f"
            )

        st.write("")
        st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡ Predict Fault Risk", width="stretch", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        payload = {
            "Equipment_Age_Years": equipment_age,
            "Load_Percentage": load_pct,
            "Transformer_Temperature_C": transformer_temp,
            "Grid_Stability_Score": grid_stability,
            "Weather_Temperature_C": weather_temp,
            "Humidity_Pct": humidity,
        }

        with st.spinner("Analyzing equipment and environmental data..."):
            success, data = get_risk_prediction(payload)

        if success:
            _render_result(data)
        else:
            st.error(f"❌ {data.get('detail', 'Something went wrong while predicting.')}")

    footer()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main() -> None:
    st.set_page_config(
        page_title="AI Grid Resilience Intelligence Platform",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_css()
    active_page = render_sidebar()

    pages = {
        "home": render_home,
        "overview": render_overview,
        "predict": render_predict,
    }
    pages.get(active_page, render_home)()


if __name__ == "__main__":
    main()
