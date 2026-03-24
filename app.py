"""
========================================================
  Customer Churn Predictor
  Author : Jibran Shahid
  Run    : streamlit run app.py
========================================================
"""

import streamlit as st

st.set_page_config(
    page_title="Churn Predictor · Jibran Shahid",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base ── */
    .stApp                          { background-color: #0f1117; }
    .main .block-container          { padding-top: 1.5rem; max-width: 1200px; }
    
    [data-testid="stSidebar"]        { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    /* ── KPI Cards ── */
    .kpi-card {
        background: #1c2333;
        border: 1px solid #2a2f3e;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        height: 100%;
    }
    .kpi-value  { font-size: 1.9rem; font-weight: 700; color: #00d4ff; margin: 0.2rem 0; line-height: 1.1; }
    .kpi-label  { font-size: 0.75rem; color: #8892a4; text-transform: uppercase; letter-spacing: 0.1em; margin: 0; }
    .kpi-sub    { font-size: 0.8rem;  color: #10b981; margin-top: 0.3rem; }

    /* ── Section titles ── */
    .section-title {
        font-size: 1rem; font-weight: 600; color: #e2e8f0;
        border-left: 3px solid #00d4ff;
        padding-left: 0.75rem;
        margin: 1.8rem 0 0.8rem 0;
    }

    /* ── Risk badge ── */
    .risk-high   { background:#2d1515; color:#ef4444; border:1px solid #ef4444; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.78rem; font-weight:600; }
    .risk-medium { background:#2d2415; color:#f59e0b; border:1px solid #f59e0b; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.78rem; font-weight:600; }
    .risk-low    { background:#15291e; color:#10b981; border:1px solid #10b981; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.78rem; font-weight:600; }

    /* ── Insight box ── */
    .insight-box {
        background: #1c2333;
        border: 1px solid #2a2f3e;
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        color: #c9d1d9;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ── Upload area ── */
    [data-testid="stFileUploader"] {
        background: #1c2333 !important;
        border: 2px dashed #2a2f3e !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border: 1px solid #2a2f3e; border-radius: 8px; }

    /* ── Radio nav pills ── */
    [data-testid="stRadio"] label { color: #c9d1d9 !important; }

    /* ── General text ── */
    p, li, span { color: #c9d1d9; }
    h1, h2, h3  { color: #e2e8f0; }

    /* ── Hide Streamlit chrome ── */
    footer    { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

from utils.model  import load_model_and_data
from pages.eda    import render_eda
from pages.predict import render_predict
from pages.insights import render_insights

# ── Train / load model ────────────────────────────────────────────────────────
model, scaler, feature_cols, df_raw, metrics = load_model_and_data()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📉 Customer Churn Predictor")
st.markdown(
    "<p style='color:#8892a4;margin-top:-0.5rem;margin-bottom:1.5rem;'>"
    "IBM Telco Dataset · XGBoost + Random Forest · Live CSV Predictions</p>",
    unsafe_allow_html=True
)

# ── Navigation ────────────────────────────────────────────────────────────────
col_nav1, col_nav2, col_nav3 = st.columns(3)
with col_nav1:
    if st.button("Model & EDA", use_container_width=True):
        st.session_state["page"] = "Model & EDA"
with col_nav2:
    if st.button("Live Predictor", use_container_width=True):
        st.session_state["page"] = "Live Predictor"
with col_nav3:
    if st.button("Insights", use_container_width=True):
        st.session_state["page"] = "Insights"

if "page" not in st.session_state:
    st.session_state["page"] = "Model & EDA"

page = st.session_state["page"]
st.markdown("---")

# ── Render page ───────────────────────────────────────────────────────────────
if page == "Model & EDA":
    render_eda(df_raw, metrics)
elif page == "Live Predictor":
    render_predict(model, scaler, feature_cols)
elif page == "Insights":
    render_insights(model, scaler, feature_cols, df_raw)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#4a5568;font-size:0.78rem;'>"
    "Built by <strong style='color:#00d4ff'>Jibran Shahid</strong> · "
    "<a href='https://jb00911.github.io/jibranshahid.github.io' style='color:#00d4ff;'>Portfolio</a> · "
    "</p>",
    unsafe_allow_html=True
)
