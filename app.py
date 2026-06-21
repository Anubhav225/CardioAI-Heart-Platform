import os
import pickle
import sqlite3
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from PIL import Image
from sklearn.metrics import confusion_matrix
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph

# ── Local helper modules (Groq voice/chat + multi-model comparison) ──
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules import groq_client
from modules import multi_model

# =============================================================
# PAGE SETUP
# =============================================================
st.set_page_config(
    page_title="CardioAI — Heart Intelligence Platform",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================
# LIGHT PROFESSIONAL THEME
# =============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700&display=swap');

:root {
    --bg:          #f0f4f8;
    --surface:     #ffffff;
    --surface2:    #f8fafc;
    --red:         #e63946;
    --red-light:   #fde8ea;
    --red-dark:    #b02030;
    --teal:        #0077b6;
    --teal-light:  #e0f0fa;
    --green:       #2a9d5c;
    --green-light: #e6f7ee;
    --amber:       #e07b39;
    --amber-light: #fef3e8;
    --text:        #0f172a;
    --text2:       #475569;
    --text3:       #94a3b8;
    --border:      #e2e8f0;
    --shadow:      0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-md:   0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.06);
    --radius:      10px;
    --radius-lg:   16px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem !important;
    max-width: 1400px !important;
    background: var(--bg) !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.05) !important;
}

.sidebar-logo {
    background: linear-gradient(135deg, #e63946 0%, #9b1c27 100%);
    padding: 28px 20px 22px 20px;
    margin: -1rem -1rem 1.5rem -1rem;
    text-align: center;
}
.sidebar-logo h1 {
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.7rem !important;
    color: white !important;
    margin: 0 !important;
    letter-spacing: -0.5px;
}
.sidebar-logo p {
    font-size: 0.68rem !important;
    color: rgba(255,255,255,0.8) !important;
    margin: 5px 0 0 0 !important;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.sidebar-section {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text3) !important;
    padding: 0 0 6px 4px !important;
    margin-top: 16px !important;
}

[data-testid="stSidebar"] .stRadio > label { display: none !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 9px 12px !important;
    transition: all 0.15s ease !important;
    color: var(--text2) !important;
    font-size: 0.88rem !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: var(--red-light) !important;
    color: var(--red) !important;
}

.sidebar-stat {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 9px 14px;
    margin: 4px 0;
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
}
.sidebar-stat .k { color: var(--text3); }
.sidebar-stat .v { color: var(--text); font-weight: 600; }

/* ── HERO ── */
.page-hero {
    background: linear-gradient(135deg, #fff 0%, #fde8ea 100%);
    border: 1px solid #f5c2c7;
    border-left: 5px solid var(--red);
    border-radius: var(--radius-lg);
    padding: 26px 30px;
    margin-bottom: 24px;
    box-shadow: var(--shadow);
}
.page-hero h2 {
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.6rem !important;
    color: var(--text) !important;
    margin: 0 0 5px 0 !important;
}
.page-hero p {
    color: var(--text2) !important;
    font-size: 0.88rem !important;
    margin: 0 !important;
}

/* ── METRIC CARDS ── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    box-shadow: var(--shadow);
    transition: transform 0.15s, box-shadow 0.15s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
.metric-card .label {
    font-size: 0.67rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: var(--text3); margin-bottom: 8px;
}
.metric-card .value { font-size: 1.9rem; font-weight: 700; color: var(--text); line-height: 1; }
.metric-card .sub   { font-size: 0.76rem; color: var(--text2); margin-top: 4px; }
.metric-card.red   { border-top: 3px solid var(--red);   background: linear-gradient(135deg,#fff,#fde8ea); }
.metric-card.teal  { border-top: 3px solid var(--teal);  background: linear-gradient(135deg,#fff,#e0f0fa); }
.metric-card.green { border-top: 3px solid var(--green); background: linear-gradient(135deg,#fff,#e6f7ee); }
.metric-card.amber { border-top: 3px solid var(--amber); background: linear-gradient(135deg,#fff,#fef3e8); }

/* ── SECTION HEADER ── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 26px 0 14px 0; padding-bottom: 10px;
    border-bottom: 2px solid var(--border);
}
.section-header span { font-size: 0.95rem; font-weight: 600; color: var(--text); }

/* ── RESULT BANNERS ── */
.result-banner {
    border-radius: var(--radius); padding: 20px 24px; margin: 16px 0;
    display: flex; align-items: center; gap: 16px;
    box-shadow: var(--shadow);
}
.result-banner.high {
    background: linear-gradient(135deg, #fde8ea, #fff);
    border: 1px solid #f5c2c7; border-left: 5px solid var(--red);
}
.result-banner.low {
    background: linear-gradient(135deg, #e6f7ee, #fff);
    border: 1px solid #b7dfc9; border-left: 5px solid var(--green);
}
.result-banner .ricon  { font-size: 2.2rem; }
.result-banner .rtitle { font-size: 1.2rem; font-weight: 700; }
.result-banner .rsub   { font-size: 0.84rem; color: var(--text2); margin-top: 3px; }

/* ── ALERT CARDS ── */
.alert-card {
    border-radius: var(--radius); padding: 13px 16px; margin: 6px 0;
    display: flex; align-items: flex-start; gap: 12px; font-size: 0.875rem;
    box-shadow: var(--shadow);
}
.alert-card.critical { background: #fde8ea; border: 1px solid #f5c2c7; border-left: 4px solid var(--red); }
.alert-card.warning  { background: #fef3e8; border: 1px solid #f5d5b5; border-left: 4px solid var(--amber); }
.alert-card.info     { background: var(--teal-light); border: 1px solid #bddff5; border-left: 4px solid var(--teal); }
.alert-icon { font-size: 1.1rem; margin-top: 1px; }
.alert-type { font-weight: 700; font-size: 0.7rem; letter-spacing: 1px; text-transform: uppercase; color: var(--text); }
.alert-msg  { font-size: 0.83rem; color: var(--text2); margin-top: 2px; }

/* ── REC PILLS ── */
.rec-pill {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 16px; margin: 5px 0;
    font-size: 0.875rem; color: var(--text2);
    box-shadow: var(--shadow);
    transition: border-color 0.15s, transform 0.15s;
}
.rec-pill:hover { border-color: var(--teal); transform: translateX(3px); }

/* ── INFO PANEL ── */
.info-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    font-size: 0.875rem; color: var(--text2);
    line-height: 1.7;
    box-shadow: var(--shadow);
}

/* ── BADGES ── */
.badge { display: inline-block; font-size: 0.68rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; padding: 3px 10px; border-radius: 20px; }
.badge-red   { background: var(--red-light);   color: var(--red-dark); border: 1px solid #f5c2c7; }
.badge-green { background: var(--green-light); color: #1a7a40;         border: 1px solid #b7dfc9; }
.badge-teal  { background: var(--teal-light);  color: #005a8e;         border: 1px solid #bddff5; }

/* ── CMD TABLE ── */
.cmd-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 0.875rem;
}
.cmd-row:last-child { border-bottom: none; }
.cmd-text   { color: var(--teal); font-weight: 600; }
.cmd-action { color: var(--text2); }

/* ── INPUTS ── */
.stTextInput > div > div,
.stNumberInput > div > div,
.stSelectbox > div > div {
    background: #fff !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.stTextInput > div > div:focus-within,
.stNumberInput > div > div:focus-within {
    border-color: var(--red) !important;
    box-shadow: 0 0 0 3px rgba(230,57,70,0.12) !important;
}
.stSlider > div > div > div > div { background-color: var(--red) !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #e63946 0%, #b02030 100%) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 0.95rem !important; padding: 12px 28px !important;
    box-shadow: 0 4px 14px rgba(230,57,70,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(230,57,70,0.45) !important;
}
.stDownloadButton > button {
    background: white !important; color: var(--teal) !important;
    border: 1.5px solid var(--teal) !important; border-radius: 8px !important;
    font-weight: 600 !important; padding: 10px 20px !important; box-shadow: none !important;
}
.stDownloadButton > button:hover {
    background: var(--teal-light) !important; transform: translateY(-1px) !important;
}

/* ── PROGRESS BAR ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--green), var(--amber), var(--red)) !important;
    border-radius: 4px !important;
}
.stProgress > div > div { background: var(--border) !important; border-radius: 4px !important; }

/* ── NATIVE METRICS ── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 16px 20px !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="metric-container"] label {
    color: var(--text3) !important; font-size: 0.72rem !important;
    letter-spacing: 1px !important; text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text) !important; font-weight: 700 !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow) !important;
    overflow: hidden !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--red) !important; }

/* ── ALERTS ── */
.stAlert { border-radius: var(--radius) !important; font-size: 0.875rem !important; }

hr { border-color: var(--border) !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

@keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.12)} }
.pulse { animation: pulse 2s ease-in-out infinite; display: inline-block; }

/* ── VOICE CONTROL ── */
.voice-orb {
    width: 90px; height: 90px; border-radius: 50%;
    background: linear-gradient(135deg, #e63946, #b02030);
    display: flex; align-items: center; justify-content: center;
    font-size: 2.2rem; margin: 0 auto 14px auto;
    box-shadow: 0 6px 24px rgba(230,57,70,0.35);
}
.voice-transcript {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 4px solid var(--teal); border-radius: var(--radius);
    padding: 14px 18px; margin: 10px 0; font-size: 0.9rem; color: var(--text);
}
.voice-reply {
    background: var(--teal-light); border: 1px solid #bddff5;
    border-left: 4px solid var(--teal); border-radius: var(--radius);
    padding: 14px 18px; margin: 10px 0; font-size: 0.9rem; color: var(--text);
}
.chat-bubble-user {
    background: var(--red-light); border: 1px solid #f5c2c7;
    border-radius: 12px 12px 2px 12px; padding: 10px 16px; margin: 6px 0;
    font-size: 0.875rem; color: var(--text); max-width: 85%; margin-left: auto;
}
.chat-bubble-ai {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 12px 12px 12px 2px; padding: 10px 16px; margin: 6px 0;
    font-size: 0.875rem; color: var(--text); max-width: 85%;
}

/* ── WHAT-IF SIMULATOR ── */
.whatif-delta-up   { color: var(--red);   font-weight: 700; }
.whatif-delta-down { color: var(--green); font-weight: 700; }
.whatif-delta-same { color: var(--text3); font-weight: 700; }

/* ── MULTI-MODEL COMPARE ── */
.model-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 16px 18px; box-shadow: var(--shadow);
    text-align: center;
}
.agree-banner {
    border-radius: var(--radius); padding: 16px 20px; margin: 14px 0;
    text-align: center; font-weight: 600; box-shadow: var(--shadow);
}
.agree-banner.yes { background: var(--green-light); border: 1px solid #b7dfc9; color: #1a7a40; }
.agree-banner.no  { background: var(--amber-light); border: 1px solid #f5d5b5; color: #b35e1a; }

/* ── DIFF TOOL ── */
.diff-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 14px; border-bottom: 1px solid var(--border); font-size: 0.875rem;
}
.diff-row:last-child { border-bottom: none; }
.diff-label { color: var(--text2); font-weight: 600; }
.diff-values { display:flex; gap:14px; align-items:center; }
</style>
""", unsafe_allow_html=True)


# =============================================================
# FILE PATHS — resolved relative to this script's folder
# =============================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _path(filename):
    return os.path.join(_SCRIPT_DIR, filename)

HEART_DATA_PATH    = _path("heart_disease_data.csv")
MEDICINE_DATA_PATH = _path("medicine_dataset.csv")
SYMPTOM_DATA_PATH  = _path("symptoms_dataset.csv")
MODEL_PATH         = _path("heart_model.pkl")
DB_PATH            = _path("patients.db")
IMAGE_PATH         = _path("human_heart.png")

# =============================================================
# HELPERS (original — untouched)
# =============================================================
def load_csv_optional(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Could not read {path}: {e}")
    return None

def numeric_default(series, fallback=0.0):
    if series is None:
        return fallback
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.median()) if not s.empty else fallback

def get_target_column(df):
    return "target" if "target" in df.columns else df.columns[-1]

def predict_probability(model, input_df):
    """
    Returns the percentage probability of HIGH cardiovascular risk (0-100).

    IMPORTANT — DATASET LABEL ORIENTATION:
    This specific heart_disease_data.csv (the popular "heart.csv" reupload)
    has its 'target' column inverted from normal clinical convention.
    Verified by group-mean analysis: patients with target=1 have LOWER
    average cholesterol, blood pressure, and oldpeak, and HIGHER average
    max heart rate, than patients with target=0 — i.e. target=1 is the
    clinically HEALTHIER group, not the diseased one. The trained model
    therefore predicts class 1 for low-risk patients. This function flips
    the raw output so every caller in the app receives a risk percentage
    where higher = higher clinical risk, as expected.
    """
    try:
        if hasattr(model, "predict_proba"):
            raw_p_class1 = float(model.predict_proba(input_df)[0][1])
            return (1.0 - raw_p_class1) * 100  # flip: high risk = low P(class 1)
    except Exception:
        pass
    try:
        raw_pred = int(model.predict(input_df)[0])
        return 100.0 if raw_pred == 0 else 0.0  # flip
    except Exception:
        return 0.0


def predict_high_risk_class(model, input_df):
    """
    Returns 1 if the patient is classified as HIGH cardiovascular risk,
    0 if low risk — flipped to correct clinical orientation.
    See predict_probability() docstring for why the flip is necessary.
    """
    try:
        raw_pred = int(model.predict(input_df)[0])
        return 1 - raw_pred
    except Exception:
        return 0

def calc_bmi(height_cm, weight_kg):
    if height_cm <= 0:
        return 0.0
    return round(weight_kg / (height_cm / 100.0) ** 2, 2)

def calculate_lifestyle_score(bmi, chol, trestbps, exang):
    score = 100
    if bmi < 18.5:        score -= 15
    elif 25 <= bmi < 30:  score -= 10
    elif bmi >= 30:        score -= 20
    if chol > 240:         score -= 15
    elif chol > 200:       score -= 5
    if trestbps > 140:     score -= 15
    elif trestbps > 120:   score -= 5
    if exang == 1:         score -= 10
    return max(0, min(100, score))

def build_input_widget(col_name, df):
    key    = col_name.lower().strip()
    series = df[col_name] if col_name in df.columns else None
    if key == "age":
        return st.slider("Age (years)", 20, 100, int(numeric_default(series, 50)))
    elif key == "sex":
        return 1 if st.selectbox("Biological Sex", ["Female","Male"]) == "Male" else 0
    elif key == "cp":
        return st.selectbox("Chest Pain Type", [0,1,2,3],
            format_func=lambda x:{0:"Typical Angina",1:"Atypical Angina",
                                   2:"Non-Anginal Pain",3:"Asymptomatic"}[x])
    elif key == "trestbps":
        return st.number_input("Resting Blood Pressure (mmHg)", 80, 250, int(numeric_default(series,120)))
    elif key == "chol":
        return st.number_input("Serum Cholesterol (mg/dl)", 100, 700, int(numeric_default(series,240)))
    elif key == "fbs":
        return st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1],
            format_func=lambda x:"Yes" if x==1 else "No")
    elif key == "restecg":
        return st.selectbox("Resting ECG Result", [0,1,2],
            format_func=lambda x:{0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}[x])
    elif key == "thalach":
        return st.number_input("Max Heart Rate Achieved (bpm)", 60, 250, int(numeric_default(series,150)))
    elif key == "exang":
        return st.selectbox("Exercise-Induced Angina", [0,1],
            format_func=lambda x:"Yes" if x==1 else "No")
    elif key == "oldpeak":
        return st.number_input("ST Depression (Oldpeak)", 0.0, 10.0,
            float(numeric_default(series,1.0)), step=0.1)
    elif key == "slope":
        return st.selectbox("Peak ST Slope", [0,1,2],
            format_func=lambda x:{0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
    elif key == "ca":
        return st.selectbox("Major Vessels Coloured (0-4)", [0,1,2,3,4])
    elif key == "thal":
        return st.selectbox("Thalassemia", [0,1,2,3],
            format_func=lambda x:{0:"Normal",1:"Fixed Defect",
                                   2:"Reversible Defect",3:"Not Described"}[x])
    if series is not None and pd.api.types.is_numeric_dtype(series):
        return st.number_input(col_name, value=float(numeric_default(series,0.0)))
    return st.text_input(col_name, value="")

def search_all_text_columns(df, query):
    """
    Case-insensitive substring search across all text-like columns.

    Uses pd.api.types.is_string_dtype() / is_object_dtype() instead of a
    raw `dtype == object` check. Pandas 3.x stores inferred string columns
    using its own StringDtype rather than the legacy numpy `object` dtype,
    so the old equality check silently matched zero columns on pandas 3.x
    even when the CSV was perfectly well-formed. This version works
    correctly on pandas 2.x and 3.x.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    text_cols = [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]
    if not text_cols:
        return pd.DataFrame()
    mask = pd.Series(False, index=df.index)
    for c in text_cols:
        mask |= df[c].astype(str).str.contains(query, case=False, na=False, regex=False)
    return df[mask]

def generate_health_recommendations(prediction, risk_score, input_values):
    recs = []
    if prediction == 1:
        recs += ["🚨 Consult a cardiologist immediately — high cardiac risk detected.",
                 "💊 Follow prescribed medication plans strictly.",
                 "🏥 Schedule regular cardiac monitoring and stress tests."]
    else:
        recs += ["✅ Maintain your current healthy lifestyle.",
                 "🥗 Balanced diet — low sodium, high fibre, lean protein.",
                 "🏃 150+ minutes of moderate aerobic exercise per week."]
    if input_values.get("age",0) > 50:
        recs.append("👴 Annual cardiac checkups recommended for age 50+.")
    if input_values.get("chol",0) > 240:
        recs.append("🍔 Elevated cholesterol — reduce saturated fats, increase fibre.")
    if input_values.get("trestbps",0) > 140:
        recs.append("🩸 High blood pressure — monitor daily, reduce salt, manage stress.")
    if input_values.get("thalach",0) < 120:
        recs.append("❤️ Low max heart rate — discuss cardiac fitness with your doctor.")
    if input_values.get("exang",0) == 1:
        recs.append("⚠️ Exercise-induced angina — avoid strenuous activity until cleared.")
    return recs

def generate_pdf_report(input_values, prediction, risk_score, bmi, lifestyle_score, recommendations):
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("<b>CardioAI — Heart Disease Risk Report</b>", styles["Heading1"]))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Paragraph("<br/>", styles["Normal"]))
    data = [
        ["Parameter","Value"],
        ["Age", str(input_values.get("age","N/A"))],
        ["Sex","Male" if input_values.get("sex",0)==1 else "Female"],
        ["Chest Pain Type", str(input_values.get("cp","N/A"))],
        ["Blood Pressure", f"{input_values.get('trestbps','N/A')} mmHg"],
        ["Cholesterol", f"{input_values.get('chol','N/A')} mg/dl"],
        ["Max Heart Rate", f"{input_values.get('thalach','N/A')} bpm"],
        ["BMI", f"{bmi:.2f}"],
        ["Lifestyle Score", f"{lifestyle_score}/100"],
        ["Risk Score", f"{risk_score:.2f}%"],
        ["Prediction", "HIGH RISK" if prediction==1 else "LOW RISK"],
    ]
    table = Table(data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e63946")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f3f3f3")]),
    ]))
    elements.append(table)
    elements.append(Paragraph("<br/>", styles["Normal"]))
    elements.append(Paragraph("<b>Health Recommendations:</b>", styles["Heading2"]))
    for rec in recommendations:
        elements.append(Paragraph(rec.replace("**",""), styles["Normal"]))
    doc.build(elements)
    buffer.seek(0)
    return buffer

def process_bulk_predictions(file, model, feature_columns):
    try:
        df = pd.read_csv(file)
        missing = set(feature_columns) - set(df.columns)
        if missing:
            return None, f"Missing columns: {missing}"
        X = df[feature_columns]
        raw_predictions = model.predict(X)
        predictions = 1 - raw_predictions  # flip — see predict_probability() docstring
        if hasattr(model, "predict_proba"):
            raw_proba_class1 = model.predict_proba(X)[:, 1]
            probabilities = (1.0 - raw_proba_class1) * 100  # flip
        else:
            probabilities = [100.0 if p == 1 else 0.0 for p in predictions]
        df["Prediction"]     = ["High Risk" if p==1 else "Low Risk" for p in predictions]
        df["Risk Score (%)"] = [round(p,2) for p in probabilities]
        return df, None
    except Exception as e:
        return None, str(e)

def check_smart_alerts(prediction, risk_score, input_values):
    alerts = []
    if prediction == 1:
        alerts.append(("critical","🚨","CRITICAL",
                        "High risk of heart disease — immediate medical consultation required."))
    if risk_score > 70:
        alerts.append(("warning","⚠️","HIGH RISK",
                        f"Risk score {risk_score:.1f}% exceeds safe threshold of 70%."))
    # Enhanced multi-condition alert (Feature 27 — Intelligent Alert Engine)
    high_bp   = input_values.get("trestbps",0) > 140
    high_chol = input_values.get("chol",0) > 240
    old_age   = input_values.get("age",0) > 60
    if high_bp and high_chol and old_age:
        alerts.append(("critical","🔥","COMBINED RISK",
                        "CRITICAL: High BP + High Cholesterol + Age >60 — Elevated cardiac event probability."))
    if input_values.get("age",0) > 65:
        alerts.append(("info","👴","AGE ALERT",
                        "Patient age >65 — enhanced cardiac monitoring recommended."))
    if input_values.get("chol",0) > 300:
        alerts.append(("warning","🍔","CHOLESTEROL",
                        f"Cholesterol {input_values.get('chol')} mg/dl is critically elevated."))
    if input_values.get("trestbps",0) > 160:
        alerts.append(("warning","🩸","BP ALERT",
                        f"Blood pressure {input_values.get('trestbps')} mmHg requires immediate attention."))
    if input_values.get("thalach",0) < 100:
        alerts.append(("info","❤️","HEART RATE",
                        f"Max heart rate {input_values.get('thalach')} bpm is abnormally low."))
    return alerts

# ── PLOTLY CHART BUILDERS (original — untouched) ──────────────
PLOTLY_BASE = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font=dict(color="#475569", family="Inter"),
)

def create_risk_gauge(risk_score):
    col = "#2a9d5c" if risk_score < 30 else "#e07b39" if risk_score < 60 else "#e63946"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"suffix":"%","font":{"size":44,"color":col,"family":"Inter"}},
        domain={"x":[0,1],"y":[0,1]},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#94a3b8",
                    "tickfont":{"color":"#94a3b8","size":11}},
            "bar":{"color":col,"thickness":0.28},
            "bgcolor":"#f8fafc","borderwidth":0,
            "steps":[{"range":[0,30],"color":"rgba(42,157,92,0.12)"},
                     {"range":[30,60],"color":"rgba(224,123,57,0.12)"},
                     {"range":[60,100],"color":"rgba(230,57,70,0.12)"}],
            "threshold":{"line":{"color":col,"width":3},"thickness":0.8,"value":risk_score},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20,r=20,t=20,b=10),
                      paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      font=dict(family="Inter",color="#475569"))
    return fig

def create_plotly_confusion_matrix(cm):
    labels = ["No Disease","Disease"]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=[f"Pred: {l}" for l in labels], y=[f"Actual: {l}" for l in labels],
        colorscale=[[0,"#f8fafc"],[1,"#e63946"]],
        text=cm, texttemplate="<b>%{text}</b>",
        textfont={"size":24,"color":"#0f172a"},
        showscale=False,
    ))
    fig.update_layout(title="Confusion Matrix", height=320,
                      margin=dict(l=10,r=10,t=50,b=10), **PLOTLY_BASE)
    return fig

def create_plotly_feature_importance(feature_columns, coef):
    df_c = pd.DataFrame({"Feature":feature_columns,"Value":coef}).sort_values("Value")
    bar_colors = ["#e63946" if v>0 else "#0077b6" for v in df_c["Value"]]
    fig = go.Figure(go.Bar(
        x=df_c["Value"], y=df_c["Feature"], orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in df_c["Value"]], textposition="outside",
        textfont=dict(size=10,color="#475569"),
    ))
    fig.add_vline(x=0, line_color="#e2e8f0", line_width=1)
    fig.update_layout(title="Feature Importance (Model Coefficients)",
        xaxis_title="Coefficient Value", height=400,
        margin=dict(l=10,r=30,t=50,b=10), **PLOTLY_BASE)
    fig.update_xaxes(gridcolor="#e2e8f0"); fig.update_yaxes(gridcolor="#e2e8f0")
    return fig

def create_plotly_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(2)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0,"#0077b6"],[0.5,"#f8fafc"],[1,"#e63946"]],
        zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text:.2f}",
        textfont={"size":9}, hoverongaps=False,
    ))
    fig.update_layout(title="Feature Correlation Matrix", height=480,
        margin=dict(l=10,r=10,t=50,b=10), **PLOTLY_BASE)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    return fig

def create_risk_trend_forecast(df_records):
    if df_records.empty or len(df_records) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_records["rowid"] if "rowid" in df_records.columns else df_records.index,
        y=df_records["risk"],
        mode="lines+markers", name="Risk Score",
        line=dict(color="#e63946",width=2.5), marker=dict(size=7,color="#e63946"),
        fill="tozeroy", fillcolor="rgba(230,57,70,0.07)",
    ))
    if len(df_records) >= 3:
        df_records = df_records.copy()
        df_records["forecast"] = df_records["risk"].rolling(window=3,min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df_records["rowid"] if "rowid" in df_records.columns else df_records.index,
            y=df_records["forecast"],
            mode="lines", name="3-pt Moving Avg",
            line=dict(color="#0077b6",width=2,dash="dash"),
        ))
    fig.update_layout(title="Patient Risk Trend & Forecast",
        xaxis_title="Prediction #", yaxis_title="Risk Score (%)",
        hovermode="x unified", height=380, legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10,r=10,t=50,b=10), **PLOTLY_BASE)
    fig.update_xaxes(gridcolor="#e2e8f0"); fig.update_yaxes(gridcolor="#e2e8f0")
    return fig

def create_analytics_pie(high, low):
    fig = go.Figure(go.Pie(
        labels=["High Risk","Low Risk"], values=[high,low], hole=0.55,
        marker=dict(colors=["#e63946","#2a9d5c"], line=dict(color="white",width=3)),
        textfont=dict(family="Inter",size=13,color="white"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(title="Risk Distribution",
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#475569")),
        height=340, margin=dict(l=10,r=10,t=50,b=10),
        annotations=[dict(text=f"<b>{high+low}</b><br>total",x=0.5,y=0.5,
            font=dict(size=16,color="#0f172a",family="Inter"),showarrow=False)],
        **PLOTLY_BASE)
    return fig

def create_risk_histogram(risks):
    fig = go.Figure(go.Histogram(
        x=risks, nbinsx=12,
        marker=dict(color="rgba(230,57,70,0.75)", line=dict(color="#e63946",width=1)),
    ))
    fig.update_layout(title="Risk Score Distribution",
        xaxis_title="Risk Score (%)", yaxis_title="Patients",
        height=320, bargap=0.05, margin=dict(l=10,r=10,t=50,b=10), **PLOTLY_BASE)
    fig.update_xaxes(gridcolor="#e2e8f0"); fig.update_yaxes(gridcolor="#e2e8f0")
    return fig

# =============================================================
# SHAP helpers (original — untouched)
# =============================================================
def get_shap_explainer(model, background_data):
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        scaler = model.named_steps.get("scaler", None)
        inner  = model.named_steps.get("model", None)
        if scaler is not None and inner is not None:
            scaled_bg = scaler.transform(background_data)
            try:
                explainer = shap.LinearExplainer(inner, scaled_bg)
                return explainer, scaler
            except Exception:
                pass
        sample_bg = shap.sample(background_data, 50)
        explainer = shap.KernelExplainer(model.predict_proba, sample_bg)
        return explainer, None
    try:
        explainer = shap.Explainer(model, background_data)
        return explainer, None
    except Exception:
        sample_bg = shap.sample(background_data, 50)
        explainer = shap.KernelExplainer(model.predict_proba, sample_bg)
        return explainer, None

def compute_shap_values(model, background_data, input_df):
    """
    Returns SHAP values oriented so POSITIVE = pushes risk UP (clinically
    high risk), NEGATIVE = pushes risk DOWN. Raw SHAP values explain
    contribution toward class 1, which in this dataset's inverted labelling
    is the LOW-risk class (see predict_probability() docstring) — so the
    raw values are negated here, once, at the source.
    """
    from sklearn.pipeline import Pipeline
    feature_names = list(input_df.columns)
    if isinstance(model, Pipeline):
        scaler = model.named_steps.get("scaler", None)
        inner  = model.named_steps.get("model", None)
        if scaler is not None and inner is not None:
            scaled_bg    = scaler.transform(background_data)
            scaled_input = scaler.transform(input_df)
            try:
                exp = shap.LinearExplainer(inner, scaled_bg)
                sv  = exp.shap_values(scaled_input)
                if isinstance(sv, list):
                    sv = sv[1]
                sv = -sv  # flip orientation — see docstring
                ev_raw = exp.expected_value if not hasattr(exp.expected_value,"__len__") else exp.expected_value[1]
                return sv[0], -ev_raw, feature_names
            except Exception:
                pass
    sample_bg = shap.sample(background_data, 50)
    exp = shap.KernelExplainer(model.predict_proba, sample_bg)
    sv  = exp.shap_values(input_df)
    if isinstance(sv, list):
        sv = sv[1]
    sv = -sv  # flip orientation
    return sv[0], -exp.expected_value[1], feature_names

def compute_shap_global(model, background_data):
    """
    Global SHAP values, flipped to the same clinical-risk orientation as
    compute_shap_values() — positive = pushes risk up. See that function's
    docstring for why the flip is necessary on this specific dataset.
    """
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        scaler = model.named_steps.get("scaler", None)
        inner  = model.named_steps.get("model", None)
        if scaler is not None and inner is not None:
            scaled_bg = scaler.transform(background_data)
            try:
                exp = shap.LinearExplainer(inner, scaled_bg)
                sv  = exp.shap_values(scaled_bg)
                if isinstance(sv, list):
                    sv = sv[1]
                return -sv  # flip orientation
            except Exception:
                pass
    sample_bg = shap.sample(background_data, 50)
    exp = shap.KernelExplainer(model.predict_proba, sample_bg)
    sv  = exp.shap_values(background_data)
    if isinstance(sv, list):
        sv = sv[1]
    return -sv  # flip orientation

# =============================================================
# ══ NEW FEATURE FUNCTIONS (Features 24–33) ══
# =============================================================

# ── Feature 24: AI Clinical Decision Engine ──────────────────
def generate_clinical_decision(prediction, risk_score, input_values, lifestyle_score):
    """Doctor-like clinical reasoning output based on all patient parameters."""
    age   = input_values.get("age", 0)
    chol  = input_values.get("chol", 0)
    bp    = input_values.get("trestbps", 0)
    hr    = input_values.get("thalach", 0)
    exang = input_values.get("exang", 0)
    ca    = input_values.get("ca", 0)
    cp    = input_values.get("cp", 0)

    urgency = "IMMEDIATE" if risk_score > 70 else "URGENT" if risk_score > 50 else "ROUTINE"
    color   = "#e63946" if urgency == "IMMEDIATE" else "#e07b39" if urgency == "URGENT" else "#2a9d5c"

    findings = []
    if chol > 240:
        findings.append(f"serum cholesterol elevated at {chol} mg/dl (borderline high ≥200, high ≥240)")
    if bp > 140:
        findings.append(f"resting blood pressure at {bp} mmHg indicates Stage 2 hypertension")
    if hr < 120:
        findings.append(f"maximum heart rate of {hr} bpm suggests reduced cardiac reserve")
    if exang == 1:
        findings.append("exercise-induced angina present — possible myocardial ischaemia")
    if ca > 1:
        findings.append(f"{ca} major vessels coloured by fluoroscopy — significant coronary involvement")
    if cp == 0:
        findings.append("typical angina pattern reported — characteristic of obstructive coronary disease")
    if age > 60:
        findings.append(f"age {age} — cardiovascular risk increases substantially after 60")

    if not findings:
        findings.append("no single critical parameter isolated; combined risk factors drive assessment")

    if prediction == 1:
        if urgency == "IMMEDIATE":
            decision = f"High cardiovascular risk detected (risk score: {risk_score:.1f}%). Immediate cardiology consultation within 24–48 hours is strongly recommended. Do not delay further evaluation."
        else:
            decision = f"Elevated cardiovascular risk (risk score: {risk_score:.1f}%). Cardiology referral recommended within 2 weeks. Initiate lifestyle modification programme and consider pharmacological intervention."
    else:
        if risk_score > 30:
            decision = f"Moderate-low cardiovascular risk (risk score: {risk_score:.1f}%). Annual cardiac review recommended. Lifestyle optimisation advised to reduce future risk."
        else:
            decision = f"Low cardiovascular risk detected (risk score: {risk_score:.1f}%). Maintain current lifestyle. Routine cardiac screening at next annual health check is sufficient."

    return urgency, color, findings, decision


# ── Feature 25: Digital Twin — Future Risk Simulation ────────
def simulate_future_risk(current_risk, input_values, months=6):
    """Simulates future risk trajectory over 3–6 months using trend extrapolation."""
    age   = input_values.get("age", 50)
    chol  = input_values.get("chol", 200)
    bp    = input_values.get("trestbps", 120)
    bmi_v = input_values.get("_bmi", 25.0)

    # Monthly drift factors based on risk parameters
    age_drift   = 0.15 if age > 55 else 0.08
    chol_drift  = 0.20 if chol > 240 else 0.05
    bp_drift    = 0.18 if bp > 140 else 0.04
    bmi_drift   = 0.12 if bmi_v > 28 else 0.02
    base_drift  = (age_drift + chol_drift + bp_drift + bmi_drift) / 4

    timeline = []
    r = current_risk
    for m in range(months + 1):
        label = "Now" if m == 0 else f"Month {m}"
        timeline.append({"month": label, "risk": round(min(r, 98), 1)})
        r += base_drift + np.random.uniform(-0.3, 0.5)

    df_twin = pd.DataFrame(timeline)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_twin["month"], y=df_twin["risk"],
        mode="lines+markers", name="Projected Risk",
        line=dict(color="#e63946", width=2.5),
        marker=dict(size=8, color="#e63946"),
        fill="tozeroy", fillcolor="rgba(230,57,70,0.08)",
    ))
    fig.add_hline(y=60, line_dash="dash", line_color="#e07b39",
                  annotation_text="High Risk Threshold (60%)",
                  annotation_position="bottom right")
    fig.add_hline(y=30, line_dash="dot", line_color="#2a9d5c",
                  annotation_text="Safe Zone (30%)",
                  annotation_position="top right")
    fig.update_layout(
        title=f"Digital Twin — {months}-Month Risk Projection",
        xaxis_title="Timeline", yaxis_title="Projected Risk (%)",
        yaxis=dict(range=[0, 100]),
        height=360, **PLOTLY_BASE,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig, df_twin


# ── Feature 26: Patient Segmentation (Clustering AI) ─────────
def segment_patients(df):
    """K-Means clustering to group patients into risk profiles."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler as SS

    if len(df) < 3:
        return None, None

    features = ["age", "chol", "trestbps", "thalach", "bmi"]
    available = [f for f in features if f in df.columns]
    if len(available) < 3:
        return None, None

    X = df[available].fillna(df[available].median())
    scaler = SS()
    Xs = scaler.fit_transform(X)

    n_clusters = min(3, len(df))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)

    cluster_names = {0: "Lifestyle Risk", 1: "High Clinical Risk", 2: "Age-Based Risk"}
    cluster_colors= {0: "#e07b39", 1: "#e63946", 2: "#0077b6"}
    df_seg = df[available].copy()
    df_seg["Cluster"] = [cluster_names.get(l, f"Group {l}") for l in labels]
    df_seg["Color"]   = [cluster_colors.get(l, "#94a3b8") for l in labels]

    fig = go.Figure()
    for name, color in cluster_colors.items():
        cname = cluster_names.get(name, f"Group {name}")
        mask  = df_seg["Cluster"] == cname
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=df_seg.loc[mask, available[0]],
            y=df_seg.loc[mask, available[1]],
            mode="markers",
            name=cname,
            marker=dict(color=color, size=9, opacity=0.8,
                        line=dict(color="white", width=1)),
            text=[f"{cname}" for _ in range(mask.sum())],
        ))
    fig.update_layout(
        title="Patient Segmentation Clusters",
        xaxis_title=available[0].upper(),
        yaxis_title=available[1].upper(),
        height=380, **PLOTLY_BASE,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig, df_seg


# ── Feature 27: Disease Progression Timeline ─────────────────
def render_progression_timeline(risk_score, prediction):
    """Visual disease progression stages from Healthy to Cardiac Event."""
    stages = [
        ("Healthy",   "#2a9d5c", "No significant risk factors"),
        ("At Risk",   "#e07b39", "Mild indicators present"),
        ("High Risk", "#e63946", "Multiple risk factors active"),
        ("Critical",  "#7b1d1d", "Immediate intervention required"),
    ]
    if risk_score < 25:
        active = 0
    elif risk_score < 50:
        active = 1
    elif risk_score < 75:
        active = 2
    else:
        active = 3

    html = '<div style="display:flex;gap:0;margin:12px 0;">'
    for i, (label, color, desc) in enumerate(stages):
        is_active = (i == active)
        opacity   = "1.0" if i <= active else "0.3"
        border    = f"3px solid {color}" if is_active else f"1px solid {color}"
        bg        = color if is_active else "white"
        txt_color = "white" if is_active else color
        arrow     = '<div style="display:flex;align-items:center;color:#94a3b8;font-size:1.2rem;padding:0 4px;">›</div>' if i < 3 else ""
        html += f'''
        <div style="flex:1;text-align:center;padding:12px 8px;background:{bg};
                    border:{border};border-radius:8px;opacity:{opacity};
                    transition:all 0.3s;">
            <div style="font-weight:700;font-size:0.85rem;color:{txt_color};">{label}</div>
            <div style="font-size:0.7rem;color:{"rgba(255,255,255,0.85)" if is_active else "#94a3b8"};
                        margin-top:4px;">{desc}</div>
        </div>{arrow}'''
    html += '</div>'
    return html, active, stages[active][0]


# ── Feature 28: Auto Medical Report Writer ───────────────────
def generate_doctor_notes(input_values, prediction, risk_score, bmi, lifestyle_score):
    """Generates professional clinical summary notes like a doctor's documentation."""
    age   = input_values.get("age", 0)
    sex   = "male" if input_values.get("sex", 0) == 1 else "female"
    chol  = input_values.get("chol", 0)
    bp    = input_values.get("trestbps", 0)
    hr    = input_values.get("thalach", 0)
    exang = input_values.get("exang", 0)
    cp    = input_values.get("cp", 0)
    ca    = input_values.get("ca", 0)
    thal  = input_values.get("thal", 0)

    cp_desc   = {0:"typical angina",1:"atypical angina",2:"non-anginal chest pain",3:"asymptomatic"}.get(cp,"unspecified")
    thal_desc = {0:"normal thalassemia",1:"fixed defect",2:"reversible defect",3:"unspecified thalassemia"}.get(thal,"unspecified")

    lines = []
    lines.append(f"PATIENT: {age}-year-old {sex}.")
    lines.append(f"PRESENTING COMPLAINT: Cardiovascular risk assessment via AI-assisted clinical platform.")
    lines.append(f"CLINICAL FINDINGS:")
    lines.append(f"  · Chest pain classification: {cp_desc}.")
    lines.append(f"  · Resting blood pressure: {bp} mmHg{'  — hypertensive range' if bp > 140 else ''}.")
    lines.append(f"  · Serum cholesterol: {chol} mg/dl{'  — elevated' if chol > 200 else ''}.")
    lines.append(f"  · Maximum heart rate achieved: {hr} bpm.")
    lines.append(f"  · Exercise-induced angina: {'Present' if exang==1 else 'Absent'}.")
    lines.append(f"  · Vessels coloured by fluoroscopy: {ca}.")
    lines.append(f"  · Thalassemia status: {thal_desc}.")
    lines.append(f"  · BMI: {bmi:.1f} kg/m².")
    lines.append(f"ALGORITHMIC ASSESSMENT: Risk score {risk_score:.1f}% — {'HIGH RISK' if prediction==1 else 'LOW RISK'}.")
    lines.append(f"LIFESTYLE INDEX: {lifestyle_score}/100.")

    if prediction == 1:
        lines.append("CLINICAL IMPRESSION: Patient exhibits multiple cardiovascular risk indicators consistent with elevated probability of coronary artery disease. Specialist referral and further diagnostic workup are warranted.")
        if bp > 140 and chol > 240:
            lines.append("NOTE: Concurrent hypertension and hypercholesterolaemia significantly compound cardiovascular risk. Pharmacological intervention should be considered.")
    else:
        lines.append("CLINICAL IMPRESSION: No dominant risk pattern identified. Patient advised to maintain current lifestyle practices and attend routine annual cardiac screening.")

    lines.append("DISCLAIMER: This assessment is generated by an AI-assisted educational tool and does not constitute a formal medical diagnosis. All findings must be confirmed by a qualified clinician.")
    return "\n".join(lines)


# ── Feature 29: Real-Time Monitoring Simulation (IoT) ────────
def simulate_iot_vitals(base_hr=72, base_bp=120, base_spo2=98, cycles=30):
    """Simulates real-time IoT vital signs — heart rate, BP, SpO2."""
    import random
    t = list(range(cycles))
    hr   = [base_hr   + random.uniform(-8, 8)   for _ in t]
    bp_s = [base_bp   + random.uniform(-10, 10) for _ in t]
    spo2 = [base_spo2 + random.uniform(-1.5, 0.5) for _ in t]
    spo2 = [min(100, max(90, v)) for v in spo2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=hr,   mode="lines", name="Heart Rate (bpm)",
                             line=dict(color="#e63946", width=2)))
    fig.add_trace(go.Scatter(x=t, y=bp_s, mode="lines", name="Systolic BP (mmHg)",
                             line=dict(color="#0077b6", width=2)))
    fig.add_trace(go.Scatter(x=t, y=spo2, mode="lines", name="SpO₂ (%)",
                             line=dict(color="#2a9d5c", width=2)))
    fig.update_layout(
        title="Real-Time Vital Signs Simulation (IoT Monitor)",
        xaxis_title="Time (seconds)", yaxis_title="Value",
        height=380, hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=50, b=10),
        **PLOTLY_BASE,
    )
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig, round(float(np.mean(hr)),1), round(float(np.mean(bp_s)),1), round(float(np.mean(spo2)),1)


# ── Feature 30: Anomaly Detection ────────────────────────────
def detect_anomalies(input_values, heart_df, feature_columns):
    """Detects statistically unusual patient values vs dataset population."""
    anomalies = []
    for feat in feature_columns:
        val = input_values.get(feat)
        if val is None or feat not in heart_df.columns:
            continue
        series = pd.to_numeric(heart_df[feat], errors="coerce").dropna()
        if series.empty:
            continue
        mean = series.mean()
        std  = series.std()
        if std == 0:
            continue
        z = (float(val) - mean) / std
        if abs(z) > 2.5:
            direction = "unusually high" if z > 0 else "unusually low"
            anomalies.append({
                "Feature": feat.upper(),
                "Value": val,
                "Population Mean": round(mean, 2),
                "Z-Score": round(z, 2),
                "Status": f"{direction} compared to dataset (|Z|={abs(z):.2f})",
                "Severity": "critical" if abs(z) > 3.5 else "warning",
            })
    return sorted(anomalies, key=lambda x: abs(x["Z-Score"]), reverse=True)


# ── Feature 31: Multi-Patient Comparison ─────────────────────
def render_patient_comparison(df_records, selected_ids, feature_columns):
    """Side-by-side comparison of selected patients from history."""
    if df_records.empty or not selected_ids:
        return None
    subset = df_records[df_records["rowid"].isin(selected_ids)]
    if len(subset) < 2:
        return None

    compare_cols = ["age","sex","chol","trestbps","thalach","bmi","risk","lifestyle_score","prediction"]
    available    = [c for c in compare_cols if c in subset.columns]
    subset_disp  = subset[available].copy()
    subset_disp.index = [f"Patient {r}" for r in subset["rowid"].values]

    fig = go.Figure()
    numeric_cols = [c for c in available if c not in ["prediction","sex"] and subset[c].dtype in [float, int, np.float64, np.int64]]
    for col in numeric_cols[:6]:
        fig.add_trace(go.Bar(
            name=col.upper(),
            x=[f"Patient {r}" for r in subset["rowid"].values],
            y=subset[col].values,
            text=[f"{v:.1f}" for v in subset[col].values],
            textposition="outside",
        ))
    fig.update_layout(
        title="Multi-Patient Comparison",
        barmode="group", height=400,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10,r=10,t=50,b=10),
        **PLOTLY_BASE,
    )
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig, subset_disp


# ── Feature 34: What-If Risk Simulator helper ────────────────
def whatif_risk_gauge_pair(baseline_risk, live_risk):
    """Side-by-side mini gauges: baseline vs live what-if risk."""
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=baseline_risk,
        title={"text": "Baseline", "font": {"size": 13, "color": "#475569"}},
        number={"suffix": "%", "font": {"size": 26, "color": "#475569"}},
        domain={"x": [0, 0.46], "y": [0, 1]},
        gauge={"axis": {"range": [0,100], "tickfont":{"size":8}}, "bar": {"color": "#94a3b8"},
               "steps":[{"range":[0,30],"color":"rgba(42,157,92,0.10)"},
                        {"range":[30,60],"color":"rgba(224,123,57,0.10)"},
                        {"range":[60,100],"color":"rgba(230,57,70,0.10)"}]},
    ))
    live_col = "#2a9d5c" if live_risk < 30 else "#e07b39" if live_risk < 60 else "#e63946"
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=live_risk,
        title={"text": "Live What-If", "font": {"size": 13, "color": live_col}},
        number={"suffix": "%", "font": {"size": 26, "color": live_col}},
        domain={"x": [0.54, 1], "y": [0, 1]},
        gauge={"axis": {"range": [0,100], "tickfont":{"size":8}}, "bar": {"color": live_col},
               "steps":[{"range":[0,30],"color":"rgba(42,157,92,0.10)"},
                        {"range":[30,60],"color":"rgba(224,123,57,0.10)"},
                        {"range":[60,100],"color":"rgba(230,57,70,0.10)"}]},
    ))
    fig.update_layout(height=220, margin=dict(l=10,r=10,t=40,b=10),
                      paper_bgcolor="#ffffff", font=dict(family="Inter"))
    return fig


# ── Feature 37: Risk Diff Tool helper ────────────────────────
def compute_patient_diff(row_a, row_b, compare_cols):
    """Computes deltas between two patient records for the diff tool."""
    diffs = []
    for col in compare_cols:
        if col not in row_a or col not in row_b:
            continue
        try:
            va, vb = float(row_a[col]), float(row_b[col])
        except (TypeError, ValueError):
            continue
        delta = vb - va
        diffs.append({
            "Parameter": col.upper(),
            "Visit A": round(va, 2),
            "Visit B": round(vb, 2),
            "Change": round(delta, 2),
            "Direction": "up" if delta > 0.01 else "down" if delta < -0.01 else "same",
        })
    return diffs


# =============================================================
# LOAD DATA & MODEL
# =============================================================
heart_df = load_csv_optional(HEART_DATA_PATH)
if heart_df is None:
    st.error(f"Missing '{HEART_DATA_PATH}'. Place it in the project folder and restart.")
    st.stop()

target_col      = get_target_column(heart_df)
feature_columns = [c for c in heart_df.columns if c != target_col]
medicine_df     = load_csv_optional(MEDICINE_DATA_PATH)
symptom_df      = load_csv_optional(SYMPTOM_DATA_PATH)

if not os.path.exists(MODEL_PATH):
    st.error(f"Missing '{MODEL_PATH}'. Run `python train_model.py` first.")
    st.stop()
with open(MODEL_PATH,"rb") as f:
    model = pickle.load(f)

# =============================================================
# DATABASE SETUP
# =============================================================
conn   = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(patients)")
existing_cols = {row[1] for row in cursor.fetchall()}
EXPECTED_COLS = {"age","sex","cp","trestbps","chol","fbs","restecg","thalach",
                 "exang","oldpeak","slope","ca","thal","bmi","lifestyle_score",
                 "risk","prediction","created_at"}
if existing_cols and not EXPECTED_COLS.issubset(existing_cols):
    cursor.execute("ALTER TABLE patients RENAME TO patients_old")
    conn.commit()
cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (
    rowid          INTEGER PRIMARY KEY AUTOINCREMENT,
    age            INTEGER, sex INTEGER, cp INTEGER, trestbps INTEGER,
    chol           INTEGER, fbs INTEGER, restecg INTEGER, thalach INTEGER,
    exang          INTEGER, oldpeak REAL, slope INTEGER, ca INTEGER, thal INTEGER,
    bmi            REAL, lifestyle_score REAL, risk REAL, prediction TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)""")
conn.commit()

# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h1><span class="pulse">🫀</span> CardioAI</h1>
        <p>Heart Intelligence Platform</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)

    NAV_PAGES = [
        "🩺  Prediction",
        "🎙️  Voice Control",
        "🧠  AI Clinical Engine",
        "🔮  Digital Twin",
        "🧪  What-If Simulator",
        "🏆  Multi-Model Compare",
        "📊  Model Insights",
        "📁  Patient History",
        "👥  Patient Comparison",
        "🆚  Risk Diff Tool",
        "📈  Analytics",
        "🔬  Patient Segmentation",
        "📡  IoT Monitor",
        "🚨  Anomaly Detection",
        "💊  Medicine Assistant",
        "🧬  Symptom Checker",
        "📄  Reports & Batch",
        "⚙️  Advanced",
    ]

    # If a voice command set a navigation target, jump there once, then clear it
    _default_idx = 0
    if st.session_state.get("voice_nav_target") in NAV_PAGES:
        _default_idx = NAV_PAGES.index(st.session_state["voice_nav_target"])
        st.session_state["voice_nav_target"] = None

    page = st.radio("nav", NAV_PAGES, index=_default_idx, label_visibility="collapsed")

    st.divider()
    st.markdown('<div class="sidebar-section">System Status</div>', unsafe_allow_html=True)
    df_count = pd.read_sql_query("SELECT COUNT(*) as n FROM patients", conn).iloc[0]["n"]
    _groq_ok = groq_client.is_configured()
    st.markdown(f"""
    <div class="sidebar-stat"><span class="k">Dataset rows</span><span class="v">{len(heart_df):,}</span></div>
    <div class="sidebar-stat"><span class="k">Features</span><span class="v">{len(feature_columns)}</span></div>
    <div class="sidebar-stat"><span class="k">Predictions made</span><span class="v">{int(df_count)}</span></div>
    <div class="sidebar-stat"><span class="k">Medicine DB</span><span class="v">{"✅" if medicine_df is not None else "❌"}</span></div>
    <div class="sidebar-stat"><span class="k">Symptom DB</span><span class="v">{"✅" if symptom_df is not None else "❌"}</span></div>
    <div class="sidebar-stat"><span class="k">Groq Voice/AI</span><span class="v">{"✅ Connected" if _groq_ok else "⚪ Not configured"}</span></div>
    """, unsafe_allow_html=True)
    if not _groq_ok:
        st.markdown('<div style="font-size:0.68rem;color:#94a3b8;padding:4px 2px;">Add <code>GROQ_API_KEY</code> in Secrets to enable Voice Control &amp; AI Chat.</div>', unsafe_allow_html=True)
    st.markdown('<br><div style="font-size:0.7rem;color:#94a3b8;text-align:center;">v5.0 · 37 Features · Educational use only</div>', unsafe_allow_html=True)


# =============================================================
# PAGE ROUTING
# =============================================================

# ── PREDICTION (original — untouched) ────────────────────────
if "Prediction" in page:
    st.markdown("""<div class="page-hero">
        <h2>🩺 Heart Disease Risk Prediction</h2>
        <p>Enter patient clinical parameters to assess cardiovascular risk using the trained AI model.</p>
    </div>""", unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2 = st.columns([1.1, 0.9], gap="large")
        with col1:
            st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#e63946;margin-bottom:12px;">📋 Clinical Parameters</p>', unsafe_allow_html=True)
            input_values = {}
            for col_name in feature_columns:
                input_values[col_name] = build_input_widget(col_name, heart_df)
        with col2:
            st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#e63946;margin-bottom:12px;">📏 Body Measurements</p>', unsafe_allow_html=True)
            height_cm = st.number_input("Height (cm)", 50.0, 250.0, 170.0, step=1.0)
            weight_kg = st.number_input("Weight (kg)", 20.0, 250.0, 70.0, step=1.0)
            bmi = calc_bmi(height_cm, weight_kg)
            bmi_label = ("Underweight" if bmi<18.5 else "Normal" if bmi<25 else "Overweight" if bmi<30 else "Obese")
            bmi_color = ("#2a9d5c" if bmi<25 else "#e07b39" if bmi<30 else "#e63946")
            st.markdown(f"""
            <div class="metric-card" style="margin-top:8px;border-top:3px solid {bmi_color};">
                <div class="label">Body Mass Index</div>
                <div class="value" style="color:{bmi_color}">{bmi:.1f}</div>
                <div class="sub">{bmi_label}</div>
            </div>""", unsafe_allow_html=True)
            if os.path.exists(IMAGE_PATH):
                try:
                    st.image(Image.open(IMAGE_PATH), caption="Human Heart Anatomy", width="stretch")
                except Exception:
                    pass
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍  Run Risk Assessment", width="stretch")

    if submitted:
        input_df   = pd.DataFrame([[input_values[c] for c in feature_columns]], columns=feature_columns)
        prediction = predict_high_risk_class(model, input_df)
        risk_score = predict_probability(model, input_df)
        lifestyle_score = calculate_lifestyle_score(bmi, input_values.get("chol",0),
                                                    input_values.get("trestbps",0), input_values.get("exang",0))
        result_text = "High Risk" if prediction==1 else "Low Risk"

        st.markdown('<div class="section-header"><span>📊 Assessment Results</span></div>', unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(f"""<div class="result-banner high">
                <div class="ricon">⚠️</div>
                <div><div class="rtitle" style="color:#b02030;">High Risk of Heart Disease</div>
                <div class="rsub">Risk Score: {risk_score:.1f}% — Please consult a cardiologist immediately.</div></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-banner low">
                <div class="ricon">✅</div>
                <div><div class="rtitle" style="color:#1a7a40;">Low Risk of Heart Disease</div>
                <div class="rsub">Risk Score: {risk_score:.1f}% — Maintain your healthy lifestyle.</div></div>
            </div>""", unsafe_allow_html=True)

        g1,g2,g3,g4 = st.columns(4)
        with g1:
            st.markdown(f"""<div class="metric-card {'red' if risk_score>60 else 'green'}">
                <div class="label">Risk Score</div><div class="value">{risk_score:.1f}%</div>
                <div class="sub">{"Above threshold" if risk_score>60 else "Within safe range"}</div></div>""", unsafe_allow_html=True)
        with g2:
            st.markdown(f"""<div class="metric-card {'green' if lifestyle_score>=70 else 'amber'}">
                <div class="label">Lifestyle Score</div><div class="value">{lifestyle_score}</div>
                <div class="sub">out of 100</div></div>""", unsafe_allow_html=True)
        with g3:
            st.markdown(f"""<div class="metric-card teal">
                <div class="label">BMI</div><div class="value">{bmi:.1f}</div>
                <div class="sub">{bmi_label}</div></div>""", unsafe_allow_html=True)
        with g4:
            st.markdown(f"""<div class="metric-card {'red' if prediction==1 else 'green'}">
                <div class="label">Prediction</div>
                <div class="value" style="font-size:1.1rem;margin-top:6px;">{"🔴 High Risk" if prediction==1 else "🟢 Low Risk"}</div>
                <div class="sub">Confidence: {max(risk_score,100-risk_score):.1f}%</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        gc1, gc2 = st.columns(2)
        with gc1:
            st.plotly_chart(create_risk_gauge(risk_score), width="stretch")
        with gc2:
            st.markdown('<div class="section-header"><span>🚨 Smart Alerts</span></div>', unsafe_allow_html=True)
            alerts = check_smart_alerts(prediction, risk_score, input_values)
            if alerts:
                for cls,icon,atype,msg in alerts:
                    st.markdown(f"""<div class="alert-card {cls}">
                        <div class="alert-icon">{icon}</div>
                        <div><div class="alert-type">{atype}</div><div class="alert-msg">{msg}</div></div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-panel">✅ No critical alerts for this patient profile.</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header"><span>💡 Personalised Health Recommendations</span></div>', unsafe_allow_html=True)
        recommendations = generate_health_recommendations(prediction, risk_score, input_values)
        rec_cols = st.columns(2)
        for i, rec in enumerate(recommendations):
            with rec_cols[i % 2]:
                st.markdown(f'<div class="rec-pill">{rec}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header"><span>🔍 SHAP Explainability — Why This Prediction?</span></div>', unsafe_allow_html=True)
        with st.spinner("Computing SHAP explanation…"):
            try:
                sv, ev, feat_names = compute_shap_values(model, heart_df[feature_columns], input_df)
                shap_df = pd.DataFrame({"Feature": feat_names, "SHAP": sv})
                shap_df = shap_df.reindex(shap_df["SHAP"].abs().sort_values().index)
                bar_colors = ["#e63946" if v>0 else "#0077b6" for v in shap_df["SHAP"]]
                fig_sv = go.Figure(go.Bar(x=shap_df["SHAP"], y=shap_df["Feature"], orientation="h",
                    marker_color=bar_colors, text=[f"{v:+.3f}" for v in shap_df["SHAP"]],
                    textposition="outside", textfont=dict(size=11, color="#475569")))
                fig_sv.add_vline(x=0, line_color="#e2e8f0", line_width=1.5)
                fig_sv.update_layout(title="SHAP Feature Contributions (Red = increases risk, Blue = reduces risk)",
                    xaxis_title="SHAP Value", height=420, margin=dict(l=10,r=40,t=50,b=10), **PLOTLY_BASE)
                st.plotly_chart(fig_sv, width="stretch")
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

        try:
            cursor.execute("""INSERT INTO patients
                (age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,bmi,lifestyle_score,risk,prediction)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                int(input_values.get("age",0)), int(input_values.get("sex",0)),
                int(input_values.get("cp",0)), int(input_values.get("trestbps",0)),
                int(input_values.get("chol",0)), int(input_values.get("fbs",0)),
                int(input_values.get("restecg",0)), int(input_values.get("thalach",0)),
                int(input_values.get("exang",0)), float(input_values.get("oldpeak",0.0)),
                int(input_values.get("slope",0)), int(input_values.get("ca",0)),
                int(input_values.get("thal",0)), float(bmi), float(lifestyle_score),
                float(risk_score), result_text,
            ))
            conn.commit()
            st.success("✅ Patient record saved to database.")
        except Exception as e:
            st.warning(f"Database save failed: {e}")

        st.markdown('<div class="section-header"><span>📄 Download Report</span></div>', unsafe_allow_html=True)
        pdf_buf = generate_pdf_report(input_values, prediction, risk_score, bmi, lifestyle_score, recommendations)
        st.download_button("📥  Download PDF Health Report", data=pdf_buf,
            file_name=f"cardioai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf")

# ── FEATURE 34: VOICE CONTROL (real-time, Groq Whisper + GPT-OSS) ─
elif "Voice Control" in page:
    st.markdown("""<div class="page-hero">
        <h2>🎙️ Real-Time Voice Control</h2>
        <p>Speak a page name to navigate hands-free, or ask the AI assistant a question about your last result.</p>
    </div>""", unsafe_allow_html=True)

    if not groq_client.is_configured():
        st.markdown("""<div class="alert-card warning">
            <div class="alert-icon">⚪</div>
            <div><div class="alert-type">VOICE FEATURES NOT CONFIGURED</div>
            <div class="alert-msg">Add <code>GROQ_API_KEY</code> to your Streamlit Secrets to enable voice navigation and AI chat. See the README for setup instructions. The Groq API key is never displayed or logged by this app.</div></div>
        </div>""", unsafe_allow_html=True)

    vc1, vc2 = st.columns([1, 1.2], gap="large")

    with vc1:
        st.markdown('<div class="section-header"><span>🎤 Voice Navigation</span></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;padding:10px 0;">
            <div class="voice-orb">🎙️</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="info-panel">Record yourself saying a page name — e.g. "Open Digital Twin", "Show Analytics", "Go to Medicine Assistant" — and the app will navigate there automatically.</div>', unsafe_allow_html=True)

        audio_value = st.audio_input("Record your voice command")

        if audio_value is not None:
            with st.spinner("Transcribing with Groq Whisper…"):
                transcript, err = groq_client.transcribe_audio(audio_value.getvalue(), "command.wav")
            if err:
                st.warning(err)
            elif transcript:
                st.markdown(f'<div class="voice-transcript">🗣️ You said: <b>"{transcript}"</b></div>', unsafe_allow_html=True)
                kw, target_page = groq_client.parse_voice_command(transcript)
                if target_page:
                    st.markdown(f'<div class="voice-reply">✅ Navigating to <b>{target_page}</b> — switching now…</div>', unsafe_allow_html=True)
                    st.session_state["voice_nav_target"] = target_page
                    st.rerun()
                else:
                    st.markdown('<div class="voice-reply">❓ Command not recognised. Try saying a page name like "Prediction", "Analytics", "Digital Twin", or "Medicine Assistant".</div>', unsafe_allow_html=True)
            else:
                st.info("No speech detected. Try recording again.")

        st.markdown('<div class="section-header"><span>📜 Available Commands</span></div>', unsafe_allow_html=True)
        for kw_phrase, desc in groq_client.VOICE_COMMANDS.items():
            st.markdown(f'<div class="cmd-row"><span class="cmd-text">🎙 "{kw_phrase}"</span><span class="cmd-action">→ {desc}</span></div>', unsafe_allow_html=True)

    with vc2:
        st.markdown('<div class="section-header"><span>💬 AI Health Assistant</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="info-panel">Ask questions about cardiovascular risk factors, lifestyle changes, or how to interpret your assessment. This is general educational information, not a diagnosis.</div>', unsafe_allow_html=True)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history[-10:]:
            cls = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-ai"
            icon = "🧑" if msg["role"] == "user" else "🫀"
            st.markdown(f'<div class="{cls}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

        chat_input = st.chat_input("Ask about heart health, your risk factors, or this report…")
        if chat_input:
            st.session_state.chat_history.append({"role": "user", "content": chat_input})
            system_prompt = {
                "role": "system",
                "content": (
                    "You are a careful, friendly cardiovascular health educator inside the CardioAI app. "
                    "Explain concepts about heart disease risk factors, lifestyle changes, and how to read "
                    "BMI/cholesterol/blood pressure numbers in plain language. Keep answers under 120 words. "
                    "Always include a brief reminder that this is educational information, not a diagnosis, "
                    "and to consult a doctor for medical decisions. Never invent specific patient data you "
                    "were not given."
                ),
            }
            history_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history[-6:]]
            reply, err = groq_client.chat_completion([system_prompt] + history_msgs)
            if err:
                st.session_state.chat_history.append({"role": "assistant", "content": err})
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️  Clear Chat", width="content"):
                st.session_state.chat_history = []
                st.rerun()

# ── FEATURE 24: AI CLINICAL ENGINE ───────────────────────────
elif "AI Clinical Engine" in page:
    st.markdown("""<div class="page-hero">
        <h2>🧠 AI Clinical Decision Engine</h2>
        <p>Doctor-like clinical reasoning — moves beyond prediction to actionable medical guidance.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="info-panel">Enter patient values below to receive a structured clinical assessment with findings, urgency classification, and recommended action.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("clinical_form"):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            c_age   = st.slider("Age", 20, 100, 55)
            c_chol  = st.number_input("Cholesterol (mg/dl)", 100, 700, 240)
        with cc2:
            c_bp    = st.number_input("Blood Pressure (mmHg)", 80, 250, 130)
            c_hr    = st.number_input("Max Heart Rate (bpm)", 60, 250, 140)
        with cc3:
            c_exang = st.selectbox("Exercise Angina", [0,1], format_func=lambda x:"Yes" if x==1 else "No")
            c_ca    = st.selectbox("Vessels Coloured (0-4)", [0,1,2,3,4])
            c_cp    = st.selectbox("Chest Pain Type", [0,1,2,3],
                format_func=lambda x:{0:"Typical Angina",1:"Atypical",2:"Non-Anginal",3:"Asymptomatic"}[x])
        c_height = st.number_input("Height (cm)", 50.0, 250.0, 170.0, step=1.0)
        c_weight = st.number_input("Weight (kg)", 20.0, 250.0, 75.0, step=1.0)
        c_bmi    = calc_bmi(c_height, c_weight)
        c_submitted = st.form_submit_button("🩺  Generate Clinical Assessment", width="stretch")

    if c_submitted:
        c_vals = {"age":c_age,"chol":c_chol,"trestbps":c_bp,"thalach":c_hr,
                  "exang":c_exang,"ca":c_ca,"cp":c_cp,"_bmi":c_bmi}
        c_input_df = pd.DataFrame([[c_vals.get(col, 0) for col in feature_columns]], columns=feature_columns)
        c_pred     = predict_high_risk_class(model, c_input_df)
        c_risk     = predict_probability(model, c_input_df)
        c_ls       = calculate_lifestyle_score(c_bmi, c_chol, c_bp, c_exang)

        urgency, color, findings, decision = generate_clinical_decision(c_pred, c_risk, c_vals, c_ls)

        st.markdown(f"""
        <div style="background:{'#fde8ea' if urgency=='IMMEDIATE' else '#fef3e8' if urgency=='URGENT' else '#e6f7ee'};
                    border:1px solid {color};border-left:5px solid {color};border-radius:10px;
                    padding:20px 24px;margin:16px 0;">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
                <span style="font-size:1.8rem;">{"🚨" if urgency=="IMMEDIATE" else "⚠️" if urgency=="URGENT" else "✅"}</span>
                <div>
                    <div style="font-weight:700;font-size:1.1rem;color:{color};">
                        {urgency} — {c_risk:.1f}% Cardiovascular Risk
                    </div>
                    <div style="font-size:0.85rem;color:#475569;margin-top:2px;">{decision}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header"><span>🔬 Clinical Findings</span></div>', unsafe_allow_html=True)
        for f in findings:
            st.markdown(f"""<div class="alert-card {'critical' if urgency=='IMMEDIATE' else 'warning' if urgency=='URGENT' else 'info'}">
                <div class="alert-icon">📋</div>
                <div><div class="alert-msg">{f}</div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header"><span>📋 Disease Progression Stage</span></div>', unsafe_allow_html=True)
        prog_html, active_stage, stage_name = render_progression_timeline(c_risk, c_pred)
        st.markdown(prog_html, unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center;margin-top:6px;font-size:0.85rem;color:#475569;">Current Stage: <b style="color:{color}">{stage_name}</b></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header"><span>📝 Auto Medical Report (Doctor Notes)</span></div>', unsafe_allow_html=True)
        doctor_notes = generate_doctor_notes(c_vals, c_pred, c_risk, c_bmi, c_ls)
        st.code(doctor_notes, language=None)
        st.download_button("📥  Download Doctor Notes (.txt)",
            data=doctor_notes.encode("utf-8"),
            file_name=f"clinical_notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain")

# ── FEATURE 25: DIGITAL TWIN ──────────────────────────────────
elif "Digital Twin" in page:
    st.markdown("""<div class="page-hero">
        <h2>🔮 Digital Twin — Future Risk Simulation</h2>
        <p>Simulate how a patient's cardiovascular risk is projected to evolve over the next 3–6 months based on current clinical parameters.</p>
    </div>""", unsafe_allow_html=True)

    with st.form("twin_form"):
        dt1, dt2, dt3 = st.columns(3)
        with dt1:
            dt_age    = st.slider("Age", 20, 100, 55)
            dt_chol   = st.number_input("Cholesterol (mg/dl)", 100, 700, 230)
        with dt2:
            dt_bp     = st.number_input("Blood Pressure (mmHg)", 80, 250, 130)
            dt_height = st.number_input("Height (cm)", 100.0, 220.0, 170.0, step=1.0)
            dt_weight = st.number_input("Weight (kg)", 30.0, 200.0, 75.0, step=1.0)
        with dt3:
            dt_months = st.selectbox("Projection Period", [3, 6], index=1, format_func=lambda x:f"{x} months")
            dt_exang  = st.selectbox("Exercise Angina", [0,1], format_func=lambda x:"Yes" if x==1 else "No")
        dt_sub = st.form_submit_button("🔮  Run Digital Twin Simulation", width="stretch")

    if dt_sub:
        dt_bmi   = calc_bmi(dt_height, dt_weight)
        dt_vals  = {"age":dt_age,"chol":dt_chol,"trestbps":dt_bp,"exang":dt_exang,"_bmi":dt_bmi}
        dt_input = pd.DataFrame([[dt_vals.get(col, 0) for col in feature_columns]], columns=feature_columns)
        dt_risk  = predict_probability(model, dt_input)

        fig_twin, df_twin = simulate_future_risk(dt_risk, dt_vals, dt_months)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="metric-card teal">
                <div class="label">Current Risk</div><div class="value">{dt_risk:.1f}%</div></div>""", unsafe_allow_html=True)
        with m2:
            end_risk = df_twin["risk"].iloc[-1]
            delta_color = "red" if end_risk > dt_risk else "green"
            st.markdown(f"""<div class="metric-card {'red' if end_risk>60 else 'green'}">
                <div class="label">Projected Risk ({dt_months}m)</div>
                <div class="value" style="color:var(--{delta_color})">{end_risk:.1f}%</div></div>""", unsafe_allow_html=True)
        with m3:
            delta = end_risk - dt_risk
            st.markdown(f"""<div class="metric-card {'red' if delta>0 else 'green'}">
                <div class="label">Projected Change</div>
                <div class="value" style="font-size:1.5rem;">{"▲" if delta>0 else "▼"} {abs(delta):.1f}%</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(fig_twin, width="stretch")

        st.markdown('<div class="section-header"><span>📊 Projection Data Table</span></div>', unsafe_allow_html=True)
        st.dataframe(df_twin, width="stretch")

        if end_risk > 60:
            st.markdown("""<div class="alert-card critical">
                <div class="alert-icon">🔮</div>
                <div><div class="alert-type">DIGITAL TWIN WARNING</div>
                <div class="alert-msg">Simulation projects risk exceeding 60% within the forecast period. Immediate lifestyle intervention or medical review is strongly advised to alter this trajectory.</div></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="alert-card info">
                <div class="alert-icon">✅</div>
                <div><div class="alert-type">PROJECTION STABLE</div>
                <div class="alert-msg">Risk trajectory remains within manageable range. Continue current lifestyle practices and attend scheduled annual review.</div></div>
            </div>""", unsafe_allow_html=True)

# ── FEATURE 35: WHAT-IF RISK SIMULATOR (live, no submit) ──────
elif "What-If Simulator" in page:
    st.markdown("""<div class="page-hero">
        <h2>🧪 What-If Risk Simulator</h2>
        <p>Drag the sliders below — risk recalculates instantly. See exactly how much each lifestyle change could lower (or raise) cardiovascular risk.</p>
    </div>""", unsafe_allow_html=True)

    # ── Apply any pending reset BEFORE the widgets below are instantiated.
    # Streamlit forbids writing to st.session_state[key] after a widget with
    # that key has already been created in the same run, so the reset must
    # happen here, at the top of the page, before the sliders are built.
    if st.session_state.get("_wi_reset_pending"):
        _wi_baseline = {"wi_age":50, "wi_chol":200, "wi_bp":120, "wi_hr":150,
                        "wi_weight":70, "wi_height":170, "wi_exang":0, "wi_cp":2}
        for _k, _v in _wi_baseline.items():
            st.session_state[_k] = _v
        st.session_state["_wi_reset_pending"] = False

    wi1, wi2 = st.columns([1, 1.1], gap="large")

    with wi1:
        st.markdown('<div class="section-header"><span>⚙️ Adjust Parameters</span></div>', unsafe_allow_html=True)
        wi_age    = st.slider("Age", 20, 100, 55, key="wi_age")
        wi_chol   = st.slider("Cholesterol (mg/dl)", 100, 600, 230, key="wi_chol")
        wi_bp     = st.slider("Blood Pressure (mmHg)", 80, 220, 130, key="wi_bp")
        wi_hr     = st.slider("Max Heart Rate (bpm)", 60, 220, 150, key="wi_hr")
        wi_weight = st.slider("Weight (kg)", 40, 160, 78, key="wi_weight")
        wi_height = st.slider("Height (cm)", 140, 210, 170, key="wi_height")
        wi_exang  = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="wi_exang")
        wi_cp     = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                        format_func=lambda x: {0:"Typical Angina",1:"Atypical",2:"Non-Anginal",3:"Asymptomatic"}[x], key="wi_cp")

        if st.button("🔄  Reset to Baseline (50, 200mg/dl, 120mmHg)", width="stretch"):
            st.session_state["_wi_reset_pending"] = True
            st.rerun()

    with wi2:
        wi_bmi = calc_bmi(wi_height, wi_weight)
        wi_vals = {"age":wi_age, "chol":wi_chol, "trestbps":wi_bp, "thalach":wi_hr,
                   "exang":wi_exang, "cp":wi_cp, "_bmi":wi_bmi}
        wi_input = pd.DataFrame([[wi_vals.get(c, 0) for c in feature_columns]], columns=feature_columns)
        wi_risk  = predict_probability(model, wi_input)

        # Baseline reference for comparison
        baseline_vals = {"age":50, "chol":200, "trestbps":120, "thalach":150, "exang":0, "cp":2}
        baseline_input = pd.DataFrame([[baseline_vals.get(c, 0) for c in feature_columns]], columns=feature_columns)
        baseline_risk  = predict_probability(model, baseline_input)

        st.markdown('<div class="section-header"><span>📊 Live Risk Comparison</span></div>', unsafe_allow_html=True)
        st.plotly_chart(whatif_risk_gauge_pair(baseline_risk, wi_risk), width="stretch")

        delta = wi_risk - baseline_risk
        delta_cls = "whatif-delta-up" if delta > 0.5 else "whatif-delta-down" if delta < -0.5 else "whatif-delta-same"
        delta_icon = "▲" if delta > 0.5 else "▼" if delta < -0.5 else "●"
        st.markdown(f"""
        <div class="metric-card {'red' if delta>0.5 else 'green' if delta<-0.5 else 'teal'}" style="text-align:center;">
            <div class="label">Change vs Baseline</div>
            <div class="value {delta_cls}">{delta_icon} {abs(delta):.1f}%</div>
            <div class="sub">{"Higher risk with these settings" if delta>0.5 else "Lower risk with these settings" if delta<-0.5 else "Roughly the same as baseline"}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        wm1, wm2, wm3 = st.columns(3)
        with wm1:
            bmi_color = "green" if wi_bmi < 25 else "amber" if wi_bmi < 30 else "red"
            st.markdown(f"""<div class="metric-card {bmi_color}">
                <div class="label">BMI</div><div class="value">{wi_bmi:.1f}</div></div>""", unsafe_allow_html=True)
        with wm2:
            chol_color = "green" if wi_chol < 200 else "amber" if wi_chol < 240 else "red"
            st.markdown(f"""<div class="metric-card {chol_color}">
                <div class="label">Cholesterol</div><div class="value">{wi_chol}</div></div>""", unsafe_allow_html=True)
        with wm3:
            bp_color = "green" if wi_bp < 120 else "amber" if wi_bp < 140 else "red"
            st.markdown(f"""<div class="metric-card {bp_color}">
                <div class="label">Blood Pressure</div><div class="value">{wi_bp}</div></div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header"><span>💡 Quick Insight</span></div>', unsafe_allow_html=True)
        insights = []
        if wi_chol > 240:
            insights.append("🍔 Lowering cholesterol below 240 mg/dl would likely reduce your risk score.")
        if wi_bp > 140:
            insights.append("🩸 Bringing blood pressure under 140 mmHg is one of the biggest single risk reducers.")
        if wi_bmi > 30:
            insights.append("⚖️ Reducing BMI below 30 typically improves both risk score and lifestyle score.")
        if wi_exang == 1:
            insights.append("⚠️ Exercise-induced angina is a strong risk driver — medical clearance before exertion is advised.")
        if not insights:
            insights.append("✅ Current settings are within generally favourable ranges.")
        for ins in insights:
            st.markdown(f'<div class="rec-pill">{ins}</div>', unsafe_allow_html=True)

# ── MODEL INSIGHTS (original — untouched) ────────────────────
elif "Model Insights" in page:
    st.markdown("""<div class="page-hero">
        <h2>📊 Model Insights & Explainability</h2>
        <p>Confusion matrix, SHAP importance, feature coefficients, and correlation analysis.</p>
    </div>""", unsafe_allow_html=True)
    X_all  = heart_df[feature_columns]
    y_true = heart_df[target_col]
    mi1,mi2 = st.columns(2, gap="large")
    with mi1:
        st.markdown('<div class="section-header"><span>🎯 Confusion Matrix</span></div>', unsafe_allow_html=True)
        try:
            y_pred = model.predict(X_all)
            cm = confusion_matrix(y_true, y_pred)
            st.plotly_chart(create_plotly_confusion_matrix(cm), width="stretch")
            tn,fp,fn,tp = cm.ravel()
            acc = (tp+tn)/(tp+tn+fp+fn)*100
            st.markdown(f"""<div style="display:flex;gap:12px;margin-top:8px;">
                <div class="metric-card green" style="flex:1;padding:14px;">
                    <div class="label">Accuracy</div><div class="value" style="font-size:1.5rem">{acc:.1f}%</div></div>
                <div class="metric-card teal" style="flex:1;padding:14px;">
                    <div class="label">True Positives</div><div class="value" style="font-size:1.5rem">{tp}</div></div>
                <div class="metric-card red" style="flex:1;padding:14px;">
                    <div class="label">False Negatives</div><div class="value" style="font-size:1.5rem">{fn}</div></div>
            </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Confusion matrix unavailable: {e}")
    with mi2:
        st.markdown('<div class="section-header"><span>⚖️ Feature Importance</span></div>', unsafe_allow_html=True)
        coef = None
        try:
            if hasattr(model,"named_steps") and "model" in model.named_steps:
                inner = model.named_steps["model"]
                coef  = inner.coef_[0] if hasattr(inner,"coef_") else None
            elif hasattr(model,"coef_"):
                coef = model.coef_[0]
            elif hasattr(model,"feature_importances_"):
                coef = model.feature_importances_
        except Exception:
            coef = None
        if coef is not None and len(coef)==len(feature_columns):
            st.plotly_chart(create_plotly_feature_importance(feature_columns, coef), width="stretch")
        else:
            st.info("Feature importance not available for this model type.")
    st.markdown('<div class="section-header"><span>🧠 SHAP Global Feature Importance</span></div>', unsafe_allow_html=True)
    with st.spinner("Computing SHAP values for all patients…"):
        try:
            sv_global = compute_shap_global(model, X_all)
            mean_abs  = np.abs(sv_global).mean(axis=0)
            sgdf = pd.DataFrame({"Feature":feature_columns,"Mean |SHAP|":mean_abs}).sort_values("Mean |SHAP|")
            fig_g = go.Figure(go.Bar(x=sgdf["Mean |SHAP|"], y=sgdf["Feature"], orientation="h",
                marker_color="#e63946", text=[f"{v:.3f}" for v in sgdf["Mean |SHAP|"]],
                textposition="outside", textfont=dict(size=10,color="#475569")))
            fig_g.update_layout(title="Mean |SHAP| — Global Feature Importance",
                xaxis_title="Mean |SHAP Value|", height=420,
                margin=dict(l=10,r=40,t=50,b=10), **PLOTLY_BASE)
            st.plotly_chart(fig_g, width="stretch")
        except Exception as e:
            st.warning(f"SHAP global plot unavailable: {e}")
    st.markdown('<div class="section-header"><span>🔥 Feature Correlation Matrix</span></div>', unsafe_allow_html=True)
    try:
        st.plotly_chart(create_plotly_correlation_heatmap(heart_df), width="stretch")
    except Exception as e:
        st.warning(f"Correlation heatmap unavailable: {e}")
    st.markdown('<div class="section-header"><span>🗂️ Dataset Preview</span></div>', unsafe_allow_html=True)
    st.dataframe(heart_df.head(20), width="stretch")

# ── FEATURE 36: MULTI-MODEL COMPARISON ───────────────────────
elif "Multi-Model Compare" in page:
    st.markdown("""<div class="page-hero">
        <h2>🏆 Multi-Model Comparison</h2>
        <p>Run the same patient through Logistic Regression, Random Forest, and XGBoost simultaneously — demonstrating robustness beyond a single algorithm's opinion.</p>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Training comparison models (cached after first run)…"):
        models_dict = multi_model.train_comparison_models(heart_df, target_col, feature_columns)

    st.markdown('<div class="section-header"><span>📊 Model Benchmark (held-out test split)</span></div>', unsafe_allow_html=True)
    bm_cols = st.columns(len(models_dict))
    for col, (name, info) in zip(bm_cols, models_dict.items()):
        with col:
            st.markdown(f"""<div class="model-card" style="border-top:3px solid {info['color']};">
                <div style="font-weight:700;color:{info['color']};font-size:0.95rem;">{name}</div>
                <div style="font-size:1.6rem;font-weight:700;margin-top:6px;">{info['accuracy']:.1f}%</div>
                <div style="font-size:0.75rem;color:#94a3b8;">Accuracy</div>
                <div style="font-size:0.85rem;color:#475569;margin-top:6px;">AUC: {info['auc']:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header"><span>🩺 Test a Patient Across All Models</span></div>', unsafe_allow_html=True)

    with st.form("mm_form"):
        mm_cols = st.columns(3)
        mm_vals = {}
        for i, col_name in enumerate(feature_columns):
            with mm_cols[i % 3]:
                mm_vals[col_name] = build_input_widget(col_name, heart_df)
        mm_sub = st.form_submit_button("🏆  Compare Models", width="stretch")

    if mm_sub:
        mm_input = pd.DataFrame([[mm_vals[c] for c in feature_columns]], columns=feature_columns)
        predictions = multi_model.predict_with_all_models(models_dict, mm_input)
        summary = multi_model.agreement_summary(predictions)

        if summary["all_agree"]:
            st.markdown(f"""<div class="agree-banner yes">
                ✅ All {len(predictions)} models agree — {"High Risk" if summary["majority"]==1 else "Low Risk"}
                (risk estimates range {summary['min_risk']:.1f}%–{summary['max_risk']:.1f}%)
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="agree-banner no">
                ⚠️ Models disagree — majority vote: {"High Risk" if summary["majority"]==1 else "Low Risk"}
                (risk estimates range {summary['min_risk']:.1f}%–{summary['max_risk']:.1f}%, spread {summary['spread']:.1f} points)
            </div>""", unsafe_allow_html=True)

        pc1, pc2 = st.columns([1, 1])
        with pc1:
            fig_cmp = go.Figure(go.Bar(
                x=[p["name"] for p in predictions],
                y=[p["risk"] for p in predictions],
                marker_color=[p["color"] for p in predictions],
                text=[f"{p['risk']:.1f}%" for p in predictions],
                textposition="outside",
            ))
            fig_cmp.update_layout(title="Risk Score by Model", yaxis_title="Risk Score (%)",
                yaxis=dict(range=[0,105]), height=360, **PLOTLY_BASE,
                margin=dict(l=10,r=10,t=50,b=10))
            fig_cmp.update_xaxes(gridcolor="#e2e8f0")
            fig_cmp.update_yaxes(gridcolor="#e2e8f0")
            st.plotly_chart(fig_cmp, width="stretch")
        with pc2:
            for p in predictions:
                st.markdown(f"""<div class="metric-card" style="border-top:3px solid {p['color']};margin-bottom:8px;">
                    <div class="label">{p['name']}</div>
                    <div class="value" style="font-size:1.4rem;color:{p['color']};">{p['risk']:.1f}%</div>
                    <div class="sub">{"🔴 High Risk" if p['prediction']==1 else "🟢 Low Risk"} · model accuracy {p['accuracy']:.1f}%</div>
                </div>""", unsafe_allow_html=True)

# ── PATIENT HISTORY (original — untouched) ───────────────────
elif "Patient History" in page:
    st.markdown("""<div class="page-hero">
        <h2>📁 Patient History</h2>
        <p>All predictions stored in the local SQLite database, newest first.</p>
    </div>""", unsafe_allow_html=True)
    df_records = pd.read_sql_query("SELECT * FROM patients ORDER BY rowid DESC", conn)
    if df_records.empty:
        st.markdown('<div class="info-panel">No records yet — run a prediction first.</div>', unsafe_allow_html=True)
    else:
        ph1,ph2,ph3 = st.columns(3)
        with ph1:
            st.markdown(f"""<div class="metric-card teal"><div class="label">Total Records</div><div class="value">{len(df_records)}</div></div>""", unsafe_allow_html=True)
        with ph2:
            hr = (df_records["prediction"]=="High Risk").sum()
            st.markdown(f"""<div class="metric-card red"><div class="label">High Risk</div><div class="value">{hr}</div></div>""", unsafe_allow_html=True)
        with ph3:
            lr = (df_records["prediction"]=="Low Risk").sum()
            st.markdown(f"""<div class="metric-card green"><div class="label">Low Risk</div><div class="value">{lr}</div></div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_records, width="stretch", height=420)
        st.download_button("📥  Export History CSV",
            data=df_records.to_csv(index=False).encode("utf-8"),
            file_name="patient_history.csv", mime="text/csv")

# ── FEATURE 31: PATIENT COMPARISON ───────────────────────────
elif "Patient Comparison" in page:
    st.markdown("""<div class="page-hero">
        <h2>👥 Multi-Patient Comparison Dashboard</h2>
        <p>Select patients from history to compare their clinical parameters side by side.</p>
    </div>""", unsafe_allow_html=True)
    df_cmp = pd.read_sql_query("SELECT * FROM patients ORDER BY rowid DESC", conn)
    if df_cmp.empty or len(df_cmp) < 2:
        st.markdown('<div class="info-panel">Need at least 2 patient records. Run more predictions first.</div>', unsafe_allow_html=True)
    else:
        if "rowid" not in df_cmp.columns:
            df_cmp = df_cmp.reset_index()
            df_cmp["rowid"] = df_cmp.index + 1
        options = [f"Patient {r} | {row.prediction} | Risk: {row.risk:.1f}%"
                   for r, row in zip(df_cmp["rowid"], df_cmp.itertuples())]
        selected = st.multiselect("Select 2 or more patients to compare:", options, default=options[:min(3,len(options))])
        if len(selected) >= 2:
            ids = [int(s.split("Patient ")[1].split(" ")[0]) for s in selected]
            result = render_patient_comparison(df_cmp, ids, feature_columns)
            if result:
                fig_cmp, subset_disp = result
                st.plotly_chart(fig_cmp, width="stretch")
                st.markdown('<div class="section-header"><span>📊 Side-by-Side Data Table</span></div>', unsafe_allow_html=True)
                st.dataframe(subset_disp.T, width="stretch")
        else:
            st.info("Select at least 2 patients to enable comparison.")

# ── FEATURE 37: RISK FACTOR "WHAT CHANGED" DIFF TOOL ─────────
elif "Risk Diff Tool" in page:
    st.markdown("""<div class="page-hero">
        <h2>🆚 Risk Factor "What Changed" Diff Tool</h2>
        <p>Compare the same or two different patients' visits to see exactly which parameters changed and by how much — useful for tracking progress between visits.</p>
    </div>""", unsafe_allow_html=True)

    df_diff = pd.read_sql_query("SELECT * FROM patients ORDER BY rowid DESC", conn)
    if df_diff.empty or len(df_diff) < 2:
        st.markdown('<div class="info-panel">Need at least 2 patient records to compare. Run more predictions first.</div>', unsafe_allow_html=True)
    else:
        if "rowid" not in df_diff.columns:
            df_diff = df_diff.reset_index()
            df_diff["rowid"] = df_diff.index + 1

        diff_options = [f"Patient {r} | {row.prediction} | Risk: {row.risk:.1f}% | {row.created_at}"
                         for r, row in zip(df_diff["rowid"], df_diff.itertuples())]

        dd1, dd2 = st.columns(2)
        with dd1:
            visit_a_label = st.selectbox("Visit A (earlier)", diff_options, index=min(1, len(diff_options)-1))
        with dd2:
            visit_b_label = st.selectbox("Visit B (later)", diff_options, index=0)

        id_a = int(visit_a_label.split("Patient ")[1].split(" ")[0])
        id_b = int(visit_b_label.split("Patient ")[1].split(" ")[0])

        if id_a == id_b:
            st.warning("Select two different records to compare.")
        else:
            row_a = df_diff[df_diff["rowid"] == id_a].iloc[0]
            row_b = df_diff[df_diff["rowid"] == id_b].iloc[0]

            compare_cols = ["age","chol","trestbps","thalach","bmi","lifestyle_score","risk"]
            diffs = compute_patient_diff(row_a, row_b, compare_cols)

            risk_delta = float(row_b["risk"]) - float(row_a["risk"])
            rd1, rd2, rd3 = st.columns(3)
            with rd1:
                st.markdown(f"""<div class="metric-card teal">
                    <div class="label">Visit A Risk</div><div class="value">{row_a['risk']:.1f}%</div></div>""", unsafe_allow_html=True)
            with rd2:
                st.markdown(f"""<div class="metric-card teal">
                    <div class="label">Visit B Risk</div><div class="value">{row_b['risk']:.1f}%</div></div>""", unsafe_allow_html=True)
            with rd3:
                rcolor = "red" if risk_delta > 0.5 else "green" if risk_delta < -0.5 else "teal"
                ricon  = "▲" if risk_delta > 0.5 else "▼" if risk_delta < -0.5 else "●"
                st.markdown(f"""<div class="metric-card {rcolor}">
                    <div class="label">Risk Change</div>
                    <div class="value">{ricon} {abs(risk_delta):.1f}%</div></div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header"><span>📋 Parameter-by-Parameter Changes</span></div>', unsafe_allow_html=True)
            for d in diffs:
                icon  = "▲" if d["Direction"]=="up" else "▼" if d["Direction"]=="down" else "●"
                color = "#e63946" if d["Direction"]=="up" else "#2a9d5c" if d["Direction"]=="down" else "#94a3b8"
                st.markdown(f"""<div class="diff-row">
                    <span class="diff-label">{d['Parameter']}</span>
                    <span class="diff-values">
                        <span style="color:#94a3b8;">{d['Visit A']}</span>
                        <span style="color:#cbd5e1;">→</span>
                        <span style="color:#0f172a;font-weight:600;">{d['Visit B']}</span>
                        <span style="color:{color};font-weight:700;">{icon} {abs(d['Change'])}</span>
                    </span>
                </div>""", unsafe_allow_html=True)

            if risk_delta < -2:
                st.markdown("""<div class="alert-card info" style="margin-top:14px;">
                    <div class="alert-icon">✅</div>
                    <div><div class="alert-type">IMPROVEMENT DETECTED</div>
                    <div class="alert-msg">Risk score decreased between visits — current lifestyle changes appear to be effective. Continue the same approach.</div></div>
                </div>""", unsafe_allow_html=True)
            elif risk_delta > 2:
                st.markdown("""<div class="alert-card warning" style="margin-top:14px;">
                    <div class="alert-icon">⚠️</div>
                    <div><div class="alert-type">RISK INCREASED</div>
                    <div class="alert-msg">Risk score increased between visits. Review which parameters worsened above and consider discussing this trend with a healthcare professional.</div></div>
                </div>""", unsafe_allow_html=True)

# ── ANALYTICS (original — untouched) ─────────────────────────
elif "Analytics" in page:
    st.markdown("""<div class="page-hero">
        <h2>📈 Real-Time Analytics Dashboard</h2>
        <p>Aggregated insights from all patient predictions in the database.</p>
    </div>""", unsafe_allow_html=True)
    df_an = pd.read_sql_query("SELECT * FROM patients ORDER BY rowid ASC", conn)
    if df_an.empty:
        st.markdown('<div class="info-panel">No data yet — run at least one prediction first.</div>', unsafe_allow_html=True)
    else:
        high = (df_an["prediction"]=="High Risk").sum()
        low  = (df_an["prediction"]=="Low Risk").sum()
        a1,a2,a3,a4 = st.columns(4)
        with a1:
            st.markdown(f"""<div class="metric-card teal"><div class="label">Total Predictions</div><div class="value">{len(df_an)}</div></div>""", unsafe_allow_html=True)
        with a2:
            avg_r = df_an["risk"].mean()
            st.markdown(f"""<div class="metric-card {'red' if avg_r>50 else 'green'}"><div class="label">Avg Risk Score</div><div class="value">{avg_r:.1f}%</div></div>""", unsafe_allow_html=True)
        with a3:
            st.markdown(f"""<div class="metric-card amber"><div class="label">Avg BMI</div><div class="value">{df_an['bmi'].mean():.1f}</div></div>""", unsafe_allow_html=True)
        with a4:
            st.markdown(f"""<div class="metric-card green"><div class="label">Avg Lifestyle Score</div><div class="value">{df_an['lifestyle_score'].mean():.0f}</div></div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        ch1,ch2 = st.columns(2, gap="large")
        with ch1:
            st.plotly_chart(create_analytics_pie(int(high),int(low)), width="stretch")
        with ch2:
            st.plotly_chart(create_risk_histogram(df_an["risk"].tolist()), width="stretch")
        fig_trend = create_risk_trend_forecast(df_an.copy())
        if fig_trend:
            st.plotly_chart(fig_trend, width="stretch")
        else:
            st.markdown('<div class="info-panel">Need at least 2 records to plot the risk trend.</div>', unsafe_allow_html=True)

# ── FEATURE 26: PATIENT SEGMENTATION ─────────────────────────
elif "Patient Segmentation" in page:
    st.markdown("""<div class="page-hero">
        <h2>🔬 Patient Segmentation — Clustering AI</h2>
        <p>K-Means clustering groups patients by shared clinical risk profiles — used in hospital population health management.</p>
    </div>""", unsafe_allow_html=True)
    df_seg_data = pd.read_sql_query("SELECT * FROM patients ORDER BY rowid ASC", conn)
    if len(df_seg_data) < 3:
        st.markdown('<div class="info-panel">Need at least 3 patient records for clustering. Run more predictions first.</div>', unsafe_allow_html=True)
    else:
        fig_seg, df_seg_result = segment_patients(df_seg_data)
        if fig_seg:
            seg_counts = df_seg_result["Cluster"].value_counts()
            sc1,sc2,sc3 = st.columns(3)
            cluster_list = seg_counts.index.tolist()
            colors_map = {"Lifestyle Risk":"amber","High Clinical Risk":"red","Age-Based Risk":"teal"}
            for i, (col, cname) in enumerate(zip([sc1,sc2,sc3], cluster_list)):
                with col:
                    cc = colors_map.get(cname,"teal")
                    st.markdown(f"""<div class="metric-card {cc}">
                        <div class="label">{cname}</div>
                        <div class="value">{seg_counts[cname]}</div>
                        <div class="sub">patients in cluster</div></div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.plotly_chart(fig_seg, width="stretch")
            st.markdown('<div class="section-header"><span>📊 Cluster Assignment Table</span></div>', unsafe_allow_html=True)
            st.dataframe(df_seg_result.drop(columns=["Color"], errors="ignore"), width="stretch")
        else:
            st.warning("Clustering requires numeric clinical features in the database.")

# ── FEATURE 29: IoT MONITOR ───────────────────────────────────
elif "IoT Monitor" in page:
    st.markdown("""<div class="page-hero">
        <h2>📡 Real-Time IoT Vital Signs Monitor</h2>
        <p>Simulates wearable device output — heart rate, blood pressure, and SpO₂ — as a live monitoring feed.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="info-panel">Configure baseline vitals and simulate a real-time monitoring session. Each run generates a new waveform based on the configured baseline with physiological variation.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("iot_form"):
        io1, io2, io3 = st.columns(3)
        with io1:
            base_hr   = st.number_input("Baseline Heart Rate (bpm)", 50, 150, 72)
        with io2:
            base_bp   = st.number_input("Baseline Systolic BP (mmHg)", 90, 180, 120)
        with io3:
            base_spo2 = st.number_input("Baseline SpO₂ (%)", 92, 100, 98)
        cycles = st.slider("Monitoring Duration (seconds)", 10, 60, 30)
        iot_sub = st.form_submit_button("📡  Start Monitoring Simulation", width="stretch")

    if iot_sub:
        with st.spinner("Streaming vitals…"):
            fig_iot, avg_hr, avg_bp, avg_spo2 = simulate_iot_vitals(base_hr, base_bp, base_spo2, cycles)

        i1,i2,i3,i4 = st.columns(4)
        with i1:
            hr_color = "red" if avg_hr > 100 or avg_hr < 60 else "green"
            st.markdown(f"""<div class="metric-card {hr_color}">
                <div class="label">Avg Heart Rate</div><div class="value">{avg_hr} bpm</div>
                <div class="sub">{"⚠️ Abnormal" if hr_color=="red" else "✅ Normal"}</div></div>""", unsafe_allow_html=True)
        with i2:
            bp_color = "red" if avg_bp > 140 else "amber" if avg_bp > 120 else "green"
            st.markdown(f"""<div class="metric-card {bp_color}">
                <div class="label">Avg Systolic BP</div><div class="value">{avg_bp} mmHg</div>
                <div class="sub">{"⚠️ High" if bp_color=="red" else "⚡ Elevated" if bp_color=="amber" else "✅ Normal"}</div></div>""", unsafe_allow_html=True)
        with i3:
            spo2_color = "red" if avg_spo2 < 95 else "green"
            st.markdown(f"""<div class="metric-card {spo2_color}">
                <div class="label">Avg SpO₂</div><div class="value">{avg_spo2}%</div>
                <div class="sub">{"⚠️ Low — seek attention" if spo2_color=="red" else "✅ Normal"}</div></div>""", unsafe_allow_html=True)
        with i4:
            overall = "red" if (avg_hr>100 or avg_bp>140 or avg_spo2<95) else "green"
            st.markdown(f"""<div class="metric-card {overall}">
                <div class="label">Overall Status</div>
                <div class="value" style="font-size:1.1rem;margin-top:6px;">{"🔴 Review" if overall=="red" else "🟢 Stable"}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(fig_iot, width="stretch")

# ── FEATURE 30: ANOMALY DETECTION ────────────────────────────
elif "Anomaly Detection" in page:
    st.markdown("""<div class="page-hero">
        <h2>🚨 Anomaly Detection System</h2>
        <p>Detects statistically unusual patient values relative to the training dataset population using Z-score analysis.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="info-panel">Enter patient values below. The system will flag any measurements that deviate significantly from the population baseline (|Z-score| > 2.5 = anomaly, |Z-score| > 3.5 = critical anomaly).</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("anomaly_form"):
        an_cols = st.columns(3)
        an_vals = {}
        for i, col_name in enumerate(feature_columns):
            with an_cols[i % 3]:
                an_vals[col_name] = build_input_widget(col_name, heart_df)
        an_sub = st.form_submit_button("🔍  Run Anomaly Detection", width="stretch")

    if an_sub:
        anomalies = detect_anomalies(an_vals, heart_df, feature_columns)
        if not anomalies:
            st.markdown("""<div class="alert-card info">
                <div class="alert-icon">✅</div>
                <div><div class="alert-type">NO ANOMALIES DETECTED</div>
                <div class="alert-msg">All patient values fall within 2.5 standard deviations of the population mean. No statistically unusual measurements identified.</div></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge badge-red">{len(anomalies)} Anomal{"y" if len(anomalies)==1 else "ies"} Detected</span><br><br>', unsafe_allow_html=True)
            for a in anomalies:
                cls = "critical" if a["Severity"]=="critical" else "warning"
                st.markdown(f"""<div class="alert-card {cls}">
                    <div class="alert-icon">{"🔴" if cls=="critical" else "🟠"}</div>
                    <div>
                        <div class="alert-type">{a['Feature']} — ANOMALY (Z={a['Z-Score']:+.2f})</div>
                        <div class="alert-msg">
                            Patient value: <b>{a['Value']}</b> &nbsp;|&nbsp;
                            Population mean: <b>{a['Population Mean']}</b> &nbsp;|&nbsp;
                            {a['Status']}
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

            an_df = pd.DataFrame(anomalies).drop(columns=["Severity"])
            st.markdown('<div class="section-header"><span>📊 Anomaly Summary Table</span></div>', unsafe_allow_html=True)
            st.dataframe(an_df, width="stretch")

# ── MEDICINE ASSISTANT (original — untouched) ─────────────────
elif "Medicine" in page:
    st.markdown("""<div class="page-hero">
        <h2>💊 Medicine Suggestion System</h2>
        <p>Search the medicine database by symptom, disease, or drug name.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="info-panel">⚕️ <b>Educational purposes only.</b> This is not a prescription. Always consult a licensed physician.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if medicine_df is None:
        st.warning("Medicine dataset not found. Add `medicine_dataset.csv` to the project folder.")
    else:
        query = st.text_input("🔍  Search", placeholder="e.g. chest pain, hypertension, aspirin…")
        if query:
            matches = search_all_text_columns(medicine_df, query)
            if matches.empty:
                st.markdown('<div class="info-panel">No matches found. Try a different keyword.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="badge badge-teal">Found {len(matches)} result(s)</span><br><br>', unsafe_allow_html=True)
                st.dataframe(matches.head(10), width="stretch")

# ── SYMPTOM CHECKER (original — untouched) ───────────────────
elif "Symptom" in page:
    st.markdown("""<div class="page-hero">
        <h2>🧬 AI Symptom Checker</h2>
        <p>Describe a symptom to find possible associated conditions from the dataset.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="info-panel">⚕️ <b>Not a medical diagnosis.</b> Always consult a qualified healthcare professional.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if symptom_df is None:
        st.warning("Symptom dataset not found. Add `symptoms_dataset.csv` to the project folder.")
    else:
        user_symptom = st.text_input("🔍  Describe your symptom",
            placeholder="e.g. chest tightness, palpitations, shortness of breath…")
        if user_symptom:
            results = search_all_text_columns(symptom_df, user_symptom)
            if results.empty:
                st.markdown('<div class="info-panel">Symptom not recognised. Please consult a medical professional.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="badge badge-teal">Found {len(results)} match(es)</span><br><br>', unsafe_allow_html=True)
                st.dataframe(results.head(10), width="stretch")
                for _, row in results.head(3).iterrows():
                    if "condition" in results.columns:
                        st.markdown(f"""<div class="alert-card info" style="margin-top:10px;">
                            <div class="alert-icon">🔎</div>
                            <div><div class="alert-type">Possible Condition</div>
                            <div class="alert-msg">{row.get('condition','–')}</div></div>
                        </div>""", unsafe_allow_html=True)
                    if "advice" in results.columns and str(row.get("advice","")).strip():
                        st.markdown(f"""<div class="alert-card info">
                            <div class="alert-icon">💬</div>
                            <div><div class="alert-type">Advice</div>
                            <div class="alert-msg">{row.get('advice','–')}</div></div>
                        </div>""", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

# ── REPORTS & BATCH (original — untouched) ───────────────────
elif "Reports" in page:
    st.markdown("""<div class="page-hero">
        <h2>📄 Reports & Bulk Prediction</h2>
        <p>Upload a CSV for batch predictions, or download individual PDF reports from the Prediction tab.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span>📊 Bulk CSV Prediction</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-panel">Upload a CSV with these columns: <code>{", ".join(feature_columns)}</code></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded_file:
        st.markdown('<span class="badge badge-green">File uploaded successfully</span><br><br>', unsafe_allow_html=True)
        if st.button("🚀  Run Bulk Predictions"):
            with st.spinner("Processing predictions…"):
                result_df, error = process_bulk_predictions(uploaded_file, model, feature_columns)
            if error:
                st.error(f"Error: {error}")
            else:
                b1,b2,b3 = st.columns(3)
                with b1:
                    st.markdown(f"""<div class="metric-card teal"><div class="label">Processed</div><div class="value">{len(result_df)}</div></div>""", unsafe_allow_html=True)
                with b2:
                    hr = (result_df["Prediction"]=="High Risk").sum()
                    st.markdown(f"""<div class="metric-card red"><div class="label">High Risk</div><div class="value">{hr}</div></div>""", unsafe_allow_html=True)
                with b3:
                    lr = (result_df["Prediction"]=="Low Risk").sum()
                    st.markdown(f"""<div class="metric-card green"><div class="label">Low Risk</div><div class="value">{lr}</div></div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(result_df, width="stretch")
                st.download_button("📥  Download Results CSV",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="bulk_predictions.csv", mime="text/csv")
    st.divider()
    st.markdown('<div class="section-header"><span>📋 Individual PDF Reports</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-panel">After each prediction in the <b>Prediction</b> tab, a personalised PDF report is available for download on that page.</div>', unsafe_allow_html=True)

# ── ADVANCED (original voice assistant + diagnostics) ─────────
elif "Advanced" in page:
    st.markdown("""<div class="page-hero">
        <h2>⚙️ Advanced Features</h2>
        <p>Voice assistant, cardiac reference guide, and system diagnostics.</p>
    </div>""", unsafe_allow_html=True)
    adv1, adv2 = st.columns(2, gap="large")
    with adv1:
        st.markdown('<div class="section-header"><span>🎤 Jarvis Voice Assistant</span></div>', unsafe_allow_html=True)
        voice_cmds = [
            ("'Predict heart risk'",    "Opens Prediction page"),
            ("'Show analytics'",        "Opens Analytics dashboard"),
            ("'Generate report'",       "Creates PDF health report"),
            ("'Medicine suggestions'",  "Opens Medicine Assistant"),
            ("'Symptom checker'",       "Opens Symptom Checker"),
            ("'Show trends'",           "Opens Analytics trends"),
            ("'Check alerts'",          "Runs smart alert checks"),
            ("'Help'",                  "Lists all commands"),
        ]
        st.markdown('<div class="info-panel">', unsafe_allow_html=True)
        for cmd, action in voice_cmds:
            st.markdown(f'<div class="cmd-row"><span class="cmd-text">🎙 {cmd}</span><span class="cmd-action">→ {action}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        import streamlit.components.v1 as components
        components.html("""<!DOCTYPE html><html><head><style>
        *{box-sizing:border-box;margin:0;padding:0;font-family:'Inter',sans-serif;}
        body{background:transparent;padding:4px;}
        #jarvis-box{background:linear-gradient(135deg,#fff,#fde8ea);border:1px solid #f5c2c7;border-left:4px solid #e63946;border-radius:10px;padding:16px 18px;}
        #mic-btn{display:flex;align-items:center;gap:10px;background:linear-gradient(135deg,#e63946,#b02030);color:white;border:none;border-radius:8px;padding:10px 20px;cursor:pointer;font-size:0.95rem;font-weight:600;box-shadow:0 4px 14px rgba(230,57,70,0.35);transition:all 0.2s;width:100%;justify-content:center;}
        #mic-btn.listening{background:linear-gradient(135deg,#b02030,#7a0f1e);animation:pulse 1s infinite;}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.75}}
        #status{font-size:0.8rem;color:#475569;margin-top:10px;min-height:18px;}
        #transcript{margin-top:10px;padding:10px 14px;background:#fff;border:1px solid #e2e8f0;border-radius:8px;font-size:0.88rem;color:#0f172a;min-height:40px;line-height:1.5;display:none;}
        #response{margin-top:10px;padding:10px 14px;background:#e0f0fa;border:1px solid #bddff5;border-left:4px solid #0077b6;border-radius:8px;font-size:0.88rem;color:#0f172a;min-height:40px;line-height:1.5;display:none;}
        </style></head><body>
        <div id="jarvis-box">
        <button id="mic-btn" onclick="toggleListen()">🎤 &nbsp; Activate Jarvis</button>
        <div id="status">Click the button and speak a command.</div>
        <div id="transcript"></div><div id="response"></div>
        </div>
        <script>
        const COMMANDS={"predict heart risk":"🩺 Navigate to Prediction in the sidebar.","show analytics":"📈 Navigate to Analytics in the sidebar.","generate report":"📄 Navigate to Reports & Batch tab.","medicine suggestions":"💊 Navigate to Medicine Assistant.","symptom checker":"🧬 Navigate to Symptom Checker.","show trends":"📊 Check the Analytics dashboard for trends.","check alerts":"🚨 Alerts appear after each prediction.","help":"Available: predict heart risk, show analytics, generate report, medicine suggestions, symptom checker, show trends, check alerts."};
        let recog=null,listening=false;
        const btn=document.getElementById('mic-btn'),status=document.getElementById('status'),transcript=document.getElementById('transcript'),response=document.getElementById('response');
        const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
        if(SR){recog=new SR();recog.lang='en-US';recog.interimResults=false;recog.maxAlternatives=1;
        recog.onstart=()=>{listening=true;btn.textContent='🔴  Listening…';btn.classList.add('listening');status.textContent='Listening… speak now.';transcript.style.display='none';response.style.display='none';};
        recog.onresult=(e)=>{const said=e.results[0][0].transcript.toLowerCase().trim();transcript.style.display='block';transcript.innerHTML='🗣️ You said: <b>"'+e.results[0][0].transcript+'"</b>';const match=Object.keys(COMMANDS).find(k=>said.includes(k));response.style.display='block';response.innerHTML=match?COMMANDS[match]:'❓ Command not recognised. Say "Help" for available commands.';if(window.speechSynthesis){const u=new SpeechSynthesisUtterance(response.textContent.replace(/[🩺📈📄💊🧬📊🚨❓]/g,''));u.lang='en-US';window.speechSynthesis.speak(u);}};
        recog.onerror=(e)=>{status.textContent='⚠️ '+e.error;};
        recog.onend=()=>{listening=false;btn.innerHTML='🎤 &nbsp; Activate Jarvis';btn.classList.remove('listening');status.textContent='Done. Click again to speak another command.';};}
        function toggleListen(){if(!recog)return;if(listening)recog.stop();else try{recog.start();}catch(e){status.textContent='⚠️ '+e.message;}}
        </script></body></html>""", height=280)

    with adv2:
        st.markdown('<div class="section-header"><span>🫀 Cardiac Monitoring Reference</span></div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-panel">
            <b style="color:#e63946">Heart Location</b><br>
            Centre-left of chest · Behind the sternum · Between both lungs<br><br>
            <b style="color:#e07b39">Warning Signs to Monitor</b><br>
            🔴 Chest pain or pressure &nbsp; 🔴 Left arm numbness<br>
            🔴 Jaw or neck discomfort &nbsp; 🔴 Upper back pain<br>
            🔴 Shortness of breath &nbsp;&nbsp;&nbsp;&nbsp; 🔴 Cold sweats + nausea<br><br>
            <b style="color:#2a9d5c">When to Call Emergency Services</b><br>
            Sudden severe chest pain · Pain radiating to arm or jaw<br>
            Difficulty breathing · Loss of consciousness<br><br>
            📞 <b>Emergency: 112 / 102 (India)</b>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span>ℹ️ System Diagnostics</span></div>', unsafe_allow_html=True)
    total_preds  = pd.read_sql_query("SELECT COUNT(*) as n FROM patients", conn).iloc[0]["n"]
    avg_risk_all = pd.read_sql_query("SELECT AVG(risk) as r FROM patients", conn).iloc[0]["r"]
    st.markdown(f"""<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
        <div class="sidebar-stat"><span class="k">Platform Version</span><span class="v">5.0</span></div>
        <div class="sidebar-stat"><span class="k">Model Type</span><span class="v">Logistic Regression Pipeline</span></div>
        <div class="sidebar-stat"><span class="k">Total Features</span><span class="v">37 (23 original + 14 new)</span></div>
        <div class="sidebar-stat"><span class="k">Dataset Rows</span><span class="v">{len(heart_df):,}</span></div>
        <div class="sidebar-stat"><span class="k">Total Predictions</span><span class="v">{int(total_preds)}</span></div>
        <div class="sidebar-stat"><span class="k">Overall Avg Risk</span><span class="v">{f"{avg_risk_all:.1f}%" if avg_risk_all else "N/A"}</span></div>
        <div class="sidebar-stat"><span class="k">Medicine DB</span><span class="v">{"✅ Loaded" if medicine_df is not None else "❌ Not found"}</span></div>
        <div class="sidebar-stat"><span class="k">Symptom DB</span><span class="v">{"✅ Loaded" if symptom_df is not None else "❌ Not found"}</span></div>
    </div>""", unsafe_allow_html=True)

# =============================================================
# FOOTER
# =============================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:20px;border-top:2px solid #e2e8f0;margin-top:20px;">
    <span style="font-size:0.75rem;color:#94a3b8;letter-spacing:0.5px;">
        🫀 CardioAI Heart Intelligence Platform v5.0 &nbsp;·&nbsp;
        37 Features &nbsp;·&nbsp;
        For educational purposes only &nbsp;·&nbsp;
        Always consult a qualified healthcare professional
    </span>
</div>
""", unsafe_allow_html=True)
