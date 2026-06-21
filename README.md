# 🫀 CardioAI — Heart Intelligence Platform (v5.0)

An AI-powered heart disease risk prediction platform built with scikit-learn, SHAP, and Groq — 37 features across prediction, explainability, real-time voice control, simulation, and analytics.

---

## ✨ What's New in v5.0

| # | Feature | Page |
|---|---|---|
| 34 | **Real-Time Voice Control** — speak a page name, app navigates automatically (Groq Whisper) | 🎙️ Voice Control |
| — | **AI Health Assistant chatbot** — ask questions, get plain-language answers (Groq GPT-OSS-120B) | 🎙️ Voice Control |
| 35 | **What-If Risk Simulator** — drag sliders, risk recalculates live, no submit button | 🧪 What-If Simulator |
| 36 | **Multi-Model Comparison** — Logistic Regression vs Random Forest vs XGBoost on the same patient | 🏆 Multi-Model Compare |
| 37 | **Risk Factor "What Changed" Diff Tool** — compare two visits, see exact deltas | 🆚 Risk Diff Tool |

Plus all 33 features from v4.0 (AI Clinical Engine, Digital Twin, Patient Segmentation, IoT Monitor, Anomaly Detection, and the original 23).

---

## 🚀 Quick Start (Local)

### 1. Clone or extract this project
```bash
cd CardioAI_Final
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional but recommended) Add your Groq API key
Voice Control and the AI chatbot need a free Groq API key. Everything else in the app works without it.

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Open `.streamlit/secrets.toml` and paste your key:
```toml
GROQ_API_KEY = "gsk_your_real_key_here"
```
Get a free key at **https://console.groq.com/keys**

`.streamlit/secrets.toml` is in `.gitignore` — it will never be committed or pushed to GitHub, and the key is never displayed anywhere in the app UI, logs, or error messages.

### 5. (Optional) Retrain the base model
```bash
python train_model.py
```

### 6. Run the app
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this project to a **public or private GitHub repository** (the `.gitignore` already excludes `venv/`, `secrets.toml`, and `patients.db`).
2. Go to **https://share.streamlit.io** → **New app** → select your repo and `app.py` as the entry point.
3. Before clicking Deploy, open **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_real_key_here"
   ```
   This is the cloud equivalent of your local `secrets.toml` — the key is stored encrypted by Streamlit Cloud and is never exposed in your repo, logs, or to end users.
4. Click **Deploy**. First build takes 2–5 minutes (installing scikit-learn, SHAP, XGBoost, etc.).

The app works fully **without** a Groq key — only the Voice Control and AI Chat page will show a "not configured" notice and everything else (all 35 other features) functions normally.

---

## 📁 Project Structure

```
CardioAI_Final/
├── app.py                        ← Main Streamlit application (run this)
├── train_model.py                ← Retrains the base Logistic Regression model
├── requirements.txt              ← All Python dependencies
├── .gitignore                    ← Excludes venv, secrets, cache, local DB
├── .streamlit/
│   ├── config.toml               ← Forces light theme, high contrast (works even if viewer's OS is dark mode)
│   └── secrets.toml.example      ← Template — copy to secrets.toml and fill in your key
├── modules/
│   ├── __init__.py
│   ├── groq_client.py            ← Groq API wrapper (voice transcription + chat, key never exposed)
│   └── multi_model.py            ← Trains & caches LR / Random Forest / XGBoost for comparison
├── heart_disease_data.csv        ← Training dataset (303 patients, 13 features)
├── heart_model.pkl               ← Pre-trained Logistic Regression pipeline
├── medicine_dataset.csv          ← Medicine reference database
├── symptoms_dataset.csv          ← Disease ↔ symptom lookup (compact, searchable format)
├── human_heart.png               ← Heart anatomy image shown in Prediction tab
└── patients.db                   ← SQLite database (auto-created/updated; excluded from git)
```

---

## 🔐 Security Notes

- The Groq API key is read **only** via `st.secrets` (cloud) or `.streamlit/secrets.toml` (local) — never hardcoded.
- The key is **never** printed, logged, displayed in the UI, or included in error messages. All failures fail closed with a generic, user-safe message.
- `secrets.toml` and `patients.db` are excluded from version control by `.gitignore`.
- The app uses `whisper-large-v3-turbo` (speech-to-text) and `openai/gpt-oss-120b` (chat) — both current, non-deprecated Groq models as of this build.

---

## 🎙️ Using Voice Control

1. Open the **🎙️ Voice Control** page from the sidebar.
2. Click the microphone recorder, say a page name (e.g. *"Open Digital Twin"*, *"Show analytics"*, *"Go to medicine assistant"*), then stop recording.
3. The app transcribes your speech via Groq Whisper and automatically switches to the matching page.
4. Use the chat box on the right to ask general heart-health questions — answered by the Groq-powered AI assistant.

If no Groq key is configured, this page shows a clear setup notice; every other page in the app still works normally.

---

## ⚠️ Disclaimer

CardioAI is an **educational and demonstration platform**. It is not a medical device and its predictions are not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified cardiologist or healthcare professional for any cardiac concerns.

---

**Version:** 5.0 · **Total Features:** 37 · **License:** Educational use only
