# 🫀 CardioAI — Heart Intelligence Platform (v5.0)

An AI-powered heart disease risk prediction platform built with Streamlit, scikit-learn, SHAP, and Groq — **37 features** across prediction, explainability, real-time voice control, simulation, and analytics.

---

## ✅ All 37 Features

### 🩺 Prediction Page
| # | Feature | What it does |
|---|---|---|
| 1 | Heart Disease Prediction | Core ML prediction from 13 clinical inputs |
| 2 | ML Pipeline Model | StandardScaler + Logistic Regression pipeline |
| 3 | Pickle Model Loading | Pre-trained model loads instantly from `heart_model.pkl` |
| 4 | Explainable AI (SHAP) | Per-patient SHAP waterfall — shows why the model decided what it decided |
| 5 | BMI Calculator | Auto-calculated from height/weight, with category label |
| 6 | Lifestyle Score Indicator | 0–100 score from BMI, cholesterol, BP, exercise angina |
| 7 | Risk Percentage Score | Calibrated probability of high cardiovascular risk |
| 8 | Health Recommendation Engine | Personalised advice based on prediction + individual risk factors |
| 9 | Smart Alert System | Critical/warning/info alerts, including combined multi-condition triggers |
| 15 | Risk Gauge Meter | Interactive Plotly gauge — green/amber/red zones |
| 19 | PDF Health Report Generator | One-click downloadable, formatted clinical PDF |
| 21 | Heart Visualization | Anatomical heart image shown alongside the form |

### 🎙️ Voice Control Page
| # | Feature | What it does |
|---|---|---|
| 34 | Real-Time Voice Control | Speak a page name — Groq Whisper transcribes, app auto-navigates |
| — | AI Health Assistant Chatbot | Ask questions in plain English, answered by Groq GPT-OSS-120B |

### 🧠 AI Clinical Engine Page
| # | Feature | What it does |
|---|---|---|
| 24 | AI Clinical Decision Engine | Doctor-style structured findings, urgency level, and recommended action |
| 28 | Auto Medical Report Writer | Generates downloadable clinical notes in professional documentation style |
| 32 | Disease Progression Timeline | Visual stage indicator: Healthy → At Risk → High Risk → Critical |

### 🔮 Digital Twin Page
| # | Feature | What it does |
|---|---|---|
| 25 | Digital Twin Future Risk Simulation | Projects risk trajectory 3–6 months ahead based on current trends |

### 🧪 What-If Simulator Page
| # | Feature | What it does |
|---|---|---|
| 35 | What-If Risk Simulator | Live sliders — risk recalculates instantly, no submit button needed |

### 🏆 Multi-Model Compare Page
| # | Feature | What it does |
|---|---|---|
| 36 | Multi-Model Comparison | Runs the same patient through Logistic Regression, Random Forest, and XGBoost simultaneously, with an agreement summary |

### 📊 Model Insights Page
| # | Feature | What it does |
|---|---|---|
| 4 (global) | Global SHAP Feature Importance | Dataset-wide explanation of which features drive predictions most |
| 13 | Correlation Heatmap | Interactive feature correlation matrix |
| — | Confusion Matrix & Accuracy | Model performance on a held-out test split |
| — | Feature Importance (Coefficients) | Logistic Regression coefficient magnitudes, visualised |

### 📁 Patient History Page
| # | Feature | What it does |
|---|---|---|
| 16 | SQLite Patient Database | Every prediction automatically saved locally |
| 17 | Patient History Viewer | Searchable, sortable table with CSV export |

### 👥 Patient Comparison Page
| # | Feature | What it does |
|---|---|---|
| 31 | Multi-Patient Comparison Dashboard | Select 2+ saved patients, compare key metrics side-by-side |

### 🆚 Risk Diff Tool Page
| # | Feature | What it does |
|---|---|---|
| 37 | Risk Factor "What Changed" Diff Tool | Pick two visits, see exact parameter-by-parameter deltas and risk change |

### 📈 Analytics Page
| # | Feature | What it does |
|---|---|---|
| 12 | Real-Time Analytics Dashboard | Live KPI cards, charts, and trend updates from the full patient database |
| 14 | Risk Trend Forecast Graph | Moving-average forecast line over all saved predictions |

### 🔬 Patient Segmentation Page
| # | Feature | What it does |
|---|---|---|
| 26 | Patient Segmentation (Clustering AI) | K-Means groups patients into Lifestyle Risk / High Clinical Risk / Age-Based Risk clusters |

### 📡 IoT Monitor Page
| # | Feature | What it does |
|---|---|---|
| 29 | Real-Time Monitoring Simulation | Simulated live heart rate, blood pressure, and SpO₂ waveform — IoT-ready design pattern |

### 🚨 Anomaly Detection Page
| # | Feature | What it does |
|---|---|---|
| 30 | Anomaly Detection System | Z-score analysis flags statistically unusual patient values vs. the training population |

### 💊 Medicine Assistant Page
| # | Feature | What it does |
|---|---|---|
| 10 | Medicine Suggestion System | Full-text search across the medicine reference dataset |

### 🧬 Symptom Checker Page
| # | Feature | What it does |
|---|---|---|
| 11 | AI Symptom Checker | Search symptoms, get possible conditions, severity, and advice |

### 📄 Reports & Batch Page
| # | Feature | What it does |
|---|---|---|
| 20 | Bulk CSV Prediction | Upload a CSV of multiple patients, get predictions for all of them at once |
| — | Individual PDF Reports | Linked back to the Prediction page's report generator |

### ⚙️ Advanced Page
| # | Feature | What it does |
|---|---|---|
| 22 | Interactive Cardiac Reference Guide | Heart location, warning signs, emergency guidance |
| 27 | Intelligent Multi-Condition Alert Engine | Combined-trigger alerts (e.g. High BP + High Cholesterol + Age >60) |
| 33 | Jarvis Voice Command Reference | Full list of supported voice commands |
| 18 | System Diagnostics | Live platform version, model type, dataset stats, prediction counts |

---

## 🏗️ Platform-Level Features (apply across all pages)
| # | Feature |
|---|---|
| 18 | Multi-Page Professional Dashboard (sidebar navigation, light theme, consistent branding) |
| 23 | Voice command vocabulary shared between the chatbot and navigation engine |

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
Voice Control and the AI chatbot need a free Groq API key. Every other feature works without it.

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Open `.streamlit/secrets.toml` and paste your key:
```toml
GROQ_API_KEY = "gsk_your_real_key_here"
```
Get a free key at **https://console.groq.com/keys**

`.streamlit/secrets.toml` is in `.gitignore` — it will never be committed or pushed to GitHub, and the key is never displayed anywhere in the app UI, logs, or error messages.

### 5. Retrain the base model (recommended on first run)
This regenerates `heart_model.pkl` to exactly match your installed scikit-learn version, removing any version-mismatch warnings.
```bash
python train_model.py
```

### 6. Run the app
```bash
streamlit run app.py
```
Opens at **http://localhost:8501** — make sure you open this exact URL (not the Network/External URL also printed in the terminal), since browsers only allow microphone access on `localhost` or HTTPS.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this project to a **public or private GitHub repository** (the `.gitignore` already excludes `venv/`, `secrets.toml`, and `patients.db`).
2. Go to **https://share.streamlit.io** → **New app** → select your repo and `app.py` as the entry point.
3. Before clicking Deploy, open **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_real_key_here"
   ```
   This is the cloud equivalent of your local `secrets.toml` — the key is stored encrypted by Streamlit Cloud and never exposed in your repo, logs, or to end users.
4. Click **Deploy**. First build takes 2–5 minutes (installing scikit-learn, SHAP, XGBoost, etc.).

The app works fully **without** a Groq key — only the Voice Control / AI Chat page will show a "not configured" notice, and the other 35 features function normally. Streamlit Cloud also serves over HTTPS automatically, so microphone access works there even though it's not "localhost."

---

## 📁 Project Structure

```
CardioAI_Final/
├── app.py                        ← Main Streamlit application (run this)
├── train_model.py                ← Retrains the base Logistic Regression model
├── requirements.txt              ← All Python dependencies
├── .gitignore                    ← Excludes venv, secrets, cache, local DB
├── .streamlit/
│   ├── config.toml               ← Forces light theme, high contrast
│   └── secrets.toml.example      ← Template — copy to secrets.toml and fill in your key
├── modules/
│   ├── __init__.py
│   ├── groq_client.py            ← Groq API wrapper (voice transcription + chat, key never exposed)
│   └── multi_model.py            ← Trains & caches LR / Random Forest / XGBoost for comparison
├── heart_disease_data.csv        ← Training dataset (303 patients, 13 features)
├── heart_model.pkl               ← Pre-trained Logistic Regression pipeline
├── medicine_dataset.csv          ← Medicine reference database (searchable)
├── symptoms_dataset.csv          ← Disease ↔ symptom lookup (searchable)
├── human_heart.png               ← Heart anatomy image shown in Prediction tab
└── patients.db                   ← SQLite database (auto-created/updated; excluded from git)
```

---

## 🔐 Security Notes

- The Groq API key is read **only** via `st.secrets` (cloud) or `.streamlit/secrets.toml` (local) — never hardcoded.
- The key is **never** printed, logged, displayed in the UI, or included in error messages. All failures fail closed with a generic, user-safe message.
- `secrets.toml` and `patients.db` are excluded from version control by `.gitignore`.
- Uses `whisper-large-v3-turbo` (speech-to-text) and `openai/gpt-oss-120b` (chat) — current, non-deprecated Groq models.

---

## 🎙️ Using Voice Control

1. Open the **🎙️ Voice Control** page from the sidebar.
2. Make sure you're on **`http://localhost:8501`** specifically — browsers block microphone access on plain `http://` network/external addresses, only `localhost` or HTTPS is allowed.
3. Click the microphone recorder, say a page name (e.g. *"Open Digital Twin"*, *"Show analytics"*, *"Go to medicine assistant"*), then stop recording.
4. The app transcribes your speech via Groq Whisper and automatically switches to the matching page.
5. Use the chat box on the right to ask general heart-health questions — answered by the Groq-powered AI assistant.

If no Groq key is configured, this page shows a clear setup notice; every other page in the app still works normally.

---

## 🐛 Known Issues Already Fixed in This Build

| Issue | Fix |
|---|---|
| Dataset label inversion — `target=1` in `heart_disease_data.csv` is actually the *low-risk* group, not high-risk (a well-known quirk of this popular Kaggle reupload), which previously caused clinically sick patients to show low risk and vice versa | All prediction, SHAP, and multi-model comparison logic now flips the raw model output to the correct clinical orientation. Verified with sick-vs-healthy test cases. |
| `search_all_text_columns()` returned zero results on pandas 3.x | Replaced the dtype check with `pd.api.types.is_object_dtype()` / `is_string_dtype()`, which works on both pandas 2.x and 3.x |
| `TypeError` on Patient Comparison / Risk Diff Tool pages | Fixed incorrect dictionary-style access on `itertuples()` namedtuples — now uses attribute access |
| `StreamlitAPIException` on What-If Simulator's "Reset to Baseline" button | Session-state values are now applied before the sliders are instantiated, on the rerun that follows the button click |
| Deprecated `use_container_width` warnings | Replaced all 33 occurrences with `width="stretch"` |

---

## ⚠️ Disclaimer

CardioAI is an **educational and demonstration platform**. It is not a medical device and its predictions are not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified cardiologist or healthcare professional for any cardiac concerns.

---

**Version:** 5.0 · **Total Features:** 37 · **License:** Educational use only
