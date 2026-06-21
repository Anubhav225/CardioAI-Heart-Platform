"""
groq_client.py
──────────────
Thin wrapper around the Groq API used for:
  1. Speech-to-text transcription (Whisper Large v3 Turbo)
  2. Voice command intent parsing (GPT-OSS 120B)
  3. AI chatbot responses about a patient's report

SECURITY NOTE:
  The API key is read ONLY from Streamlit secrets or environment variables.
  It is never logged, printed, displayed in the UI, or returned in any
  function output. If the key is missing, every function fails closed
  (returns None / a friendly error) rather than raising a traceback that
  could leak configuration details.

Current (non-deprecated) Groq models used, per console.groq.com/docs/models:
  - "whisper-large-v3-turbo"  → speech-to-text
  - "openai/gpt-oss-120b"     → chat / intent parsing
    (llama-3.3-70b-versatile was deprecated 2026-06-17; gpt-oss-120b is
     the recommended replacement.)
"""

import os
import streamlit as st

GROQ_CHAT_MODEL  = "openai/gpt-oss-120b"
GROQ_STT_MODEL   = "whisper-large-v3-turbo"


def _get_api_key():
    """
    Resolve the Groq API key from Streamlit secrets first, then env vars.
    Never returns the key to the caller for display — only used internally.
    """
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def is_configured() -> bool:
    """True if a Groq API key is present. Safe to call anywhere in the UI."""
    key = _get_api_key()
    return bool(key and len(key.strip()) > 10)


def _get_client():
    """Builds a Groq client instance. Returns None if not configured."""
    key = _get_api_key()
    if not key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=key)
    except Exception:
        return None


def transcribe_audio(audio_bytes: bytes, filename: str = "command.wav"):
    """
    Sends recorded audio to Groq Whisper for transcription.
    Returns (text, error). On any failure, text is None and error is a
    short, user-safe message — never the raw exception (which could
    theoretically echo back request details).
    """
    client = _get_client()
    if client is None:
        return None, "Voice features need a Groq API key. Ask the app owner to add GROQ_API_KEY in Secrets."
    try:
        result = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=GROQ_STT_MODEL,
            response_format="text",
        )
        text = result if isinstance(result, str) else getattr(result, "text", "")
        return text.strip(), None
    except Exception:
        return None, "Could not transcribe audio. Please try again or check your connection."


def chat_completion(messages, max_tokens: int = 500, temperature: float = 0.4):
    """
    Generic Groq chat completion call.
    `messages` follows the standard OpenAI-style list of {role, content} dicts.
    Returns (text, error).
    """
    client = _get_client()
    if client is None:
        return None, "AI chat needs a Groq API key. Ask the app owner to add GROQ_API_KEY in Secrets."
    try:
        resp = client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip(), None
    except Exception:
        return None, "AI assistant is temporarily unavailable. Please try again shortly."


# ── Voice command vocabulary (used both for keyword fallback AND to ground the LLM) ──
VOICE_COMMANDS = {
    "prediction":        "Open the Prediction page",
    "clinical engine":   "Open the AI Clinical Decision Engine",
    "digital twin":      "Open the Digital Twin future risk simulator",
    "what if":           "Open the What-If Risk Simulator",
    "model insights":    "Open Model Insights",
    "patient history":   "Open Patient History",
    "compare patients":  "Open Patient Comparison",
    "analytics":         "Open the Analytics dashboard",
    "segmentation":      "Open Patient Segmentation",
    "iot monitor":       "Open the IoT vital signs monitor",
    "anomaly":           "Open Anomaly Detection",
    "medicine":          "Open Medicine Assistant",
    "symptom":           "Open Symptom Checker",
    "reports":           "Open Reports and Batch prediction",
    "advanced":          "Open Advanced settings",
    "help":              "List available voice commands",
}


def parse_voice_command(transcript: str):
    """
    Maps a spoken transcript to one of the app's sidebar pages.
    Tries a fast keyword match first (no API call, works even without Groq).
    Falls back to the Groq LLM for fuzzier phrasing if configured.
    Returns (matched_key, page_label) or (None, None).
    """
    if not transcript:
        return None, None
    t = transcript.lower()

    keyword_to_page = {
        "predict":          "🩺  Prediction",
        "prediction":       "🩺  Prediction",
        "clinical":         "🧠  AI Clinical Engine",
        "decision engine":  "🧠  AI Clinical Engine",
        "digital twin":     "🔮  Digital Twin",
        "future risk":      "🔮  Digital Twin",
        "what if":          "🧪  What-If Simulator",
        "simulator":        "🧪  What-If Simulator",
        "model insight":    "📊  Model Insights",
        "insight":          "📊  Model Insights",
        "history":          "📁  Patient History",
        "compare":          "👥  Patient Comparison",
        "comparison":       "👥  Patient Comparison",
        "diff":             "🆚  Risk Diff Tool",
        "changed":          "🆚  Risk Diff Tool",
        "analytics":        "📈  Analytics",
        "dashboard":        "📈  Analytics",
        "segment":          "🔬  Patient Segmentation",
        "cluster":          "🔬  Patient Segmentation",
        "iot":              "📡  IoT Monitor",
        "monitor":          "📡  IoT Monitor",
        "vitals":           "📡  IoT Monitor",
        "anomaly":          "🚨  Anomaly Detection",
        "medicine":         "💊  Medicine Assistant",
        "drug":             "💊  Medicine Assistant",
        "symptom":          "🧬  Symptom Checker",
        "report":           "📄  Reports & Batch",
        "batch":            "📄  Reports & Batch",
        "multi model":      "🏆  Multi-Model Compare",
        "compare model":    "🏆  Multi-Model Compare",
        "advanced":         "⚙️  Advanced",
        "setting":          "⚙️  Advanced",
    }
    for kw, page in keyword_to_page.items():
        if kw in t:
            return kw, page
    return None, None
