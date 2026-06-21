"""
multi_model.py
───────────────
Trains and caches three classifiers (Logistic Regression, Random Forest,
XGBoost) on the heart disease dataset so the app can show how different
algorithms agree or disagree on the same patient — a robustness check
rather than relying on a single black-box model.

Training is cheap (303 rows) so it's done once per session and cached
with st.cache_resource.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


@st.cache_resource(show_spinner=False)
def train_comparison_models(heart_df: pd.DataFrame, target_col: str, feature_columns: list):
    """
    Trains LR, Random Forest, and (if available) XGBoost on the dataset.
    Returns a dict: {model_name: {"model": fitted_pipeline, "accuracy": float, "auc": float}}
    """
    X = heart_df[feature_columns]
    y = heart_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # ── Logistic Regression ──
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred  = lr_pipe.predict(X_test)
    lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
    results["Logistic Regression"] = {
        "model": lr_pipe,
        "accuracy": accuracy_score(y_test, lr_pred) * 100,
        "auc": roc_auc_score(y_test, lr_proba) * 100,
        "color": "#0077b6",
    }

    # ── Random Forest ──
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
    ])
    rf_pipe.fit(X_train, y_train)
    rf_pred  = rf_pipe.predict(X_test)
    rf_proba = rf_pipe.predict_proba(X_test)[:, 1]
    results["Random Forest"] = {
        "model": rf_pipe,
        "accuracy": accuracy_score(y_test, rf_pred) * 100,
        "auc": roc_auc_score(y_test, rf_proba) * 100,
        "color": "#2a9d5c",
    }

    # ── XGBoost (optional dependency) ──
    if XGBOOST_AVAILABLE:
        xgb_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.1,
                eval_metric="logloss", random_state=42,
            )),
        ])
        xgb_pipe.fit(X_train, y_train)
        xgb_pred  = xgb_pipe.predict(X_test)
        xgb_proba = xgb_pipe.predict_proba(X_test)[:, 1]
        results["XGBoost"] = {
            "model": xgb_pipe,
            "accuracy": accuracy_score(y_test, xgb_pred) * 100,
            "auc": roc_auc_score(y_test, xgb_proba) * 100,
            "color": "#e63946",
        }

    return results


def predict_with_all_models(models_dict, input_df):
    """
    Runs the same patient through every trained model.
    Returns list of dicts: [{name, prediction, risk, color}, ...]

    NOTE: This dataset's 'target' column is inverted from normal clinical
    convention (target=1 is the clinically HEALTHIER group — verified by
    group-mean analysis of cholesterol/BP/oldpeak/max-heart-rate). Every
    model trained on it therefore predicts class 1 for low-risk patients.
    The raw prediction/probability is flipped here so this page's results
    match the main Prediction page's clinically-correct orientation.
    """
    out = []
    for name, info in models_dict.items():
        m = info["model"]
        raw_pred  = int(m.predict(input_df)[0])
        raw_proba = float(m.predict_proba(input_df)[0][1])
        pred = 1 - raw_pred             # flip
        risk = (1.0 - raw_proba) * 100  # flip
        out.append({
            "name": name,
            "prediction": pred,
            "risk": risk,
            "color": info["color"],
            "accuracy": info["accuracy"],
            "auc": info["auc"],
        })
    return out


def agreement_summary(predictions: list):
    """
    Given the output of predict_with_all_models, summarises whether
    the models agree, and computes the spread in risk estimates.
    """
    if not predictions:
        return None
    preds = [p["prediction"] for p in predictions]
    risks = [p["risk"] for p in predictions]
    all_agree = len(set(preds)) == 1
    return {
        "all_agree": all_agree,
        "majority": 1 if sum(preds) >= len(preds) / 2 else 0,
        "min_risk": min(risks),
        "max_risk": max(risks),
        "spread": max(risks) - min(risks),
        "avg_risk": sum(risks) / len(risks),
    }
