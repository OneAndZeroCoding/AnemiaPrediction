"""
Phase 7 — Streamlit UI.

Flow:  enter CBC values  ->  Predict  ->  prediction + confidence + SHAP "why".

Run from the project root:
    streamlit run app/streamlit_app.py

This is an educational decision-support PROTOTYPE, not a diagnostic tool.
All model/scaler/explanation logic is reused from src/ — the app writes no ML
logic of its own, so it can never disagree with the notebooks or the report.
"""
import os
import sys

# Make `src` importable no matter where Streamlit is launched from.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from src import config
from src.explain import SHOWCASE_MODEL, shap_explanation

st.set_page_config(page_title="Anaemia Prediction (XAI)", page_icon="🩸", layout="centered")

# --- one-time loads (cached so the app stays snappy) -------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(config.model_path(SHOWCASE_MODEL, "full"))
    scaler = joblib.load(config.SCALER_PATH)
    return model, scaler


model, scaler = load_artifacts()

# Sensible UI ranges/defaults read off the dataset (original units).
INPUTS = {
    "Hemoglobin": dict(min=3.0, max=22.0, default=12.0, step=0.1,
                       help="g/dL. WHO anaemia cutoff ≈ 12 (women) / 13 (men)."),
    "MCH":  dict(min=10.0, max=40.0, default=27.0, step=0.1, help="pg — avg haemoglobin per red cell."),
    "MCHC": dict(min=25.0, max=40.0, default=33.0, step=0.1, help="g/dL — haemoglobin concentration."),
    "MCV":  dict(min=60.0, max=120.0, default=88.0, step=0.1, help="fL — average red-cell size."),
}

def num(field: str, label: str) -> float:
    """Render a number_input from the INPUTS spec (keeps the UI code tidy)."""
    spec = INPUTS[field]
    return st.number_input(
        label, min_value=spec["min"], max_value=spec["max"],
        value=spec["default"], step=spec["step"], help=spec["help"],
    )


UNITS = {"Hemoglobin": "g/dL", "MCH": "pg", "MCHC": "g/dL", "MCV": "fL", "Gender": ""}


def contribution_table(expl_row) -> pd.DataFrame:
    """Turn one patient's SHAP values into a ranked, human-readable table."""
    rows = []
    for name, shap_val, value in zip(
        expl_row.feature_names, expl_row.values, expl_row.data
    ):
        shown = (config.GENDER_MAP[int(value)] if name == "Gender"
                 else f"{value:g} {UNITS[name]}".strip())
        rows.append({
            "Feature": name,
            "This patient": shown,
            "Contribution": round(float(shap_val), 2),
            "Pushes toward": "Anaemic ⬆" if shap_val > 0 else "Not anaemic ⬇",
        })
    df = pd.DataFrame(rows)
    order = df["Contribution"].abs().sort_values(ascending=False).index
    return df.loc[order].reset_index(drop=True)


# --- header ------------------------------------------------------------------
st.title("🩸 Anaemia Prediction with Explainable AI")
st.caption(
    "Model: **XGBoost** (full feature set) · predicts anaemia from CBC blood "
    "values and explains *why* with SHAP."
)
st.warning(
    "⚠️ **Educational prototype — NOT a diagnostic tool.** Do not use for real "
    "medical decisions. Always consult a qualified clinician.",
    icon="⚠️",
)

# --- inputs ------------------------------------------------------------------
st.subheader("Enter CBC values")
col1, col2 = st.columns(2)
with col1:
    gender_label = st.selectbox("Gender", list(config.GENDER_MAP.values()), index=0)
    hb = num("Hemoglobin", "Hemoglobin (g/dL)")
    mch = num("MCH", "MCH (pg)")
with col2:
    mchc = num("MCHC", "MCHC (g/dL)")
    mcv = num("MCV", "MCV (fL)")

# gender label -> encoded 0/1 (reverse of config.GENDER_MAP)
gender_code = {v: k for k, v in config.GENDER_MAP.items()}[gender_label]

# Build the single-patient row in the exact FEATURES order the model expects.
raw_row = pd.DataFrame([{
    "Gender": gender_code, "Hemoglobin": hb, "MCH": mch, "MCHC": mchc, "MCV": mcv,
}])[config.FEATURES]

# --- predict -----------------------------------------------------------------
if st.button("Predict", type="primary"):
    scaled_row = pd.DataFrame(scaler.transform(raw_row), columns=config.FEATURES)
    pred = int(model.predict(scaled_row)[0])
    proba = float(model.predict_proba(scaled_row)[0, 1])  # P(anaemic)

    label = "Anaemic" if pred == 1 else "Not anaemic"
    confidence = proba if pred == 1 else 1 - proba

    if pred == 1:
        st.error(f"### Prediction: **{label}**")
    else:
        st.success(f"### Prediction: **{label}**")
    st.metric("Model confidence", f"{confidence * 100:.1f}%")
    st.progress(min(max(proba, 0.0), 1.0), text=f"P(anaemic) = {proba * 100:.1f}%")

    # --- SHAP "why" ----------------------------------------------------------
    st.subheader("Why this prediction? (SHAP)")
    expl = shap_explanation(model, scaled_row, raw_row)
    table = contribution_table(expl[0])
    top = table.iloc[0]
    share = abs(top["Contribution"]) / table["Contribution"].abs().sum() * 100
    direction = "anaemic" if top["Contribution"] > 0 else "not anaemic"

    # Plain-English headline — what actually drove THIS decision.
    st.markdown(
        f"**Main reason:** **{top['Feature']} = {top['This patient']}** — on its "
        f"own it accounts for **{share:.0f}%** of this decision, pushing it toward "
        f"**{direction}**."
    )

    # Clinical context for the dominant Haemoglobin driver.
    if top["Feature"] == "Hemoglobin":
        cutoff = 12 if gender_code == 0 else 13
        rel = "below" if hb < cutoff else "at or above"
        st.caption(
            f"Context: the WHO anaemia cutoff for {gender_label.lower()}s is "
            f"≈ {cutoff} g/dL. This patient's {hb:g} g/dL is **{rel}** it — which "
            f"is exactly why the model leans this way."
        )

    # Full ranked breakdown (every feature, signed contribution).
    st.markdown("**Full breakdown** — how each value pushed the prediction:")
    st.dataframe(table, hide_index=True, use_container_width=True)

    # The same thing as the standard SHAP waterfall picture.
    with st.expander("Show the SHAP waterfall chart"):
        st.caption("Bars start from the average prediction and add each feature's "
                   "push until reaching this patient's score.")
        shap.plots.waterfall(expl[0], show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 4)
        fig.tight_layout()
        st.pyplot(fig, bbox_inches="tight")
        plt.close("all")

    st.info(
        "Why is Haemoglobin almost always the top reason? Because in this dataset "
        "the label was *defined* from a Haemoglobin threshold (label leakage). The "
        "model is essentially re-reading Hb. We prove this in the report: with Hb "
        "removed, accuracy collapses to ~50% (chance).",
        icon="ℹ️",
    )

st.divider()
st.caption("Anaemia Prediction using Machine Learning & Explainable AI · "
           "CSE Mini Project · educational prototype only.")
