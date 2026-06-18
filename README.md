# Anaemia Prediction using Machine Learning and Explainable AI

A CSE mini-project that predicts anaemia from CBC blood parameters and **explains
each prediction** using SHAP and LIME. Three models (Random Forest, SVM, XGBoost)
are trained and compared; the best is served through a Streamlit web app.

> ⚠️ **Disclaimer:** This is an educational decision-support prototype, **not a
> diagnostic tool**. It is not clinically validated and must not be used for
> real medical decisions.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app/streamlit_app.py
```

## Project structure

```
data/        raw dataset + processed train/test splits
notebooks/   exploration: EDA, preprocessing, modeling, explainability
src/         reusable logic (config, data loading, preprocessing, eval, explain)
models/      saved .pkl models + scaler
app/         Streamlit application
reports/     exported figures + metrics.json
tests/       light sanity / edge-case tests
```

## Pipeline overview

1. **EDA** — distributions, class balance, correlations, **label-leakage check**.
2. **Preprocessing** — clean, encode gender, stratified split, scale, save scaler.
3. **Training** — RF / SVM / XGBoost on the same split (`python src/train.py`).
4. **Evaluation** — accuracy, precision, recall, F1, AUC-ROC + confusion matrix + ROC.
5. **Explainability** — SHAP (global + local) and one LIME example.
6. **App** — input CBC values → prediction + confidence + SHAP explanation.
