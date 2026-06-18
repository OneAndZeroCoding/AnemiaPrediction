# Presentation Outline — Anaemia Prediction using ML & Explainable AI

~13 slides. Each slide: **title**, the **bullets to show**, and *[speaker notes]*.
Figures referenced live in `reports/figures/`. Keep the disclaimer visible on the
title and demo slides.

---

### Slide 1 — Title
- Anaemia Prediction using Machine Learning and Explainable AI
- CSE Mini Project · Matrusri Engineering College
- Your name(s), guide, date
- *Small disclaimer: educational prototype, not a diagnostic tool.*

### Slide 2 — Problem & Motivation
- Anaemia affects 1.6B+ people; diagnosis is slow in low-resource settings.
- CBC tests are cheap and common, but interpretation needs scarce expertise.
- *[Goal: fast, accurate, and trustworthy screening from CBC values.]*

### Slide 3 — Why Explainable AI?
- Clinicians won't trust a black box.
- We must show *which* blood values drove each prediction.
- *[XAI = the difference between "anaemic" and "anaemic, because Hb is low."]*

### Slide 4 — Objectives
- Compare 3 models (Random Forest, SVM, XGBoost).
- Integrate SHAP + LIME explanations.
- Identify most influential features.
- Build an accessible web UI.

### Slide 5 — Dataset
- Biswaranjan Rao Anemia Dataset: 1,421 rows → **534 unique** after de-duplication.
- Features: Gender, Hemoglobin, MCH, MCHC, MCV → Target: Result (0/1).
- Class balance: 53.7% / 46.3%. No missing values.
- *[Note the missing RBC/Haematocrit vs the original spec — documented gap.]*

### Slide 6 — ⚠️ The Label-Leakage Finding (key slide)
- The label `Result` is defined from a **haemoglobin threshold** (WHO cutoffs).
- So any model with Hb just re-learns the rule → near-perfect "accuracy."
- We **disclose** this and test it instead of hiding it.
- *Figure:* `fig_leakage_hb_scatter.png`
- *[This honesty is the heart of the project — examiners reward it.]*

### Slide 7 — Methodology / Pipeline
- Clean → stratified 80/20 split → scale (fit on train only) → train 3 models →
  evaluate (5 metrics) → explain (SHAP + LIME) → Streamlit app.
- Two tracks: **full** (with Hb) and **nohb** (Hb removed).
- *[Reusable `src/` logic; notebooks + app import from it.]*

### Slide 8 — Model Comparison (full track)
- RF 1.00 · XGBoost 1.00 · SVM 0.96 (F1).
- *Figures:* `fig_metrics_comparison.png`, `fig_roc_full.png`.
- *[High — but expected, because of leakage. Lead into next slide.]*

### Slide 9 — Robustness: Remove Haemoglobin (nohb track)
- Performance collapses to ~chance: AUC ≈ 0.48–0.63.
- Proves the accuracy came from Hb, not the other indices.
- *Figures:* `fig_roc_nohb.png`, `fig_confusion_nohb.png`.
- *[This is the experiment that makes the 100% honest.]*

### Slide 10 — Explainability with SHAP
- Global: Haemoglobin dominates, Gender second, rest near zero.
- Local: per-patient waterfall in real blood units.
- *Figures:* `fig_shap_beeswarm_full.png`, `fig_shap_waterfall_full_anaemic.png`.

### Slide 11 — LIME Cross-Check
- LIME independently finds the same low-Hb rule (e.g. `Hemoglobin ≤ 11.55`).
- Two methods agreeing → confidence in the explanation.
- *Figure:* `fig_lime_full_anaemic.png`.
- *[SHAP = consistent, global+local; LIME = local linear approximation.]*

### Slide 12 — The Web App (Streamlit)
- Live demo: enter CBC values → prediction + confidence → SHAP "why" + disclaimer.
- `streamlit run app/streamlit_app.py`.
- *[Show one anaemic and one healthy input if time allows.]*

### Slide 13 — Conclusion & Future Scope
- Built a full, reproducible ML + XAI pipeline; diagnosed & disclosed leakage.
- Contribution = comparison + explainability + methodological honesty.
- Future: independent (non-Hb) label, add RBC/Haematocrit, anaemia-type
  classification, clinical validation.
- *Closing disclaimer: educational prototype only.*

---

**Demo checklist:** venv active · `streamlit run app/streamlit_app.py` · have one
anaemic sample (Hb≈7.5) and one healthy sample (Hb≈16) ready.
