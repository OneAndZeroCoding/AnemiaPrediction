# Anaemia Prediction using Machine Learning and Explainable AI

**CSE Mini Project — Matrusri Engineering College**

> **Disclaimer:** This is an **educational decision-support prototype**, not a
> diagnostic tool. It must not be used for real medical decisions. All clinical
> diagnosis must be made by a qualified professional.

---

## Abstract

Anaemia affects over 1.6 billion people worldwide, and in resource-constrained
settings its diagnosis is often slow and error-prone. This project builds a fast,
accurate, and — crucially — **interpretable** predictor of anaemia from routine
Complete Blood Count (CBC) parameters. We train and compare three machine-learning
models (Random Forest, Support Vector Machine, and XGBoost), evaluate them on five
standard metrics, and integrate two explainability techniques (**SHAP** and
**LIME**) so that every prediction can be justified feature-by-feature. A central
and honestly-reported finding is that the public dataset's label is defined from a
haemoglobin threshold, which produces near-perfect accuracy through **label
leakage**; we quantify this with a deliberate haemoglobin-excluded experiment. The
project's true contribution is therefore the **comparison-plus-explainability
pipeline** and the methodological rigour around it, not the headline accuracy
number. A Streamlit web application delivers input → prediction → SHAP explanation
with disclaimers.

---

## 1. Introduction

Anaemia is a condition in which the blood has a reduced capacity to carry oxygen,
usually due to low haemoglobin (Hb). It is widespread, especially among women and
children in low-income regions. A CBC test is cheap and common, but interpreting it
and reaching a diagnosis still requires expertise that is scarce in many clinics.

Machine learning can screen CBC results quickly, but clinicians will not trust an
opaque "black box" prediction. **Explainable AI (XAI)** addresses this by showing
*which* blood values drove each prediction. This project combines accurate
prediction with per-patient explanations to produce a trustworthy screening aid.

**Objectives**
1. Train and compare Random Forest, SVM, and XGBoost on CBC data and select the best.
2. Integrate SHAP and LIME for per-prediction explanations.
3. Identify the most influential CBC features.
4. Build an accessible web UI for early screening in low-resource settings.

---

## 2. Existing System and Limitations

Conventional anaemia screening relies on manual interpretation of CBC reports by a
clinician or lab technician. Limitations:

- **Slow and expertise-dependent** — needs trained personnel, often unavailable.
- **Inconsistent** — interpretation varies between practitioners.
- **No prioritisation** — every report is read manually with equal effort.

Prior ML approaches frequently report very high accuracy but **omit
explainability** and, more importantly, **do not disclose label leakage** when the
target is derived from haemoglobin. A high accuracy number without that disclosure
is misleading. Our system explicitly addresses both gaps.

---

## 3. Proposed System

A pipeline that:
1. Loads and cleans CBC data, then produces a fixed stratified train/test split.
2. Trains three models on identical data and compares them on five metrics.
3. Explains predictions globally and locally with SHAP, cross-checked with LIME.
4. Serves predictions and explanations through a Streamlit web app with disclaimers.
5. Runs a **haemoglobin-excluded robustness experiment** to expose label leakage.

---

## 4. Dataset

- **Source:** Biswaranjan Rao "Anemia Dataset" (public, Kaggle).
- **Raw size:** 1,421 rows × 6 columns; **no missing values.**
- **De-duplication:** 887 duplicate rows were removed (they would otherwise leak
  identical patients across the train/test boundary), leaving **534 unique
  patients.**
- **Class balance (clean):** 53.7% not-anaemic, 46.3% anaemic — well balanced.
- **Features used:** `Gender` (0 = Female, 1 = Male, already encoded),
  `Hemoglobin`, `MCH`, `MCHC`, `MCV`.
- **Target:** `Result` (0 = not anaemic, 1 = anaemic).

**Note on missing PDF features:** the original specification listed RBC count and
Haematocrit. This dataset does not include them; we document the gap and proceed
with the available red-cell indices (MCH, MCHC, MCV) plus Hb and Gender.

---

## 5. Methodology

### 5.1 The label-leakage finding (central to this project)
During EDA we found that `Result` is reproducible almost perfectly from a
**gender-specific haemoglobin threshold** (consistent with WHO cutoffs: Hb < 12
g/dL for women, < 13 g/dL for men). Any model given Hb therefore re-learns the
labelling rule rather than discovering medical signal. We **report this openly**
and treat it as the project's key insight rather than hiding behind the accuracy.

### 5.2 Two-track design
- **`full` track** — all features (includes Hb): the leaky, near-perfect models.
- **`nohb` track** — Hb removed: an honest test of whether the *other* CBC indices
  carry signal. *(Marks-booster robustness experiment.)*

### 5.3 Preprocessing
- Stratified 80/20 split (`random_state = 42`) → **427 train / 107 test**.
- **StandardScaler fit on the training set only** (then applied to both), so no
  test information leaks into scaling. SVM in particular is scale-sensitive.
- Splits saved in original units; the scaler saved separately (`scaler.pkl`).

### 5.4 Models
| Model | Why chosen |
|------|-----------|
| Random Forest | Robust ensemble of decision trees; little tuning needed. |
| SVM (RBF) | Strong margin-based classifier; `probability=True` for AUC-ROC. |
| XGBoost | Gradient-boosted trees; literature-predicted best performer. |

All three share `random_state = 42` for reproducibility.

### 5.5 Evaluation metrics
Accuracy, **precision** (avoid false alarms), **recall** (avoid missed cases),
**F1** (balance of the two), and **AUC-ROC** (threshold-independent ranking
quality). Plus confusion matrices and ROC curves.

### 5.6 Explainability
- **SHAP** (primary): `TreeExplainer` for global (beeswarm + importance bar) and
  local (waterfall) explanations. Computed on scaled inputs but displayed in real
  CBC units for readability.
- **LIME** (one minimal example): a local linear approximation as an independent
  cross-check on the SHAP story.

---

## 6. Implementation

Reusable logic lives in `src/` (imported by notebooks and the app, so logic is
written once):

```
src/config.py      paths, feature lists, constants (single source of truth)
src/data_loader.py load + de-duplicate
src/preprocess.py  split, scale, save scaler  (the Phase 3→4 contract)
src/train.py       train RF/SVM/XGBoost for both tracks
src/evaluate.py    five metrics + confusion + ROC → metrics.json
src/explain.py     SHAP + LIME helpers
app/streamlit_app.py   input → prediction → SHAP + disclaimer
notebooks/01–04    EDA, preprocessing, modelling, explainability
tests/test_pipeline.py light sanity tests (10, all pass)
```

- **Stack:** Python 3.11, scikit-learn, XGBoost, SHAP, LIME, Streamlit, joblib.
- **Persistence:** models and scaler saved as `.pkl` (joblib).
- **Run the app:** `streamlit run app/streamlit_app.py`.

---

## 7. Results

### 7.1 Model comparison — `full` track (with Haemoglobin)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|------|:-------:|:--------:|:-----:|:----:|:------:|
| Random Forest | **1.000** | 1.000 | 1.000 | **1.000** | 1.000 |
| SVM | 0.963 | 0.925 | 1.000 | 0.961 | 0.999 |
| XGBoost | **1.000** | 1.000 | 1.000 | **1.000** | 1.000 |

Random Forest and XGBoost are tied at a perfect score; SVM is a hair behind. These
numbers are **expected** and reflect label leakage, not clinical skill. We showcase
**XGBoost** as the representative best model (literature-predicted winner, and SHAP
`TreeExplainer` handles it exactly).

*Figures:* `fig_metrics_comparison.png`, `fig_confusion_full.png`, `fig_roc_full.png`.

### 7.2 Robustness — `nohb` track (Haemoglobin removed)

| Model | Accuracy | F1 | AUC-ROC |
|------|:-------:|:----:|:------:|
| Random Forest | 0.542 | 0.515 | 0.505 |
| SVM | 0.626 | 0.608 | 0.628 |
| XGBoost | 0.486 | 0.455 | 0.480 |

With Hb removed, performance **collapses toward chance** (AUC ≈ 0.5). This proves
the headline accuracy came almost entirely from Haemoglobin — i.e. from the way the
label was defined — and that the remaining indices carry little independent signal
*in this dataset*.

*Figures:* `fig_confusion_nohb.png`, `fig_roc_nohb.png`.

### 7.3 Explainability

- **Global SHAP (full):** Haemoglobin dominates by a wide margin, Gender second,
  MCH/MCHC/MCV near zero — the leakage finding made visible.
  *(`fig_shap_beeswarm_full.png`, `fig_shap_bar_full.png`)*
- **Local SHAP:** for an anaemic patient, a low Hb reading delivers the dominant
  push toward *anaemic*; for a healthy patient it pushes the other way.
  *(`fig_shap_waterfall_full_anaemic.png`, `fig_shap_waterfall_full_healthy.png`)*
- **Global SHAP (nohb):** no dominant feature and muddy colour gradients — the
  visual signature of a model with no real signal. *(`fig_shap_beeswarm_nohb.png`)*
- **LIME:** independently lands on the same low-Hb rule (e.g. `Hemoglobin ≤ 11.55`)
  with Gender second, agreeing with SHAP. *(`fig_lime_full_anaemic.png`)*

---

## 8. Conclusion

We built a complete, reproducible ML + Explainable-AI pipeline for anaemia
screening: three compared models, five metrics, SHAP (global + local), a LIME
cross-check, and a Streamlit app. The headline accuracy is near-perfect, but we
**diagnose and disclose** that this is label leakage from a haemoglobin-defined
target, and we **quantify** it with a haemoglobin-excluded experiment that collapses
to chance. The project's real value is methodological honesty plus interpretability:
for any patient, we can show exactly why the model decided as it did.

---

## 9. Future Scope

- Use a dataset with an **independent clinical label** (not derived from Hb) so the
  models can learn genuine multi-feature signal.
- Add the missing CBC features (RBC count, Haematocrit) for richer inputs.
- Extend from binary to **anaemia-type classification** (microcytic / normocytic /
  macrocytic) using MCV-based logic.
- Clinical validation on real, diverse patient data before any practical use.

---

## 10. References

1. World Health Organization — Haemoglobin concentrations for the diagnosis of
   anaemia (WHO/NMH/NHD/MNM).
2. Biswaranjan Rao, "Anemia Dataset," Kaggle.
3. Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (SHAP),
   NeurIPS 2017.
4. Ribeiro, Singh & Guestrin, "Why Should I Trust You?: Explaining the Predictions
   of Any Classifier" (LIME), KDD 2016.
5. Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD 2016.
6. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR 2011.

---

*Educational prototype only — not a diagnostic tool.*
