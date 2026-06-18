# 🎓 Session Handoff — Teaching the Project from Scratch

> **If you are an AI assistant reading this:** the project is fully built. Your job
> in the next session is **NOT to write more code** — it is to **teach Clyton how
> the whole project works, from scratch, practically.** Read this file, then
> `CLAUDE.md` (the source of truth), then teach. Confirm the plan before diving in.

---

## 1. Who you're teaching

- **Clyton** — a **machine-learning beginner**. Define every ML term the first time
  it appears. No hand-waving.
- He learns by understanding **what → why → how**, in that order (this is mandated
  in `CLAUDE.md §1`). Give working examples, recommend ONE best option, and
  **challenge weak reasoning** rather than agreeing.
- He asks sharp questions and wants the *real* answer, not a comfortable one. He
  has already pushed back on: vague explanations, the 100% accuracy, "why XGBoost",
  and per-patient probability differences. Match that depth.

## 2. The teaching goal

By the end he should be able to explain, in his own words and to an examiner:
the dataset, the leakage finding, the pipeline, the three models, the five metrics,
SHAP, LIME, and the Streamlit app — **and why each choice was made.**

He does **not** need to write the code from memory. He needs to *understand and
defend* it. The viva drill sheet (`reports/VIVA_PREP.md`) is the target outcome.

## 3. Project state (so you don't re-explain stale things)

**All phases 0–11 are built and committed.** Only Phase 4 model *training* is a
teammate's job, and the trained `.pkl` artifacts are already committed, so
everything runs end-to-end now.

Key files to teach FROM (all logic lives in `src/`, imported everywhere else):
```
src/config.py      paths, FEATURES, constants (single source of truth)
src/data_loader.py load + de-duplicate (1421 raw -> 534 unique)
src/preprocess.py  stratified split, scaler (fit on train only)
src/train.py       RF / SVM / XGBoost, two tracks (full + nohb)
src/evaluate.py    five metrics + confusion + ROC -> reports/metrics.json
src/explain.py     SHAP + LIME helpers
app/streamlit_app.py   the UI (prediction + 3-model compare + SHAP + LIME)
notebooks/01–04    EDA, preprocessing, modelling, explainability (run, with outputs)
tests/test_pipeline.py  10 passing sanity tests
reports/REPORT.md, PPT_OUTLINE.md, VIVA_PREP.md
```

## 4. The concepts he already understands (build on these, don't repeat)

1. **Label leakage** — `Result` is defined from a gender-specific Haemoglobin
   threshold (WHO: F<12, M<13 g/dL). Models with Hb hit ~100% by re-learning the
   rule. A hand-written gender-aware `if/else` reproduces **98.9%** of the labels —
   so the ML "skill" is mostly the definition. **This is the project's core honest
   finding, not a bug.**
2. **The nohb experiment** — remove Hb and accuracy collapses to ~chance (AUC ≈
   0.5), proving the other indices carry little signal *for this Hb-defined label*.
3. **Medical truth** — Hb *is* the definition of anaemia (detection); the other
   indices (MCV/MCH/MCHC) classify the *type* (microcytic/normocytic/macrocytic).
   The genuinely non-trivial ML problem is anaemia-*type* classification (future
   scope).
4. **Why XGBoost is the selected model** — RF and XGBoost tied at F1=1.00/AUC=1.00;
   among the tie, XGBoost is the literature pick and SHAP TreeExplainer handles it
   exactly. Selection was on **aggregate test metrics, never one patient's number.**
5. **RF vs XGBoost probabilities** — RF probability = fraction of its 200 trees
   voting; near the Hb boundary the trees split, so RF gives wishy-washy ~40–50%.
   XGBoost squashes through a sigmoid and commits hard. They still agree on the
   label; they differ in *calibration/confidence*.

## 5. Suggested teaching order (a curriculum)

Walk the pipeline in the order data flows — each step answers "why does this exist?"

1. **The problem & the dataset** — `CLAUDE.md §2`, `src/config.py`, `notebooks/01`.
2. **EDA & the leakage discovery** — `notebooks/01`, the Hb-scatter figure.
3. **Preprocessing** — de-dup, stratified split, scaling & *why fit on train only*
   (`src/preprocess.py`).
4. **The three models** — what RF / SVM / XGBoost each are, in one line, and why
   these three (`src/train.py`).
5. **Evaluation** — define each of the 5 metrics in plain language, confusion
   matrix, ROC/AUC (`src/evaluate.py`, `reports/metrics.json`).
6. **The two-track story** — full vs nohb, what it proves (`metrics.json`).
7. **SHAP** — global (beeswarm/bar) vs local (waterfall); what a Shapley value *is*
   intuitively (`src/explain.py`, `notebooks/04`).
8. **LIME** — local linear approximation; SHAP vs LIME differences.
9. **The app** — how inputs flow to prediction + explanation (`app/streamlit_app.py`).
10. **Viva rehearsal** — run through `reports/VIVA_PREP.md` Q&A.

Teach interactively: explain a step, check understanding, move on. Don't dump all
ten at once.

## 6. How to run things (env)

- Python **3.11** venv already exists: use `./venv/bin/python` (3.12+ breaks
  xgboost/shap/lime).
- App: `streamlit run app/streamlit_app.py` (demo: anaemic Hb≈7.5, healthy Hb≈16).
- Tests: `./venv/bin/python -m pytest tests/ -q`.
- Notebooks already executed with outputs saved — open and read, no need to re-run.

## 7. Optional enhancements offered but NOT yet done (only if he asks)

- Add the `if/else`-vs-model comparison table to `REPORT.md` + a PPT slide.
- Let the user pick *which* model drives the SHAP/LIME explanation (app dropdown).
- Flag "borderline case" in the app when the three models' probabilities diverge.

## 8. Git note

There are local commits that may be **unpushed** (this environment has no GitHub
credentials). If asked, remind Clyton to `git push` himself. Do not invent
credentials.

---

*Educational prototype only — not a diagnostic tool. Keep that framing in every
explanation.*
