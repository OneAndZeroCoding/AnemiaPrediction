# 🔁 Training Handoff — read this fully before doing anything

> **If you are an AI assistant** reading this on behalf of the teammate who owns
> model training: this document is self-contained. Read it top to bottom, then
> guide the user through the steps exactly. Do **not** redesign the pipeline.

---

## 1. What this project is (30-second context)

**Anaemia Prediction using Machine Learning & Explainable AI** — a college
mini-project. It predicts anaemia from CBC blood values (Haemoglobin, MCH, MCHC,
MCV, Gender) using three models (Random Forest, SVM, XGBoost), compares them,
and explains predictions with SHAP/LIME. It is an **educational prototype, not a
diagnostic tool.**

Everything except **model training** is already built: data loading, EDA,
preprocessing, evaluation, and (soon) explainability + the Streamlit app.

## 2. Your one job

**Run the training script, verify the output, and push the trained models to GitHub.**
That's it. The script is already written and tested. You do **not** write any code.

---

## 3. Setup (one time)

> ⚠️ **Use Python 3.11.** Do NOT use Python 3.12/3.13/3.14 — `xgboost`, `shap`,
> and `lime` fail to install on very new Python versions.

```bash
# 1. Get the latest code + committed data/models
git clone <repo-url>        # or: git pull
cd AnemiaPrediction

# 2. Create and activate a virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 4. Train the models — the ONE command

```bash
python -m src.train
```

This trains **two tracks** (this is intentional — see §6):
- **full**  — all features (includes Haemoglobin)
- **nohb**  — Haemoglobin excluded (a deliberate robustness experiment)

It loads the **already-prepared, already-scaled** train/test split from
`data/processed/` and saves six model files to `models/`.

## 5. Verify you got it right (self-check)

After training, you must see **six** files in `models/` (plus `scaler.pkl`):

```
models/random_forest.pkl        models/random_forest_nohb.pkl
models/svm.pkl                  models/svm_nohb.pkl
models/xgboost.pkl              models/xgboost_nohb.pkl
```

Then run the evaluation to confirm your numbers match:

```bash
python -m src.evaluate
```

**Expected output (must look essentially like this):**

```
=== full track ===
model              acc    prec     rec      f1     auc
random_forest    1.000   1.000   1.000   1.000   1.000
svm              0.963   0.925   1.000   0.961   0.999
xgboost          1.000   1.000   1.000   1.000   1.000

=== nohb track ===
random_forest    0.542 ...   svm  0.626 ...   xgboost  0.486 ...   (all near 0.5)
```

If your **full** track is ~1.0 and your **nohb** track is ~0.5, ✅ you're correct.

## 6. ⚠️ Things that look like bugs but are NOT — do not "fix" them

1. **Random Forest & XGBoost score 100%.** This is **expected and known.** The
   dataset's label was generated from a haemoglobin threshold, so the models
   re-learn that rule perfectly. This is *label leakage*, documented as the
   project's main finding. **Do NOT try to lower it or "improve" the data.**
2. **The `nohb` models score ~50% (near random).** Also expected — it proves the
   label carries no signal beyond Haemoglobin. That collapse is a *result we
   want*, not a failure.
3. **A `FutureWarning` about `SVC(probability=True)`.** Harmless. Ignore it.

## 7. Hard rules — do NOT do these

- ❌ Do **not** edit `src/config.py`, `src/preprocess.py`, or `src/train.py`.
- ❌ Do **not** re-split, re-shuffle, or re-scale the data. The split is fixed
  (`random_state=42`) so everyone evaluates identical data.
- ❌ Do **not** change `random_state`, model parameters, or the feature list.
- ❌ Do **not** delete or regenerate files in `data/processed/`.

If you think something genuinely needs changing, **stop and message the project
owner (Clyton)** instead of changing it.

## 8. Commit & push your models

```bash
git add models/
git commit -m "Phase 4: train RF/SVM/XGBoost (full + nohb tracks)"
git push
```

Tell the team you've pushed. Evaluation (Phase 5) and explainability (Phase 6)
will then run on your exact models.

---

## 9. Where to read more
- `CLAUDE.md` — full project plan and the leakage explanation.
- `notebooks/03_modeling.ipynb` — the comparison + metrics, already run.
- `reports/metrics.json` — the saved metrics for both tracks.
