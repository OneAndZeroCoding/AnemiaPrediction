# Viva Preparation — Q&A Drill Sheet

Anaemia Prediction using ML & Explainable AI. Answers are short and speakable —
say them in your own words. The starred ★ questions are the ones examiners almost
always ask for this project.

---

## A. The big ones (★ rehearse until fluent)

**★ Q1. Why is your accuracy ~100%? Isn't that too good?**
Because the dataset's label `Result` is defined from a haemoglobin threshold (WHO:
Hb < 12 g/dL for women, < 13 for men). Any model given Hb just re-learns that rule
— this is **label leakage**, not real medical skill. We don't hide it: we report it
and run a haemoglobin-excluded experiment to prove it. When we remove Hb, accuracy
collapses to ~50% (chance), which confirms the 100% came from Hb alone.

**★ Q2. So what is the actual contribution of your project?**
The **comparison-plus-explainability pipeline** and the methodological honesty: we
compare three models on five metrics, explain every prediction with SHAP and LIME,
identify the most influential feature, and expose the leakage with a controlled
experiment. The value is a trustworthy, interpretable workflow — not the accuracy
number.

**★ Q3. Why explainability in healthcare?**
Clinicians won't act on a black-box output. Explainability shows *which* CBC values
drove a prediction, so a doctor can sanity-check it. It turns "anaemic" into
"anaemic because Hb is low," which is auditable and trustworthy.

**★ Q4. SHAP vs LIME — difference?**
SHAP is game-theory based (Shapley values): consistent, gives both global and local
explanations, and is exact for tree models. LIME fits a small linear model around
one prediction to approximate it locally — fast but only local and less stable. We
use SHAP as primary and LIME as a one-example cross-check; they agree, which
strengthens confidence.

**★ Q5. Why XGBoost as the showcase model?**
Random Forest and XGBoost tied at perfect F1, with SVM just behind. XGBoost is the
literature/PDF-predicted winner and SHAP's `TreeExplainer` handles it exactly and
fast, so we showcase it. We don't claim it's uniquely best — on this leaky data
several models max out.

---

## B. Metrics

**Q6. What is AUC-ROC?**
The ROC curve plots true-positive rate vs false-positive rate across all thresholds.
AUC (area under it) summarises ranking quality: 1.0 = perfect, 0.5 = random. It's
threshold-independent, so it's robust to class imbalance.

**Q7. Precision vs recall — which matters more here?**
Precision = of those predicted anaemic, how many really were (avoids false alarms).
Recall = of truly anaemic patients, how many we caught (avoids missed cases). In
screening, **recall** usually matters more — missing an anaemic patient is worse
than a false alarm. F1 balances both.

**Q8. Why five metrics, not just accuracy?**
Accuracy alone hides errors, especially with imbalance. Precision, recall, F1, and
AUC-ROC give a fuller picture of *how* the model is right or wrong.

---

## C. Method & data

**Q9. Why scale the features, and why fit the scaler on train only?**
SVM measures distances, so a large-range feature would dominate unless all are put
on a comparable scale (StandardScaler → mean 0, std 1). We fit on the training set
only so the test set stays "unseen" — fitting on all data would leak test
information into training.

**Q10. Why a stratified split?**
It keeps the same anaemic/not-anaemic ratio in both train and test, so the test set
is representative and the metrics are trustworthy.

**Q11. Why did you drop duplicate rows?**
1,421 rows had 887 duplicates. Identical patients in both train and test is a second
form of leakage that inflates scores. We dropped them, leaving 534 unique patients.

**Q12. Why random_state = 42 everywhere?**
Reproducibility — the same split and the same model every run, so results are
verifiable by anyone.

**Q13. The original spec wanted RBC and Haematocrit — where are they?**
This dataset doesn't include them. We document the gap and use the available indices
(Hb, MCH, MCHC, MCV, Gender). Adding them is listed in future scope.

---

## D. Models & XAI detail

**Q14. How does Random Forest work? (one line)**
Many decision trees trained on random subsets vote together; averaging reduces
overfitting.

**Q15. How does XGBoost differ from Random Forest?**
Random Forest builds trees independently and averages them (bagging). XGBoost builds
trees sequentially, each correcting the previous one's errors (boosting) — usually
more accurate, slightly more tuning.

**Q16. What does a SHAP waterfall show?**
It starts from the average prediction and adds each feature's contribution (red =
pushes toward anaemic, blue = away) until it reaches this patient's score. Ours
shows real blood values for readability.

**Q17. In the nohb SHAP plot, why is it "muddy"?**
No feature dominates and high/low values land on both sides — the signature of a
model with no real signal. It matches the ~chance accuracy.

---

## E. Limitations & ethics

**Q18. Limitations of your system?**
Small, simple dataset; label derived from Hb (leakage); not clinically validated;
missing some CBC features; binary only (no anaemia subtype). It's a prototype.

**Q19. Is this safe to use in a hospital?**
No. It's an **educational decision-support prototype**, not a diagnostic tool. Every
surface carries that disclaimer. Real diagnosis needs a clinician and validated
tools.

**Q20. How would you make it genuinely useful?**
Use a dataset whose label is an independent clinical diagnosis (not Hb-derived), add
RBC/Haematocrit, extend to anaemia-type classification, and validate clinically.

---

## F. Quick-fire facts to memorise
- Rows: 1,421 → **534** unique · split **427 train / 107 test** · classes 53.7/46.3.
- Full-track F1: RF 1.00, XGBoost 1.00, SVM 0.96.
- nohb AUC: ~0.48–0.63 (≈ chance).
- Stack: Python 3.11, scikit-learn, XGBoost, SHAP, LIME, Streamlit, joblib.
- Run app: `streamlit run app/streamlit_app.py`.
