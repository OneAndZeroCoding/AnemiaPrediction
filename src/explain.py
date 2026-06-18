"""
Phase 6 — Explainable AI (SHAP + LIME).

WHY this phase exists (the project's core objective):
  A clinician will not trust a black box. Explainability answers
  "WHY did the model say anaemic?" by attributing the decision to each CBC
  feature. We use two complementary tools:

    * SHAP (primary) — game-theory based feature attributions. Consistent,
      gives BOTH a global picture (which features matter overall) and a local
      one (why THIS patient). We expect Hemoglobin to tower over everything,
      which visually confirms the label-leakage finding.
    * LIME (required, minimal) — fits a tiny linear model around one prediction
      to approximate it locally. We show ONE example, as the spec asks.

Correctness vs readability:
  The models were trained on SCALED features, so SHAP/LIME must feed the model
  scaled inputs. But scaled values (z-scores) are unreadable on a chart. So we
  compute attributions on scaled data and DISPLAY the original blood values.

Run:  python -m src.explain      # regenerates all explainability figures
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from . import config
from .preprocess import load_processed

# Among the full-track models RF and XGBoost both score f1 = 1.0 (a tie).
# We showcase XGBoost: it is the literature/PDF-predicted winner and
# shap.TreeExplainer handles it natively and exactly.
SHOWCASE_MODEL = "xgboost"


# --- loading helpers ---------------------------------------------------------
def load_model(name: str, track: str = "full"):
    """Load one saved classifier (.pkl) for a given track."""
    return joblib.load(config.model_path(name, track))


def _data(track: str = "full"):
    """
    Return everything the explainers need for a track:
      X_train_scaled, X_test_scaled  -> model-ready inputs
      X_test_raw                     -> original blood values (for display)
      y_test                         -> true labels (to pick example patients)
    All subset to the track's feature list.
    """
    features = config.FEATURE_SETS[track]
    Xtr_s, Xte_s, _, y_test = load_processed(scaled=True)
    _, Xte_raw, _, _ = load_processed(scaled=False)
    return (
        Xtr_s[features], Xte_s[features], Xte_raw[features],
        y_test.reset_index(drop=True),
    )


# --- SHAP --------------------------------------------------------------------
def shap_explanation(model, X_scaled, X_display):
    """
    Compute SHAP values on scaled inputs, but attach the ORIGINAL blood values
    for display. Returns a shap.Explanation for the 'anaemic' (class 1) outcome.

    Tree models differ in output shape: RandomForest returns a class axis
    (n, features, 2); XGBoost returns (n, features). We normalise to class 1.
    """
    explainer = shap.TreeExplainer(model)
    expl = explainer(X_scaled)

    values = expl.values
    base = expl.base_values
    if values.ndim == 3:                 # RandomForest-style (n, feat, classes)
        values = values[:, :, 1]
        base = base[:, 1]

    return shap.Explanation(
        values=values,
        base_values=base,
        data=X_display.values,           # show real CBC numbers, not z-scores
        feature_names=list(X_display.columns),
    )


def shap_global(track: str = "full", model_name: str = SHOWCASE_MODEL):
    """Global SHAP: beeswarm (per-patient spread) + mean-|SHAP| importance bar."""
    model = load_model(model_name, track)
    _, Xte_s, Xte_raw, _ = _data(track)
    expl = shap_explanation(model, Xte_s, Xte_raw)

    # Beeswarm — each dot is a patient; colour = high/low feature value.
    shap.plots.beeswarm(expl, show=False)
    plt.title(f"SHAP summary ({model_name}, {track} track)")
    plt.tight_layout()
    plt.gcf().savefig(config.FIGURES_DIR / f"fig_shap_beeswarm_{track}.png", dpi=120)
    plt.close("all")

    # Bar — mean absolute SHAP = overall feature importance ranking.
    shap.plots.bar(expl, show=False)
    plt.title(f"SHAP feature importance ({model_name}, {track} track)")
    plt.tight_layout()
    plt.gcf().savefig(config.FIGURES_DIR / f"fig_shap_bar_{track}.png", dpi=120)
    plt.close("all")
    return expl


def shap_local(track: str = "full", model_name: str = SHOWCASE_MODEL,
               patient: str = "anaemic"):
    """
    Local SHAP: a waterfall for ONE patient, showing how each feature pushed
    the prediction up (toward anaemic) or down from the baseline.
    `patient` = 'anaemic' or 'healthy' picks the first matching test patient.
    """
    model = load_model(model_name, track)
    _, Xte_s, Xte_raw, y_test = _data(track)
    expl = shap_explanation(model, Xte_s, Xte_raw)

    want = 1 if patient == "anaemic" else 0
    idx = int(np.where(y_test.values == want)[0][0])

    shap.plots.waterfall(expl[idx], show=False)
    plt.title(f"SHAP explanation — one {patient} patient ({model_name})")
    plt.tight_layout()
    plt.gcf().savefig(
        config.FIGURES_DIR / f"fig_shap_waterfall_{track}_{patient}.png",
        dpi=120, bbox_inches="tight",
    )
    plt.close("all")
    return idx


# --- predictions from all three models (for the comparison view) -------------
def predict_all(raw_row: pd.DataFrame, track: str = "full") -> dict:
    """
    Run ALL THREE models on one raw patient row. Returns
    {model_name: (prediction, P(anaemic))}. Used by the app's comparison panel
    so the "compare three models" objective is visible live, not just offline.
    """
    scaler = joblib.load(config.SCALER_PATH)
    features = config.FEATURE_SETS[track]
    scaled = pd.DataFrame(
        scaler.transform(raw_row[config.FEATURES]), columns=config.FEATURES
    )[features]
    out = {}
    for name in config.MODEL_NAMES:
        model = load_model(name, track)
        out[name] = (int(model.predict(scaled)[0]),
                     float(model.predict_proba(scaled)[0, 1]))
    return out


# --- LIME --------------------------------------------------------------------
def build_lime_explainer(track: str = "full", model_name: str = SHOWCASE_MODEL):
    """
    Build a LIME explainer + a raw-input predict function for one model/track.

    We train LIME on the ORIGINAL (unscaled) values and wrap predict_proba so it
    scales inputs internally — so LIME's rules read in real units (e.g.
    'Hemoglobin <= 10.2') instead of z-scores. Returned so both the figure
    helper and the live app can reuse the exact same setup.
    """
    model = load_model(model_name, track)
    features = config.FEATURE_SETS[track]
    scaler = joblib.load(config.SCALER_PATH)

    def predict_proba_raw(X_raw):
        # LIME hands us RAW rows (track features only). The scaler was fit on ALL
        # config.FEATURES, so pad to the full frame, scale, then subset back.
        full = pd.DataFrame(0.0, index=range(len(X_raw)), columns=config.FEATURES)
        full[features] = X_raw
        scaled = pd.DataFrame(scaler.transform(full), columns=config.FEATURES)
        return model.predict_proba(scaled[features])

    explainer = LimeTabularExplainer(
        training_data=load_processed(scaled=False)[0][features].values,
        feature_names=features,
        class_names=["Not anaemic", "Anaemic"],
        mode="classification",
        discretize_continuous=True,
        random_state=config.RANDOM_STATE,
    )
    return explainer, predict_proba_raw, features


def lime_explain_row(raw_row: pd.DataFrame, track: str = "full",
                     model_name: str = SHOWCASE_MODEL):
    """LIME explanation for ONE arbitrary raw patient row (used by the app)."""
    explainer, predict_fn, features = build_lime_explainer(track, model_name)
    return explainer.explain_instance(
        raw_row[features].values[0], predict_fn, num_features=len(features),
    )


def lime_example(track: str = "full", model_name: str = SHOWCASE_MODEL,
                 patient: str = "anaemic"):
    """One LIME explanation for a fixed test patient -> saved figure (Phase 6)."""
    explainer, predict_fn, features = build_lime_explainer(track, model_name)
    _, _, Xte_raw, y_test = _data(track)

    want = 1 if patient == "anaemic" else 0
    idx = int(np.where(y_test.values == want)[0][0])
    exp = explainer.explain_instance(
        Xte_raw.iloc[idx].values, predict_fn, num_features=len(features),
    )

    fig = exp.as_pyplot_figure()
    fig.set_size_inches(7, 4)
    plt.title(f"LIME explanation — one {patient} patient ({model_name})")
    plt.tight_layout()
    fig.savefig(
        config.FIGURES_DIR / f"fig_lime_{track}_{patient}.png",
        dpi=120, bbox_inches="tight",
    )
    plt.close("all")
    return exp


def generate_all():
    """Regenerate every Phase 6 figure: SHAP (full + nohb) and one LIME."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("SHAP global (full track) ...")
    shap_global("full")
    print("SHAP local waterfall — anaemic patient (full track) ...")
    shap_local("full", patient="anaemic")
    print("SHAP local waterfall — healthy patient (full track) ...")
    shap_local("full", patient="healthy")

    print("SHAP global (nohb track) — the honest contrast ...")
    shap_global("nohb")

    print("LIME example (full track) ...")
    lime_example("full", patient="anaemic")

    print(f"\nDone. Figures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    generate_all()
