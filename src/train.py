"""
Train the three models (Random Forest, SVM, XGBoost) and save them.

This is the Phase 4 hand-off script. The trainer runs ONE command:

    python -m src.train

It trains TWO tracks on the EXACT processed split from Phase 3:
  * 'full' : all features (includes Haemoglobin) -> the leaky, ~100% models
  * 'nohb' : Haemoglobin EXCLUDED -> the honest robustness experiment

It deliberately does NOT re-split or re-scale, so evaluation (Phase 5) and SHAP
(Phase 6) stay consistent. We do not pick a winner here — that is Phase 5.
"""
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config
from .preprocess import load_processed


def build_models() -> dict:
    """
    Define the three classifiers with sensible, beginner-safe defaults.

    - RandomForest: many decision trees voting together; robust, little tuning.
    - SVC (SVM): `probability=True` so we get probabilities for AUC-ROC.
    - XGBoost: gradient-boosted trees; the PDF predicts this will win.
    Same random_state everywhere -> reproducible results.
    """
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=config.RANDOM_STATE
        ),
        "svm": SVC(
            kernel="rbf", probability=True, random_state=config.RANDOM_STATE
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
        ),
    }


def train_track(track: str = "full"):
    """Train all three models on one feature track and save them to models/."""
    features = config.FEATURE_SETS[track]
    X_train, X_test, y_train, y_test = load_processed(scaled=True)
    X_train, X_test = X_train[features], X_test[features]   # subset for this track
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        path = config.model_path(name, track)
        joblib.dump(model, path)
        # Quick sanity scores only — the official 5 metrics are Phase 5.
        print(f"[{track:4s}] {name:14s} "
              f"train={model.score(X_train, y_train):.3f} "
              f"test={model.score(X_test, y_test):.3f} -> {path.name}")


def train_and_save():
    """Train both tracks (full + nohb) and save all six models."""
    for track in config.FEATURE_SETS:
        train_track(track)
    print("\nAll models trained and saved to models/ (full + nohb tracks).")
    print("Next: Phase 5 evaluation -> python -m src.evaluate")


if __name__ == "__main__":
    train_and_save()
