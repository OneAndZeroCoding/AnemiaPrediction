"""
Phase 5 — Evaluation. Compute the five required metrics for every model in
both tracks, save comparison figures, and write reports/metrics.json.

The five metrics (defined for the viva):
  * accuracy  : overall fraction correct.
  * precision : of those predicted anaemic, how many really were. (avoids false alarms)
  * recall    : of those truly anaemic, how many we caught. (avoids missed cases)
  * f1        : harmonic mean of precision & recall (single balanced score).
  * roc_auc   : ranking quality across all thresholds (1.0 perfect, 0.5 random).

Run:  python -m src.evaluate
"""
import json

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config
from .preprocess import load_processed

PRETTY = {"random_forest": "Random Forest", "svm": "SVM", "xgboost": "XGBoost"}


def evaluate_model(model, X_test, y_test) -> dict:
    """Return the five metrics for one fitted model on the test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]   # P(anaemic) for AUC-ROC
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def _roc_figure(models, X_test, y_test, track):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=PRETTY[name])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_title(f"ROC curves — {track} track")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"fig_roc_{track}.png", dpi=120)
    plt.close(fig)


def _confusion_figure(models, X_test, y_test, track):
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
    for ax, (name, model) in zip(axes, models.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, ax=ax, colorbar=False,
            display_labels=["Not", "Anaemic"],
        )
        ax.set_title(PRETTY[name])
    fig.suptitle(f"Confusion matrices — {track} track")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"fig_confusion_{track}.png", dpi=120)
    plt.close(fig)


def evaluate_all(save: bool = True) -> dict:
    """Evaluate every model in every track; save metrics.json + figures."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _, X_test_full, _, y_test = load_processed(scaled=True)

    results = {}
    for track, features in config.FEATURE_SETS.items():
        X_test = X_test_full[features]
        models = {n: joblib.load(config.model_path(n, track)) for n in config.MODEL_NAMES}
        results[track] = {n: evaluate_model(m, X_test, y_test) for n, m in models.items()}
        _roc_figure(models, X_test, y_test, track)
        _confusion_figure(models, X_test, y_test, track)

    if save:
        with open(config.METRICS_PATH, "w") as f:
            json.dump(results, f, indent=2)
    return results


def best_model(results: dict, track: str = "full", metric: str = "f1") -> str:
    """Name of the best model in a track by the chosen metric."""
    track_res = results[track]
    return max(track_res, key=lambda n: track_res[n][metric])


if __name__ == "__main__":
    res = evaluate_all()
    for track in res:
        print(f"\n=== {track} track ===")
        print(f"{'model':14s}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}{'auc':>8}")
        for name, m in res[track].items():
            print(f"{name:14s}{m['accuracy']:8.3f}{m['precision']:8.3f}"
                  f"{m['recall']:8.3f}{m['f1']:8.3f}{m['roc_auc']:8.3f}")
        print(f"best by f1: {best_model(res, track)}")
    print(f"\nSaved -> {config.METRICS_PATH}")
