"""
Phase 8 — light sanity tests (pytest).

These are NOT exhaustive unit tests; per the mini-project scope they confirm the
pipeline holds together and behaves sanely:

  * artifacts load (all 6 models + scaler + metrics),
  * the processed split is the fixed, expected shape,
  * a clearly-anaemic patient is predicted anaemic (and a healthy one is not),
  * the documented findings hold (full track ~perfect, nohb track ~chance),
  * invalid inputs are rejected gracefully.

Run:  python -m pytest -q          (from the project root)
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config
from src.preprocess import load_processed
from src.explain import SHOWCASE_MODEL, shap_explanation


# --- fixtures ----------------------------------------------------------------
@pytest.fixture(scope="module")
def scaler():
    return joblib.load(config.SCALER_PATH)


@pytest.fixture(scope="module")
def best_model():
    return joblib.load(config.model_path(SHOWCASE_MODEL, "full"))


def _scale(raw: pd.DataFrame, scaler) -> pd.DataFrame:
    return pd.DataFrame(scaler.transform(raw[config.FEATURES]), columns=config.FEATURES)


# --- artifacts exist & load --------------------------------------------------
def test_all_six_models_and_scaler_exist():
    assert config.SCALER_PATH.exists(), "scaler.pkl missing"
    for track in config.FEATURE_SETS:
        for name in config.MODEL_NAMES:
            p = config.model_path(name, track)
            assert p.exists(), f"missing model: {p.name}"
            joblib.load(p)  # must deserialise without error


def test_metrics_json_has_both_tracks_and_five_metrics():
    metrics = json.loads(config.METRICS_PATH.read_text())
    assert set(metrics) == {"full", "nohb"}
    needed = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    for track in metrics.values():
        for model_metrics in track.values():
            assert needed.issubset(model_metrics)


# --- processed split is the fixed contract -----------------------------------
def test_processed_split_shape_and_columns():
    X_train, X_test, y_train, y_test = load_processed(scaled=True)
    assert list(X_train.columns) == config.FEATURES
    assert len(X_train) == 427 and len(X_test) == 107   # 534 unique patients, 80/20
    assert set(np.unique(y_train)) <= {0, 1}


# --- behaviour: anaemic vs healthy ------------------------------------------
def test_known_anaemic_patient_predicted_anaemic(best_model, scaler):
    # Very low haemoglobin -> unambiguously anaemic.
    raw = pd.DataFrame([{"Gender": 0, "Hemoglobin": 7.0, "MCH": 20, "MCHC": 30, "MCV": 75}])
    pred = int(best_model.predict(_scale(raw, scaler))[0])
    assert pred == 1


def test_known_healthy_patient_predicted_not_anaemic(best_model, scaler):
    # Comfortably high haemoglobin -> not anaemic.
    raw = pd.DataFrame([{"Gender": 1, "Hemoglobin": 16.0, "MCH": 29, "MCHC": 34, "MCV": 90}])
    pred = int(best_model.predict(_scale(raw, scaler))[0])
    assert pred == 0


def test_predict_proba_in_unit_interval(best_model, scaler):
    raw = pd.DataFrame([{"Gender": 0, "Hemoglobin": 11.0, "MCH": 25, "MCHC": 32, "MCV": 85}])
    p = float(best_model.predict_proba(_scale(raw, scaler))[0, 1])
    assert 0.0 <= p <= 1.0


# --- documented findings hold ------------------------------------------------
def test_full_track_near_perfect_and_nohb_near_chance():
    metrics = json.loads(config.METRICS_PATH.read_text())
    full_aucs = [m["roc_auc"] for m in metrics["full"].values()]
    nohb_aucs = [m["roc_auc"] for m in metrics["nohb"].values()]
    assert min(full_aucs) > 0.90, "full track should be near-perfect (leakage)"
    assert max(nohb_aucs) < 0.75, "nohb track should collapse toward chance"


# --- explainability sanity ---------------------------------------------------
def test_shap_top_feature_is_hemoglobin(best_model, scaler):
    raw = pd.DataFrame([{"Gender": 0, "Hemoglobin": 7.5, "MCH": 20, "MCHC": 30, "MCV": 75}])
    expl = shap_explanation(best_model, _scale(raw, scaler), raw)
    top = config.FEATURES[int(np.abs(expl[0].values).argmax())]
    assert top == "Hemoglobin"


# --- invalid input handled gracefully ----------------------------------------
def test_missing_feature_raises():
    # A row missing a required feature must not silently predict.
    bad = pd.DataFrame([{"Gender": 0, "Hemoglobin": 12.0}])  # MCH/MCHC/MCV absent
    with pytest.raises((KeyError, ValueError)):
        bad[config.FEATURES]  # selecting the required columns must fail


def test_nonnumeric_input_rejected(scaler):
    bad = pd.DataFrame([{"Gender": 0, "Hemoglobin": "low", "MCH": 20, "MCHC": 30, "MCV": 75}])
    with pytest.raises((ValueError, TypeError)):
        scaler.transform(bad[config.FEATURES])
