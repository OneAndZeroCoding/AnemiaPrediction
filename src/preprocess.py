"""
Preprocessing: turn the clean dataset into a train/test split + a saved scaler.

This module defines the Phase 3 -> Phase 4 HANDOFF CONTRACT:
  Inputs : cleaned, de-duplicated data (from data_loader.load_clean)
  Outputs: data/processed/{X_train,X_test,y_train,y_test}.csv  (original units)
           models/scaler.pkl                                   (fit on TRAIN only)

WHY split before scaling, and why fit the scaler on train only:
  The test set must imitate "future, unseen patients". If the scaler saw the
  test data when computing its mean/std, information about the test set would
  leak into training. So: split first, fit scaler on train, apply to both.
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config
from .data_loader import load_clean


def split_data(df: pd.DataFrame):
    """
    Stratified train/test split.

    'Stratified' = keep the same anaemic/not-anaemic ratio in BOTH halves, so
    the test set is representative and metrics are trustworthy.
    """
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,                     # <- preserves class balance
        random_state=config.RANDOM_STATE,  # <- same split every run (reproducible)
    )


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    StandardScaler: rescales each feature to mean 0, std 1.

    WHY: SVM measures distances, so a large-range feature (MCV ~85) would
    dominate a small-range one unless all are put on a comparable scale.
    Fit on TRAIN ONLY to avoid leaking test information.
    """
    return StandardScaler().fit(X_train)


def build_and_save():
    """Run the full pipeline and persist the handoff artifacts to disk."""
    df = load_clean()
    X_train, X_test, y_train, y_test = split_data(df)
    scaler = fit_scaler(X_train)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save splits in ORIGINAL units (human-readable, SHAP-friendly).
    X_train.to_csv(config.X_TRAIN_PATH, index=False)
    X_test.to_csv(config.X_TEST_PATH, index=False)
    y_train.to_csv(config.Y_TRAIN_PATH, index=False)
    y_test.to_csv(config.Y_TEST_PATH, index=False)
    joblib.dump(scaler, config.SCALER_PATH)

    return X_train, X_test, y_train, y_test, scaler


def load_processed(scaled: bool = True):
    """
    Load the saved splits. With scaled=True (default) apply the saved scaler,
    so the trainer/app get model-ready data with one call and identical scaling.
    """
    X_train = pd.read_csv(config.X_TRAIN_PATH)
    X_test = pd.read_csv(config.X_TEST_PATH)
    y_train = pd.read_csv(config.Y_TRAIN_PATH).squeeze("columns")
    y_test = pd.read_csv(config.Y_TEST_PATH).squeeze("columns")

    if scaled:
        scaler = joblib.load(config.SCALER_PATH)
        X_train = pd.DataFrame(scaler.transform(X_train), columns=config.FEATURES)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=config.FEATURES)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = build_and_save()
    print("Saved processed splits + scaler.")
    print(f"  train: {X_train.shape}   test: {X_test.shape}")
    print("  train class %:", (y_train.value_counts(normalize=True) * 100).round(1).to_dict())
    print("  test  class %:", (y_test.value_counts(normalize=True) * 100).round(1).to_dict())
