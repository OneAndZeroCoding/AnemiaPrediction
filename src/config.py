"""
Central configuration: paths, feature lists, constants.

WHY this file exists:
- One place defines where data/models/figures live and what the features are.
- Notebooks, src/ scripts, and the Streamlit app all import from here, so we
  never hardcode a path or a column name in two places (which drifts and breaks).

NOTE: FEATURES below is FINALISED once the dataset is locked (Phase 1).
Until then it reflects the columns we expect from the chosen dataset.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# PROJECT_ROOT = the AnemiaPrediction/ folder (this file is in src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Specific files
RAW_DATASET = RAW_DIR / "anemia.csv"          # <- set to the real filename in Phase 1
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# Processed split files (the Phase 3 -> Phase 4 handoff contract)
X_TRAIN_PATH = PROCESSED_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_DIR / "X_test.csv"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.csv"
Y_TEST_PATH = PROCESSED_DIR / "y_test.csv"

# Saved model files (what the trainer must produce)
MODEL_PATHS = {
    "random_forest": MODELS_DIR / "random_forest.pkl",
    "svm": MODELS_DIR / "svm.pkl",
    "xgboost": MODELS_DIR / "xgboost.pkl",
}

# --- Data schema (FINALISE in Phase 1 once dataset is locked) ----------------
# TARGET = the column we predict. FEATURES = model inputs.
TARGET = "Result"

# Placeholder feature list — update to match the chosen dataset's real columns.
FEATURES = [
    "Gender",
    "Hemoglobin",
    "MCH",
    "MCHC",
    "MCV",
]

# Features used in the [MARKS-BOOSTER] leakage robustness experiment
# (everything except the definitional Hemoglobin column).
FEATURES_NO_HB = [f for f in FEATURES if f.lower() != "hemoglobin"]

# --- Reproducibility ---------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
