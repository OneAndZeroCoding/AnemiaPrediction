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

# The three models we train and compare.
MODEL_NAMES = ["random_forest", "svm", "xgboost"]


def model_path(name: str, track: str = "full"):
    """Path for a saved model. track='full' -> name.pkl ; track='nohb' -> name_nohb.pkl."""
    suffix = "" if track == "full" else f"_{track}"
    return MODELS_DIR / f"{name}{suffix}.pkl"


# Convenience: full-track paths (what the Streamlit app loads).
MODEL_PATHS = {name: model_path(name, "full") for name in MODEL_NAMES}

# --- Data schema (FINALISED in Phase 1 — confirmed against real data) --------
# Dataset: Biswaranjan Rao "Anemia Dataset", 1421 rows x 6 cols, no missing values.
# TARGET = the column we predict. FEATURES = model inputs.
TARGET = "Result"          # 0 = not anaemic, 1 = anaemic

# Confirmed columns. NOTE: Gender is ALREADY encoded as 0/1 in the raw CSV,
# so no label-encoding step is needed (the original plan assumed text).
FEATURES = [
    "Gender",
    "Hemoglobin",
    "MCH",
    "MCHC",
    "MCV",
]

# Gender encoding observed in the data (see leakage analysis in 01_eda):
#   0 -> Female (Result=1 when Hb < ~11.95, i.e. WHO 12 g/dL cutoff)
#   1 -> Male   (Result=1 when Hb < ~13.40, i.e. WHO 13 g/dL cutoff)
GENDER_MAP = {0: "Female", 1: "Male"}

# Features used in the [MARKS-BOOSTER] leakage robustness experiment
# (everything except the definitional Hemoglobin column).
FEATURES_NO_HB = [f for f in FEATURES if f.lower() != "hemoglobin"]

# Two experiment tracks compared throughout the project:
#   'full' = all features  (LEAKY: includes Hb, the definitional feature)
#   'nohb' = Hb excluded   (HONEST signal from the other red-cell indices)
FEATURE_SETS = {"full": FEATURES, "nohb": FEATURES_NO_HB}

# --- Reproducibility ---------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
