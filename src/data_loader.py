"""
Loading the dataset — the single place that reads the raw CSV.

WHY a dedicated module: every other file (EDA notebook, preprocessing, app)
needs the data, but only THIS file knows where it lives and how it's cleaned.
If the dataset ever changes, we fix it here once.
"""
import pandas as pd

from . import config


def load_raw() -> pd.DataFrame:
    """Load the original CSV exactly as-is. Never modify this on disk."""
    return pd.read_csv(config.RAW_DATASET)


def load_clean() -> pd.DataFrame:
    """
    Load the data with exact-duplicate rows removed.

    WHY drop duplicates: 887 of 1421 rows are exact copies. If identical rows
    land in BOTH the train and test sets, the model is effectively tested on
    data it trained on -> dishonestly inflated scores (train/test leakage).
    Removing them gives ~534 unique patients and trustworthy evaluation.
    """
    df = load_raw().drop_duplicates().reset_index(drop=True)
    return df


if __name__ == "__main__":
    raw = load_raw()
    clean = load_clean()
    print(f"raw rows:   {len(raw)}")
    print(f"clean rows: {len(clean)}  (removed {len(raw) - len(clean)} duplicates)")
