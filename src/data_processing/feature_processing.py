import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = ROOT / "data" / "processed" / "features.csv"
FEATURE_NAMES = ["blur_score", "noise_level", "edge_density", "contrast", "ocr_confidence"]


def load_dataset():
    """Return the feature table (X), labels (y) and group IDs."""
    df = pd.read_csv(FEATURES_CSV)
    return df[FEATURE_NAMES], df["label"], df["group_id"]