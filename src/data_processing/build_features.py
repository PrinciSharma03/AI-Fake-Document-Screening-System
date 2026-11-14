import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "features.csv"
sys.path.append(str(ROOT / "src" / "feature_extraction"))
from feature_extraction import extract_features      # teammate's function

FEATURE_NAMES = ["blur_score", "noise_level", "edge_density", "contrast", "ocr_confidence"]


def main():
    labels = pd.read_csv(RAW / "labels.csv")
    rows = []
    for n, (_, r) in enumerate(labels.iterrows(), start=1):
        folder = "genuine" if r["label"] == 0 else "manipulated"
        path = RAW / folder / r["doc_type"] / r["filename"]
        feats = extract_features(str(path))
        assert list(feats.keys()) == FEATURE_NAMES, f"Wrong keys/order: {list(feats.keys())}"
        rows.append({**r.to_dict(), **feats})
        if n % 20 == 0:
            print(n, "images done")
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print("Saved", len(table), "rows to", OUT)
    print(table.groupby(["doc_type", "label"])[FEATURE_NAMES].mean().round(3))


if __name__ == "__main__":
    main()