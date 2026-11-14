import cv2
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ID_COLUMNS = ["filename", "doc_type", "label", "group_id"]
ROW_Y = [490, 550, 610, 670, 730]

# Boxes (x0, y0, x1, y1) on a 1000-pixel-wide marksheet. They follow OUR template.
REGIONS = {
    "name": (70, 226, 680, 274),              # student details
    "roll_no": (70, 276, 680, 324),           # student details
}
for i, y in enumerate(ROW_Y):
    REGIONS[f"marks_{i + 1}"] = (690, y - 26, 810, y + 26)   # one subject's marks
REGIONS["total"] = (690, 784, 810, 836)       # total marks
REGIONS["percentage"] = (70, 856, 500, 904)   # result section (never edited by my generator)
REGIONS["result"] = (70, 906, 500, 954)       # result section (never edited by my generator)


def load_gray(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    img = cv2.resize(img, (1000, round(h * 1000 / w)))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def region_stats(gray, box):
    """4 numbers for one region."""
    x0, y0, x1, y1 = box
    patch = gray[y0:y1, x0:x1]
    residual = patch.astype("float64") - cv2.GaussianBlur(patch, (5, 5), 0).astype("float64")
    return {
        "bg": float(np.median(patch)),                    # background brightness
        "ink": float(np.percentile(patch, 5)),            # how dark the text is
        "noise": float(residual.std()),                   # local roughness
        "edge": float((cv2.Canny(patch, 100, 200) > 0).mean()),   # share of edge pixels
    }


def main():
    base = pd.read_csv(PROCESSED / "features.csv")     # IDs + the 5 agreed features
    rows = []
    for _, r in base.iterrows():
        folder = "genuine" if r["label"] == 0 else "manipulated"
        gray = load_gray(RAW / folder / r["doc_type"] / r["filename"])
        row = {}
        for region, box in REGIONS.items():
            for measure, value in region_stats(gray, box).items():
                row[f"{region}_{measure}"] = round(value, 4)
        rows.append(row)
    regions = pd.DataFrame(rows)

    ids = base[ID_COLUMNS]
    pd.concat([ids, regions], axis=1).to_csv(PROCESSED / "features_regions.csv", index=False)
    pd.concat([base, regions], axis=1).to_csv(PROCESSED / "features_combined.csv", index=False)
    print("Region features:", regions.shape[1], "columns,", len(regions), "rows")
    print("Saved features_regions.csv and features_combined.csv")


if __name__ == "__main__":
    main()