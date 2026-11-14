import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

# 1. Read both label files and stack them into one table
parts = [pd.read_csv(RAW / "labels_marksheet.csv")]
labels = pd.concat(parts, ignore_index=True)

# 2. Sanity checks
print(labels.groupby(["doc_type", "label"]).size())
print("duplicate filenames:", labels["filename"].duplicated().sum())

missing = 0
for _, row in labels.iterrows():
    folder = "genuine" if row["label"] == 0 else "manipulated"
    if not (RAW / folder / row["doc_type"] / row["filename"]).exists():
        missing += 1
        print("missing:", row["filename"])
print("missing files:", missing)

# 3. Save the combined file
labels.to_csv(RAW / "labels.csv", index=False)
print("Saved", len(labels), "rows to labels.csv")