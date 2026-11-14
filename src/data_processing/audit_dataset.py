import hashlib
from collections import Counter
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
FEATURES_CSV = ROOT / "data" / "processed" / "features.csv"


def image_path(row):
    folder = "genuine" if row["label"] == 0 else "manipulated"
    return RAW / folder / row["doc_type"] / row["filename"]


def main():
    labels = pd.read_csv(RAW / "labels.csv")
    problems = []

    # 1. Is the dataset balanced?
    counts = labels["label"].value_counts()
    print("Rows per class (0 = genuine, 1 = manipulated):")
    print(counts.to_string(), "\n")
    if counts.max() != counts.min():
        problems.append("Classes are not balanced")

    # 2. Does every group have 1 genuine original and at least 1 edited copy?
    per_group = labels.groupby("group_id")["label"].agg(
        genuine=lambda s: int((s == 0).sum()), fake=lambda s: int((s == 1).sum()))
    bad = per_group[(per_group["genuine"] != 1) | (per_group["fake"] < 1)]
    print(f"Groups: {len(per_group)}, groups with a wrong structure: {len(bad)}\n")
    if len(bad):
        problems.append(f"{len(bad)} groups do not have exactly 1 genuine + 1 or more manipulated")

    # 3. Are labels consistent with file names and folders?
    says_edit = labels["filename"].str.contains("_edit")
    mismatch = labels[says_edit != (labels["label"] == 1)]
    missing = [r["filename"] for _, r in labels.iterrows() if not image_path(r).exists()]
    print("Labels that disagree with the file name:", len(mismatch))
    print("Labelled files missing on disk:", len(missing), "\n")
    if len(mismatch):
        problems.append("Some labels disagree with file names")
    if missing:
        problems.append("Some labelled files are missing")

    # 4. Duplicates, formats and sizes (a class must not differ in format or size)
    hashes, seen, info = {}, [], []
    for _, r in labels.iterrows():
        p = image_path(r)
        if not p.exists():
            continue
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in hashes:
            seen.append((r["filename"], hashes[h]))
        hashes[h] = r["filename"]
        with Image.open(p) as im:
            info.append((r["label"], im.format, im.size, im.mode))
    print("Exact duplicate images:", len(seen))
    for label in (0, 1):
        print(f"  class {label}:", dict(Counter(i[1:] for i in info if i[0] == label)))
    print()
    if seen:
        problems.append("Duplicate images found")
    for fmt_size in [{(i[1], i[2], i[3]) for i in info if i[0] == lab} for lab in (0, 1)]:
        pass
    if {(i[1], i[2], i[3]) for i in info if i[0] == 0} != {(i[1], i[2], i[3]) for i in info if i[0] == 1}:
        problems.append("Genuine and manipulated images differ in format or size")

    # 5. Leakage: no group may appear in both train and test
    groups = labels["group_id"]
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
                               .split(labels, labels["label"], groups))
    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    print("Groups shared by train and test (hold-out):", len(overlap))
    fold_overlap = 0
    for tr, te in GroupKFold(n_splits=5).split(labels, labels["label"], groups):
        fold_overlap += len(set(groups.iloc[tr]) & set(groups.iloc[te]))
    print("Groups shared by train and test (5-fold):", fold_overlap, "\n")
    if overlap or fold_overlap:
        problems.append("Train/test leakage")

    # 6. Does features.csv match labels.csv and have no gaps?
    feats = pd.read_csv(FEATURES_CSV)
    print("features.csv rows:", len(feats), "| missing values:", int(feats.isna().sum().sum()))
    if len(feats) != len(labels) or feats.isna().any().any():
        problems.append("features.csv does not match labels.csv or has missing values")

    print("\nRESULT:", "ALL CHECKS PASSED" if not problems else "PROBLEMS FOUND")
    for p in problems:
        print(" -", p)


if __name__ == "__main__":
    main()