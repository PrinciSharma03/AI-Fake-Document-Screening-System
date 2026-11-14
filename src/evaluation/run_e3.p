import argparse
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src" / "evaluation"))
from run_experiments import make_models, score_row, RESULTS_CSV     # same models and metrics as E0-E2

DATA = ROOT / "data" / "processed" / "e3"
BASE = ["blur_score", "noise_level", "edge_density", "contrast", "ocr_confidence"]


def pick(df, which):
    reg = [c for c in df.columns if c.startswith("reg_")]
    rel = [c for c in df.columns if c.startswith("rel_")]
    return {"baseline": BASE, "regions": reg, "relative": rel, "baseline+relative": BASE + rel}[which]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--set", required=True, choices=["baseline", "regions", "relative", "baseline+relative"])
    args = parser.parse_args()

    train = pd.read_csv(DATA / "features_train.csv")
    test = pd.read_csv(DATA / "features_test.csv")
    assert set(train["group_id"]).isdisjoint(set(test["group_id"])), "groups overlap between train and test!"
    cols = pick(train, args.set)
    print(f"{args.name}: feature set '{args.set}', {len(cols)} features")
    print(f"train: {len(train)} rows, styles {sorted(set(train['edit_style']) - {'none'})}")
    print(f"test : {len(test)} rows, styles {sorted(set(test['edit_style']) - {'none'})} (unseen)\n")

    rows = []
    for name, model in make_models().items():                       # A. cross-validation on training styles
        pred = cross_val_predict(model, train[cols], train["label"], groups=train["group_id"],
                                 cv=GroupKFold(n_splits=5))
        rows.append(score_row(args.name, name, "5-fold CV (train styles)", train["label"], pred, len(cols)))
    for name, model in make_models().items():                       # B. unseen style + unseen groups
        model.fit(train[cols], train["label"])
        rows.append(score_row(args.name, name, "UNSEEN style + groups", test["label"],
                              model.predict(test[cols]), len(cols)))

    table = pd.DataFrame(rows)
    print(table.drop(columns=["experiment", "n_features"]).to_string(index=False))
    old = pd.read_csv(RESULTS_CSV)
    pd.concat([old[old["experiment"] != args.name], table], ignore_index=True).to_csv(RESULTS_CSV, index=False)
    print("\nAdded to", RESULTS_CSV, "(E0-E2 rows are unchanged)")


if __name__ == "__main__":
    main()