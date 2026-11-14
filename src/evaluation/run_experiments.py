import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

ROOT = Path(__file__).resolve().parents[2]
ID_COLUMNS = ["filename", "doc_type", "label", "group_id"]
RESULTS_CSV = ROOT / "reports" / "results_table.csv"


def make_models():
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)),
    }


def score_row(experiment, model, evaluation, y_true, y_pred, n_features):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "experiment": experiment, "model": model, "evaluation": evaluation,
        "n_rows": len(y_true), "n_features": n_features,
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="experiment name, e.g. E0_baseline")
    parser.add_argument("--features-csv", default=str(ROOT / "data" / "processed" / "features.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv)
    cols = [c for c in df.columns if c not in ID_COLUMNS]     # every non-ID column is a feature
    X, y, groups = df[cols], df["label"], df["group_id"]
    print(f"{args.name}: {len(df)} rows, {len(cols)} features: {cols}\n")

    rows = []
    # Same hold-out split for all three models; whole groups stay together
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
                               .split(X, y, groups))
    assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx]))
    for name, model in make_models().items():
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        rows.append(score_row(args.name, name, "hold-out", y.iloc[test_idx], pred, len(cols)))

    # Grouped 5-fold cross-validation (steadier on a small dataset)
    for name, model in make_models().items():
        pred = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits=5))
        rows.append(score_row(args.name, name, "5-fold CV", y, pred, len(cols)))

    table = pd.DataFrame(rows)
    print(table.drop(columns=["experiment", "n_features"]).to_string(index=False))

    # Add to the shared results table (re-running the same experiment name replaces its rows)
    if RESULTS_CSV.exists():
        old = pd.read_csv(RESULTS_CSV)
        table = pd.concat([old[old["experiment"] != args.name], table], ignore_index=True)
    table.to_csv(RESULTS_CSV, index=False)
    print("\nSaved to", RESULTS_CSV)


if __name__ == "__main__":
    main()