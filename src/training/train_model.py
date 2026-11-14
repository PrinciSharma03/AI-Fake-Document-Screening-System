import sys
import joblib
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src" / "data_processing"))
from feature_processing import load_dataset, FEATURE_NAMES

MODEL_PATH = ROOT / "models" / "screening_model.joblib"

MODELS = {
    "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
}


def report(title, y_true, y_pred):
    print(f"\n{title}")
    print("  accuracy :", round(accuracy_score(y_true, y_pred), 3))
    print("  precision:", round(precision_score(y_true, y_pred, zero_division=0), 3))
    print("  recall   :", round(recall_score(y_true, y_pred, zero_division=0), 3))
    print("  confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_true, y_pred))


def main():
    X, y, groups = load_dataset()

    # A. Hold-out test: whole groups go to train OR test, never both
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx]))
    print(f"train rows: {len(train_idx)}, test rows: {len(test_idx)}")
    for name, model in MODELS.items():
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        report(f"{name} (hold-out test)", y.iloc[test_idx], model.predict(X.iloc[test_idx]))

    # B. 5-fold cross-validation, also grouped (steadier with a small dataset)
    cv_scores = {}
    for name, model in MODELS.items():
        pred = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits=5))
        report(f"{name} (5-fold grouped CV)", y, pred)
        cv_scores[name] = accuracy_score(y, pred)

    # C. Keep the model with the better CV accuracy, retrain on all data, save it
    best = max(cv_scores, key=cv_scores.get)
    final_model = MODELS[best].fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final_model, "feature_names": FEATURE_NAMES, "name": best}, MODEL_PATH)
    print(f"\nSaved {best} to {MODEL_PATH}")


if __name__ == "__main__":
    main()