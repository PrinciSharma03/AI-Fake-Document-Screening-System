import sys
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "screening_model.joblib"
sys.path.append(str(ROOT / "src" / "feature_extraction"))
from feature_extraction import extract_features      # Member 2's function

# Rules for "Inconclusive". These are starting guesses, to be checked on
# blurry/rotated test images on Day 5.
MIN_OCR_CONFIDENCE = 60.0    # below this the text was not read reliably
MIN_BLUR_SCORE = 100.0       # below this the image is too blurry
SUSPICIOUS_RISK = 50.0       # risk score at or above this = Suspicious

_bundle = None               # the model is loaded once, not retrained on every upload


def _load_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def quality_problems(features):
    """Return a list of reasons why the image is too poor to judge."""
    problems = []
    if features.get("ocr_confidence", 100.0) < MIN_OCR_CONFIDENCE:
        problems.append(f"Text could not be read reliably (OCR confidence {features['ocr_confidence']:.0f}).")
    if features.get("blur_score", 1e9) < MIN_BLUR_SCORE:
        problems.append(f"Image is too blurry (sharpness score {features['blur_score']:.0f}).")
    return problems


def predict(features: dict) -> dict:
    """features = dict of feature values. Returns status, risk score and reasons."""
    bundle = _load_bundle()
    names = bundle["feature_names"]
    missing = [n for n in names if n not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    model = bundle["model"]
    row = pd.DataFrame([[features[n] for n in names]], columns=names)   # same order as training
    manipulated_col = list(model.classes_).index(1)
    risk = float(model.predict_proba(row)[0][manipulated_col]) * 100

    problems = quality_problems(features)
    if problems:
        status, reasons = "Inconclusive", problems
    elif risk >= SUSPICIOUS_RISK:
        status, reasons = "Suspicious", [f"Model risk score {risk:.0f} is at or above {SUSPICIOUS_RISK:.0f}."]
    else:
        status, reasons = "Genuine-looking", [f"Model risk score {risk:.0f} is below {SUSPICIOUS_RISK:.0f}."]

    return {
        "status": status,
        "risk_score": round(risk, 1),
        "reasons": reasons,
        "features": {n: features[n] for n in names},
    }


def predict_image(image_path: str) -> dict:
    """The function the backend calls: image in, result out."""
    return predict(extract_features(image_path))


if __name__ == "__main__":
    path = sys.argv[1]
    print(predict_image(path))