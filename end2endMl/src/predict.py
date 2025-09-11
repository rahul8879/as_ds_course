from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from joblib import load


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "iris_knn.joblib"
META_PATH = MODELS / "metadata.json"


def load_artifacts():
    model = load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


def predict_sample(features: List[float]) -> Tuple[str, float, Dict[str, float]]:
    """Predict class and probability from [sepal_len, sepal_wid, petal_len, petal_wid]."""
    model, meta = load_artifacts()
    X = np.array(features, dtype=float).reshape(1, -1)
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    target_names = meta["target_names"]
    pred_label = target_names[pred_idx]
    probs = {target_names[i]: float(proba[i]) for i in range(len(target_names))}
    return pred_label, float(proba[pred_idx]), probs


def parse_args():
    p = argparse.ArgumentParser(description="Predict Iris class from 4 features (cm)")
    p.add_argument("--sepal-length", type=float, required=True)
    p.add_argument("--sepal-width", type=float, required=True)
    p.add_argument("--petal-length", type=float, required=True)
    p.add_argument("--petal-width", type=float, required=True)
    return p.parse_args()


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model not found at {MODEL_PATH}. Train it first with: python src/train.py"
        )
    args = parse_args()
    feats = [args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]
    label, conf, probs = predict_sample(feats)
    print(f"Prediction: {label} (confidence={conf:.3f})")
    print("Class probabilities:")
    for k, v in probs.items():
        print(f"  - {k}: {v:.3f}")


if __name__ == "__main__":
    main()

