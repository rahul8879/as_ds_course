from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from joblib import load


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "iris_knn.joblib"
META_PATH = MODELS / "metadata.json"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Please run: python src/train.py"
        )
    model = load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


def main():
    st.set_page_config(page_title="Iris KNN Predictor", page_icon="🌸", layout="centered")
    st.title("🌸 Iris KNN Predictor")
    st.write("Enter features (cm) to predict the Iris species.")

    # Typical ranges for Iris
    sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.5, value=5.1, step=0.1)
    sepal_width  = st.number_input("Sepal Width (cm)",  min_value=2.0, max_value=5.0, value=3.5, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.5, value=1.4, step=0.1)
    petal_width  = st.number_input("Petal Width (cm)",  min_value=0.1, max_value=3.0, value=0.2, step=0.1)

    model = None
    meta = None
    try:
        model, meta = load_artifacts()
    except FileNotFoundError as e:
        st.warning(str(e))

    if st.button("Predict"):
        if model is None:
            st.stop()
        X = np.array([[sepal_length, sepal_width, petal_length, petal_width]], dtype=float)
        proba = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        target_names = meta["target_names"]
        pred_label = target_names[pred_idx]

        st.subheader("Prediction")
        st.success(f"Species: {pred_label}")
        st.caption("Class probabilities:")
        for i, cls in enumerate(target_names):
            st.progress(float(proba[i]), text=f"{cls}: {proba[i]:.3f}")


if __name__ == "__main__":
    main()

