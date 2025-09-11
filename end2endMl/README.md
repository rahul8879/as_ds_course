End-to-End Iris KNN (Training → Prediction → UI)

Overview
- Trains a simple KNN classifier on sklearn’s Iris dataset.
- Saves a production-ready pipeline (scaler + KNN) and metadata.
- Provides a CLI predictor and a Streamlit UI to test inputs.

Quickstart
1) Install deps (preferably in a virtualenv):
   pip install -r requirements.txt

2) Train and save the model:
   python src/train.py

3) Run a CLI prediction:
   python src/predict.py \
     --sepal-length 5.1 --sepal-width 3.5 \
     --petal-length 1.4 --petal-width 0.2

4) Launch the UI (opens at http://localhost:8501):
   streamlit run app.py

Project Layout
- src/train.py: Trains pipeline, evaluates, saves model and metadata.
- src/predict.py: Loads pipeline + metadata; predicts from CLI or function.
- app.py: Streamlit UI with inputs and class probabilities.
- models/: Saved artifacts (created on first train).

Notes
- Feature order follows sklearn Iris: [sepal length, sepal width, petal length, petal width] (cm).
- You can tweak KNN params (e.g., neighbors) in train.py.
- The saved pipeline includes scaling for robust inference.

