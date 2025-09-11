from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def main() -> None:
    # Load dataset
    iris = load_iris()
    X = iris.data  # shape (n_samples, 4)
    y = iris.target  # shape (n_samples,)
    feature_names = list(iris.feature_names)
    target_names = list(iris.target_names)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build pipeline: scaling + KNN
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ]
    )

    # Train
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=target_names))

    # Persist artifacts
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "iris_knn.joblib"
    dump(pipe, model_path)
    print(f"Saved model to: {model_path}")

    meta = {
        "feature_names": feature_names,
        "target_names": target_names,
        "model": "KNeighborsClassifier",
        "scaler": "StandardScaler",
        "params": {"n_neighbors": 5, "weights": "distance"},
    }
    (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved metadata to: {models_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()

