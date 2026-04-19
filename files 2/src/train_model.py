"""
Model training module - trains Random Forest classifier on water quality data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
import joblib
import os

from preprocess import preprocess_pipeline, load_csv


def train_model(csv_path, model_save_path="models/rf_model.pkl",
                scaler_save_path="models/scaler.pkl", test_size=0.2, random_state=42):
    """Train Random Forest on water quality data and save model + scaler."""
    print(f"Loading data from {csv_path}...")
    df = load_csv(csv_path)
    print(f"Data shape: {df.shape}")

    # Preprocess
    X, y, scaler = preprocess_pipeline(df, target_col="Potability", training=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    print(f"\n=== Model Performance ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC AUC:  {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:\n{np.array(metrics['confusion_matrix'])}")

    # Save
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"\nModel saved to {model_save_path}")
    print(f"Scaler saved to {scaler_save_path}")

    return model, scaler, metrics


def load_model(model_path="models/rf_model.pkl", scaler_path="models/scaler.pkl"):
    """Load trained model and scaler."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


if __name__ == "__main__":
    train_model("data/water_potability.csv")
