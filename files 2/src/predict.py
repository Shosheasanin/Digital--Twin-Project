"""
Prediction module - uses trained model to predict water potability.
"""

import pandas as pd
import numpy as np
import joblib
from preprocess import handle_missing_values, EXPECTED_COLUMNS


def predict_from_dataframe(df, model, scaler):
    """Predict potability for all rows in dataframe."""
    df_clean = handle_missing_values(df)

    # Get only feature columns
    feature_cols = [c for c in EXPECTED_COLUMNS if c in df_clean.columns]
    X = df_clean[feature_cols].copy()

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    result = df.copy()
    result["Prediction"] = predictions
    result["Probability_Safe"] = probabilities.round(4)
    result["Status"] = result["Prediction"].map({0: "Unsafe", 1: "Safe"})

    return result


def predict_single_sample(sample_dict, model, scaler):
    """Predict for a single water sample (dict of feature values)."""
    df = pd.DataFrame([sample_dict])
    return predict_from_dataframe(df, model, scaler).iloc[0].to_dict()


def get_prediction_summary(result_df):
    """Summarize predictions."""
    total = len(result_df)
    safe = int((result_df["Prediction"] == 1).sum())
    unsafe = total - safe

    return {
        "total_samples": total,
        "safe_count": safe,
        "unsafe_count": unsafe,
        "safe_percentage": round(100 * safe / total, 2) if total > 0 else 0,
        "unsafe_percentage": round(100 * unsafe / total, 2) if total > 0 else 0,
        "avg_safety_probability": round(float(result_df["Probability_Safe"].mean()), 4)
    }
