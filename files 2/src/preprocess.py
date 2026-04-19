"""
Data preprocessing module for water quality detection.
Handles missing values, scaling, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os


EXPECTED_COLUMNS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]


def load_csv(file_path_or_buffer):
    """Load CSV file from path or uploaded buffer."""
    df = pd.read_csv(file_path_or_buffer)
    return df


def validate_columns(df):
    """Check if required columns exist. Returns (is_valid, missing_cols)."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    return len(missing) == 0, missing


def handle_missing_values(df, strategy="median"):
    """Impute missing numeric values."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy=strategy)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df


def scale_features(df, feature_cols, scaler=None, fit=True):
    """Scale features using StandardScaler."""
    df = df.copy()
    if fit or scaler is None:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    return df, scaler


def preprocess_pipeline(df, target_col="Potability", training=True, scaler=None):
    """Full preprocessing pipeline."""
    df = handle_missing_values(df)

    if training and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df.drop(columns=[target_col]) if target_col in df.columns else df
        y = None

    feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_scaled, scaler = scale_features(X, feature_cols, scaler=scaler, fit=training)

    return X_scaled, y, scaler


def get_summary_stats(df):
    """Return summary statistics for the dataframe."""
    return {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "describe": df.describe().to_dict(),
    }
