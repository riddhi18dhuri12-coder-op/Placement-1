"""
preprocessing.py
-----------------
Handles data cleaning, encoding and train/test splitting for the
placement dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

FEATURE_COLUMNS = [
    "CGPA", "Aptitude_Score", "Technical_Skills", "Communication_Skills",
    "Projects_Count", "Internships_Count", "Certifications_Count",
    "Backlogs", "Branch"
]
TARGET_COLUMN = "Placement_Status"


def load_data(path):
    df = pd.read_csv(path)
    # Handle missing values (if any) — median for numeric, mode for categorical
    numeric_cols = df.select_dtypes(include="number").columns
    for col in df.columns:
        if col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    # Drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def encode_features(df, branch_encoder=None, fit=True):
    df = df.copy()
    if fit:
        branch_encoder = LabelEncoder()
        df["Branch"] = branch_encoder.fit_transform(df["Branch"])
    else:
        df["Branch"] = branch_encoder.transform(df["Branch"])
    return df, branch_encoder


def get_train_test_split(df, test_size=0.2, random_state=42):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
