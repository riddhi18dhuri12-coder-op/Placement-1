"""
train_models.py
----------------
Trains Logistic Regression, Decision Tree, Random Forest and XGBoost
classifiers on the placement dataset.

What this does, end to end:
  1. Checks class balance and applies class-weighting (or XGBoost's
     scale_pos_weight) if the dataset is skewed.
  2. Runs Stratified 5-fold cross-validation for every model (more
     reliable than a single train/test split) and reports mean +/- std.
  3. Runs a small GridSearchCV hyperparameter search per model.
  4. Evaluates the tuned models on a held-out test set: Accuracy,
     Precision, Recall, F1-score + confusion matrix.
  5. Picks the best model by test-set F1, then wraps it with
     CalibratedClassifierCV so predict_proba outputs are trustworthy
     probabilities (important since the dashboard shows a % directly).
  6. Saves BOTH the calibrated model (used for predictions) and the raw
     uncalibrated model (used for feature_importances_ / SHAP, since
     calibration wrappers don't expose those directly).
  7. Separately trains a small Random Forest "uncertainty ensemble" —
     regardless of which model wins on F1 — so the app can show a
     confidence band using the spread of predictions across trees.
  8. Saves a timestamped copy of every artifact under models/versions/
     so retraining never silently destroys the previous version.
"""

import os
import json
import shutil
import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from preprocessing import load_data, encode_features, get_train_test_split, scale_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "placement_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
VERSIONS_DIR = os.path.join(MODELS_DIR, "versions")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VERSIONS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Small, fast hyperparameter grids — kept intentionally small so training
# stays quick; widen these if you have time/compute to spare.
PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.1, 1.0, 3.0, 10.0],
    },
    "Decision Tree": {
        "max_depth": [4, 6, 8, 10],
        "min_samples_leaf": [1, 5, 10],
    },
    "Random Forest": {
        "n_estimators": [150, 300],
        "max_depth": [6, 8, 12],
    },
    "XGBoost": {
        "n_estimators": [150, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.08, 0.12],
    },
}


def evaluate(name, y_test, y_pred):
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1_Score": round(f1_score(y_test, y_pred), 4),
    }


def build_base_models(scale_pos_weight):
    """Base estimators with class-imbalance handling wired in."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", random_state=42,
            scale_pos_weight=scale_pos_weight,
        ),
    }


def main():
    df = load_data(DATA_PATH)
    df_enc, branch_encoder = encode_features(df, fit=True)

    # ---- Class imbalance check ----
    class_counts = df_enc["Placement_Status"].value_counts()
    pos, neg = class_counts.get(1, 0), class_counts.get(0, 0)
    imbalance_ratio = neg / max(pos, 1)
    print(f"Class balance -> Placed: {pos}  Not placed: {neg}  "
          f"(ratio {imbalance_ratio:.2f}:1)")
    if imbalance_ratio > 1.5 or imbalance_ratio < 1 / 1.5:
        print("=> Meaningful imbalance detected: using class_weight='balanced' "
              "(and scale_pos_weight for XGBoost).")
    scale_pos_weight = imbalance_ratio if imbalance_ratio >= 1 else 1.0

    X_train, X_test, y_train, y_test = get_train_test_split(df_enc)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    base_models = build_base_models(scale_pos_weight)

    cv_summary = []
    tuned_models = {}
    results = []

    for name, model in base_models.items():
        # ---- Cross-validation on the training fold (pre-tuning sanity check) ----
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=CV, scoring="f1")
        cv_summary.append({
            "Model": name, "CV_F1_Mean": round(cv_scores.mean(), 4),
            "CV_F1_Std": round(cv_scores.std(), 4),
        })
        print(f"{name}: CV F1 = {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # ---- Hyperparameter tuning ----
        grid = GridSearchCV(model, PARAM_GRIDS[name], cv=CV, scoring="f1", n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_est = grid.best_estimator_
        tuned_models[name] = best_est
        print(f"{name}: best params = {grid.best_params_}")

        # ---- Held-out test evaluation ----
        y_pred = best_est.predict(X_test_scaled)
        metrics = evaluate(name, y_test, y_pred)
        metrics["Best_Params"] = json.dumps(grid.best_params_)
        results.append(metrics)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Placed", "Placed"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - {name}")
        fname = name.lower().replace(" ", "_")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"cm_{fname}.png"), dpi=150)
        plt.close()

    cv_df = pd.DataFrame(cv_summary)
    cv_df.to_csv(os.path.join(BASE_DIR, "reports", "cv_comparison.csv"), index=False)

    results_df = pd.DataFrame(results).sort_values("F1_Score", ascending=False)
    print("\n=== Model Comparison (held-out test set, tuned) ===")
    print(results_df.drop(columns=["Best_Params"]).to_string(index=False))
    results_df.to_csv(os.path.join(BASE_DIR, "reports", "model_comparison.csv"), index=False)

    # Bar chart comparing models
    plt.figure(figsize=(8, 5))
    results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1_Score"]].plot(
        kind="bar", figsize=(9, 5)
    )
    plt.title("Model Performance Comparison (tuned)")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "06_model_comparison.png"), dpi=150)
    plt.close()

    # ---- Select best model by test F1 ----
    best_name = results_df.iloc[0]["Model"]
    best_model_raw = tuned_models[best_name]
    print(f"\nBest model selected: {best_name}")

    # ---- Calibrate probabilities ----
    # Refit a calibrated wrapper on the training data using the tuned
    # hyperparameters, so predict_proba is well-calibrated for the gauge
    # shown in the dashboard.
    calibrated = CalibratedClassifierCV(best_model_raw, method="sigmoid", cv=5)
    calibrated.fit(X_train_scaled, y_train)
    y_pred_cal = calibrated.predict(X_test_scaled)
    cal_metrics = evaluate(f"{best_name} (Calibrated)", y_test, y_pred_cal)
    print(f"Calibrated model test metrics: {cal_metrics}")

    # Calibration curve figure (reliability diagram)
    prob_pos = calibrated.predict_proba(X_test_scaled)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, prob_pos, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, marker="o", label=f"{best_name} (calibrated)")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives (actually placed)")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "08_calibration_curve.png"), dpi=150)
    plt.close()

    # ---- Train a dedicated uncertainty ensemble (always a Random Forest) ----
    # Used by the app to show a confidence band, regardless of which model
    # actually wins on F1 — tree-vote spread is an easy, cheap uncertainty
    # signal that works the same way every time.
    uncertainty_model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=7, class_weight="balanced", n_jobs=-1
    )
    uncertainty_model.fit(X_train_scaled, y_train)

    # ---- Save artifacts ----
    joblib.dump(calibrated, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(best_model_raw, os.path.join(MODELS_DIR, "best_model_raw.pkl"))
    joblib.dump(uncertainty_model, os.path.join(MODELS_DIR, "uncertainty_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(branch_encoder, os.path.join(MODELS_DIR, "branch_encoder.pkl"))

    info = {
        "best_model": best_name,
        "metrics": results_df.iloc[0].drop("Best_Params").to_dict(),
        "best_params": json.loads(results_df.iloc[0]["Best_Params"]),
        "calibrated_metrics": cal_metrics,
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "class_balance": {"placed": int(pos), "not_placed": int(neg),
                           "imbalance_ratio": round(imbalance_ratio, 3)},
    }
    with open(os.path.join(MODELS_DIR, "best_model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # Feature importance (for tree-based best model) -> used for skill-gap weighting
    if hasattr(best_model_raw, "feature_importances_"):
        importances = pd.Series(
            best_model_raw.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        importances.to_csv(os.path.join(MODELS_DIR, "feature_importances.csv"))

        plt.figure(figsize=(7, 5))
        importances.plot(kind="barh", color="teal")
        plt.gca().invert_yaxis()
        plt.title(f"Feature Importance - {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "07_feature_importance.png"), dpi=150)
        plt.close()

    # ---- Model versioning: snapshot everything under models/versions/<timestamp>/ ----
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = os.path.join(VERSIONS_DIR, stamp)
    os.makedirs(version_dir, exist_ok=True)
    for fname in ["best_model.pkl", "best_model_raw.pkl", "uncertainty_model.pkl",
                  "scaler.pkl", "branch_encoder.pkl", "best_model_info.json"]:
        src = os.path.join(MODELS_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(version_dir, fname))
    print(f"\nSaved versioned snapshot to {version_dir}")
    print("Saved latest artifacts (best_model.pkl [calibrated], best_model_raw.pkl, "
          "uncertainty_model.pkl, scaler.pkl, branch_encoder.pkl) to models/")


if __name__ == "__main__":
    main()
