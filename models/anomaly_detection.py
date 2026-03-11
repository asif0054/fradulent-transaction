"""
========================================================================
  STEP 3: Anomaly Detection -- Isolation Forest & LOF
========================================================================

UNSUPERVISED ANOMALY DETECTION:
These models don't need labeled data -- they find "weird" transactions
based on how different they are from the majority.

MODEL 1: ISOLATION FOREST
  * Builds random trees that try to "isolate" each data point
  * Anomalies are isolated in fewer splits (shorter path length)
  * Normal points need many splits to be isolated
  * contamination=0.017 tells it ~1.7% of data is anomalous

MODEL 2: LOCAL OUTLIER FACTOR (LOF)
  * Measures the local density of each point vs. its neighbors
  * Points in low-density regions are flagged as outliers
  * n_neighbors=20 means it checks the 20 nearest points

USAGE:
  from models.anomaly_detection import run_isolation_forest, run_lof
========================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

SAVED_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "saved_models"
)


def run_isolation_forest(X_train, X_test, y_test, contamination=0.017, save=True):
    """
    STEP 3a: Train and evaluate an Isolation Forest model.
    
    HOW IT WORKS:
    1. Build an ensemble of random isolation trees on training data
    2. For each test point, measure how quickly it gets isolated
    3. Short path -> anomaly (fraud), Long path -> normal (legit)
    4. IsolationForest outputs: -1 for anomaly, +1 for normal
       We convert this to: 1 for fraud, 0 for legitimate
    """
    print("\n" + "=" * 60)
    print("[STEP 3a] Isolation Forest")
    print("=" * 60)
    
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        max_samples='auto'
    )
    
    print("   Training Isolation Forest...")
    iso_forest.fit(X_train)
    print("   [OK] Model trained!")
    
    raw_predictions = iso_forest.predict(X_test)
    predictions = np.where(raw_predictions == -1, 1, 0)
    scores = iso_forest.decision_function(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, 
                                    target_names=["Legitimate", "Fraud"],
                                    zero_division=0)
    
    print(f"\n   Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"\n{report}")
    
    if save:
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        model_path = os.path.join(SAVED_MODELS_DIR, "isolation_forest.joblib")
        joblib.dump(iso_forest, model_path)
        print(f"   [SAVED] Model saved to: {model_path}")
    
    return {
        "model": iso_forest,
        "predictions": predictions,
        "scores": scores,
        "report": report,
        "accuracy": accuracy
    }


def run_lof(X_train, X_test, y_test, n_neighbors=20, contamination=0.017):
    """
    STEP 3b: Train and evaluate a Local Outlier Factor model.
    
    HOW LOF WORKS:
    1. For each point, calculate the local density
    2. Compare each point's density to its neighbors' densities
    3. If a point's density is much lower than its neighbors',
       it's in a sparse area -> likely an outlier (fraud)
    """
    print("\n" + "=" * 60)
    print("[STEP 3b] Local Outlier Factor (LOF)")
    print("=" * 60)
    
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        n_jobs=-1
    )
    
    print("   Training Local Outlier Factor...")
    lof.fit(X_train)
    print("   [OK] Model trained!")
    
    raw_predictions = lof.predict(X_test)
    predictions = np.where(raw_predictions == -1, 1, 0)
    scores = lof.decision_function(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions,
                                    target_names=["Legitimate", "Fraud"],
                                    zero_division=0)
    
    print(f"\n   Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"\n{report}")
    
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    model_path = os.path.join(SAVED_MODELS_DIR, "lof_model.joblib")
    joblib.dump(lof, model_path)
    print(f"   [SAVED] Model saved to: {model_path}")
    
    return {
        "model": lof,
        "predictions": predictions,
        "scores": scores,
        "report": report,
        "accuracy": accuracy
    }


if __name__ == "__main__":
    from preprocessing import load_and_preprocess
    
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess(use_smote=False)
    
    print("\n" + "==" * 30)
    print("  ANOMALY DETECTION RESULTS")
    print("==" * 30)
    
    iso_results = run_isolation_forest(X_train, X_test, y_test)
    lof_results = run_lof(X_train, X_test, y_test)
