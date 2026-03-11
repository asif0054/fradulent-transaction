"""
========================================================================
          STEP 4: XGBoost Supervised Classifier
========================================================================

WHY XGBOOST?
XGBoost (eXtreme Gradient Boosting) is a top-performing ML algorithm
for tabular/structured data like credit card transactions.

HOW IT WORKS:
1. Starts with a simple model (weak learner)
2. Identifies samples it got WRONG
3. Builds a new tree focusing on those errors
4. Adds the new tree to the ensemble (boosting)
5. Repeats N times -- each tree corrects previous mistakes

KEY HYPERPARAMETERS:
  n_estimators   = 200  (number of boosting rounds)
  max_depth      = 6    (depth of each tree -- prevents overfitting)
  learning_rate  = 0.1  (step size -- smaller = more conservative)
  scale_pos_weight      (handles class imbalance automatically)

USAGE:
  from models.xgboost_model import train_xgboost, predict_xgboost
========================================================================
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

SAVED_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "saved_models"
)


def train_xgboost(X_train, y_train, X_test, y_test, save=True):
    """
    STEP 4a: Train an XGBoost classifier for fraud detection.
    
    TRAINING PROCESS:
    1. Calculate scale_pos_weight to handle class imbalance
    2. Initialize XGBClassifier with tuned hyperparameters
    3. Train on the (optionally SMOTE-balanced) training data
    4. Predict on test data
    5. Calculate probabilities for ROC curve
    6. Save model to disk for deployment
    """
    print("\n" + "=" * 60)
    print("[STEP 4] XGBoost Classifier")
    print("=" * 60)
    
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_weight = n_negative / max(n_positive, 1)
    
    print(f"   Training data stats:")
    print(f"      Negative (legit): {n_negative:,}")
    print(f"      Positive (fraud): {n_positive:,}")
    print(f"      scale_pos_weight : {scale_weight:.2f}")
    
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )
    
    print("\n   Training XGBoost (200 rounds)...")
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    print("   [OK] Model trained!")
    
    predictions = xgb_model.predict(X_test)
    probabilities = xgb_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions,
                                    target_names=["Legitimate", "Fraud"],
                                    zero_division=0)
    
    print(f"\n   Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"\n{report}")
    
    if hasattr(X_train, 'columns'):
        feature_names = list(X_train.columns)
    else:
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    
    importance = dict(zip(feature_names, xgb_model.feature_importances_))
    top_5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"   Top 5 Important Features:")
    for name, score in top_5:
        print(f"      {name}: {score:.4f}")
    
    if save:
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        model_path = os.path.join(SAVED_MODELS_DIR, "xgboost_model.joblib")
        joblib.dump(xgb_model, model_path)
        print(f"\n   [SAVED] Model saved to: {model_path}")
    
    return {
        "model": xgb_model,
        "predictions": predictions,
        "probabilities": probabilities,
        "report": report,
        "accuracy": accuracy,
        "feature_importance": importance
    }


def predict_xgboost(model, X_input):
    """
    STEP 4b: Make predictions with a trained XGBoost model.
    
    USED BY THE STREAMLIT APP:
    When a user enters transaction details in the dashboard,
    this function runs the prediction and returns the result.
    """
    if isinstance(model, str):
        model = joblib.load(model)
    
    if isinstance(X_input, pd.Series):
        X_input = X_input.values.reshape(1, -1)
    elif isinstance(X_input, np.ndarray) and X_input.ndim == 1:
        X_input = X_input.reshape(1, -1)
    
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0]
    
    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability[1]),
        "legit_probability": float(probability[0]),
        "label": "FRAUDULENT" if prediction == 1 else "LEGITIMATE"
    }


def load_saved_model(model_name="xgboost_model"):
    """Load a saved model from the saved_models directory."""
    model_path = os.path.join(SAVED_MODELS_DIR, f"{model_name}.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model = joblib.load(model_path)
    print(f"   [OK] Loaded model from: {model_path}")
    return model


if __name__ == "__main__":
    from preprocessing import load_and_preprocess
    
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess(use_smote=True)
    
    print("\n" + "==" * 30)
    print("  XGBOOST TRAINING")
    print("==" * 30)
    
    results = train_xgboost(X_train, y_train, X_test, y_test)
    
    print("\n\nTesting single prediction:")
    sample = X_test.iloc[0] if hasattr(X_test, 'iloc') else X_test[0]
    result = predict_xgboost(results["model"], sample)
    print(f"   Prediction: {result['label']}")
    print(f"   Fraud probability: {result['fraud_probability']:.4f}")
