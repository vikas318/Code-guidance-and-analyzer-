import os
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
FEATURE_FILE = "features.csv"
LABEL_FILE = "labels.csv"
LABEL_MAP_FILE = "label_mapping.json"
MODEL_OUTPUT = "best_model.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2
if not os.path.exists(FEATURE_FILE) or not os.path.exists(LABEL_FILE):
    print("Error: Dataset not found. Please run dataset_preprocessor.py first.")
    exit(1)

print("Loading dataset...")
X = pd.read_csv(FEATURE_FILE)
y = pd.read_csv(LABEL_FILE)["label"]
target_names = None
if os.path.exists(LABEL_MAP_FILE):
    with open(LABEL_MAP_FILE, "r") as f:
        mapping = json.load(f)
        target_names = [mapping[str(i)] for i in range(len(mapping))]
print("Samples:", len(X))
print("Features:", len(X.columns))
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))
print("\n" + "="*40)
print("       Training RandomForest")
print("="*40)
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f"\nRandomForest Accuracy: {rf_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_preds, target_names=target_names))
best_model = rf_model
best_accuracy = rf_accuracy
best_name = "RandomForest"
if XGB_AVAILABLE:
    print("\n" + "="*40)
    print("         Training XGBoost")
    print("="*40)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss"
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    xgb_preds = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_preds)
    print(f"\nXGBoost Accuracy: {xgb_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, xgb_preds, target_names=target_names))
    if xgb_accuracy > best_accuracy:
        best_model = xgb_model
        best_accuracy = xgb_accuracy
        best_name = "XGBoost"
print("\n" + "="*40)
print("     Feature Importance (Top 10)")
print("="*40)

importances = best_model.feature_importances_
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print(importance_df.head(10).to_string(index=False))
print("\n" + "="*40)
joblib.dump(best_model, MODEL_OUTPUT)
print(f"WINNER: {best_name} (Accuracy: {best_accuracy:.4f})")
print(f"Model saved to '{MODEL_OUTPUT}'")
print("="*40 + "\n")
