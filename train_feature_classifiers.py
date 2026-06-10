import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from pytorch_tabnet.tab_model import TabNetClassifier

from config import Config


# =========================================
# Load Features
# =========================================

csv_path = os.path.join(
    Config.OUTPUT_DIR,
    "mri_extracted_features.csv"
)

df = pd.read_csv(csv_path)

print("\nDataset Loaded:")
print(df.shape)


# =========================================
# Features and Labels
# =========================================

feature_cols = [
    "mask_area",
    "perimeter",
    "circularity",
    "bbox_width",
    "bbox_height",
    "aspect_ratio",
    "mean_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity"
]

X = df[feature_cols].fillna(0)

y = df["disease_label_id"]


# =========================================
# Train/Test Split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================
# Standardization
# =========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================
# XGBoost
# =========================================

print("\n" + "="*60)
print("Training XGBoost...")
print("="*60)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    objective="multi:softmax",
    num_class=5,
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)

xgb_preds = xgb_model.predict(X_test_scaled)

xgb_acc = accuracy_score(y_test, xgb_preds)

print(f"\nXGBoost Accuracy: {xgb_acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, xgb_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, xgb_preds))


# =========================================
# TabNet
# =========================================

print("\n" + "="*60)
print("Training TabNet...")
print("="*60)

tabnet_model = TabNetClassifier(
    seed=42,
    verbose=1
)

tabnet_model.fit(
    X_train_scaled,
    y_train.values,
    eval_set=[(X_test_scaled, y_test.values)],
    max_epochs=20,
    patience=5,
    batch_size=16,
    virtual_batch_size=8
)

tabnet_preds = tabnet_model.predict(X_test_scaled)

tabnet_acc = accuracy_score(y_test, tabnet_preds)

print(f"\nTabNet Accuracy: {tabnet_acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, tabnet_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tabnet_preds))


# =========================================
# Save Results
# =========================================

results_path = os.path.join(
    Config.OUTPUT_DIR,
    "classifier_results.txt"
)

with open(results_path, "w") as f:

    f.write("XGBoost Results\n")
    f.write("="*50 + "\n")
    f.write(f"Accuracy: {xgb_acc:.4f}\n\n")

    f.write(classification_report(y_test, xgb_preds))

    f.write("\n\n")

    f.write("TabNet Results\n")
    f.write("="*50 + "\n")
    f.write(f"Accuracy: {tabnet_acc:.4f}\n\n")

    f.write(classification_report(y_test, tabnet_preds))


print("\nResults saved to:")
print(results_path)