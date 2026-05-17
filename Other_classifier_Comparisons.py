from pathlib import Path
import torch 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = Path("/Users/adrianne/Desktop/AI_EX_1/Assignment_19_May/ionosphere.csv")
TARGET_COLUMN = "Class"
LABEL_NAMES = {"g": "Good", "b": "Bad"}
ions_df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

numeric_features = ions_df.drop(columns=[TARGET_COLUMN])
selector = VarianceThreshold(threshold=0.0)
selected_array = selector.fit_transform(numeric_features)
selected_columns = numeric_features.columns[selector.get_support()]

X = pd.DataFrame(selected_array, columns=selected_columns, index=ions_df.index)
y = ions_df[TARGET_COLUMN]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")

# ── 5. Feature Scaling ───────────────────────────────────────
# Used by Logistic Regression and SVM; tree-based models use raw features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
    "SVM":                 (SVC(kernel="rbf", random_state=42),                 True),
    "Decision Tree":       (DecisionTreeClassifier(max_depth=5, random_state=42), False),
    "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42), False),
}
# The boolean flag indicates whether the model needs scaled features

# ── 7. Train & Evaluate Each Model ───────────────────────────
for name, (model, use_scaling) in models.items():
    X_tr = X_train_scaled if use_scaling else X_train
    X_te = X_test_scaled if use_scaling else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy : {(y_pred == y_test).mean():.4f}")

# ── 8. Model Comparison Summary ──────────────────────────────
results = {}
for name, (model, use_scaling) in models.items():
    X_te = X_test_scaled if use_scaling else X_test
    y_pred = model.predict(X_te)
    accuracy = (y_pred == y_test).mean()
    results[name] = accuracy

print("\n── Model Accuracy Comparison ──")
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:<25} {acc:.4f}")
