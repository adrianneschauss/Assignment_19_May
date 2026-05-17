from pathlib import Path
import torch 
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


DATA_PATH = Path("/Users/adrianne/Desktop/AI_EX_1/Assignment_19_May/ionosphere.csv")
TARGET_COLUMN = "Class"
LABEL_NAMES = {"g": "Good", "b": "Bad"}


if __name__ == "__main__":
    print("=" * 55)
    print("STEP 1: Understand and Prepare the Data")
    print("=" * 55)

    ions_df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    print(f"Dataset shape: {ions_df.shape}")
    print("\nData type counts:")
    print(ions_df.dtypes.value_counts())

    ions_df.dtypes.value_counts().plot(kind="bar", rot=0)
    plt.title("Data Type Counts")
    plt.tight_layout()
    plt.show()

    missing_values = ions_df.isna().sum().sum()
    duplicate_rows = ions_df.duplicated().sum()
    print(f"\nMissing values: {missing_values}")
    print(f"Duplicate rows: {duplicate_rows}")

    print("\nOriginal class distribution:")
    print(ions_df[TARGET_COLUMN].map(LABEL_NAMES).value_counts())

    numeric_features = ions_df.drop(columns=[TARGET_COLUMN])
    constant_columns = numeric_features.columns[numeric_features.nunique() <= 1].tolist()
    if constant_columns:
        print(f"\nDropping constant columns: {constant_columns}")

    selector = VarianceThreshold(threshold=0.0)
    selected_array = selector.fit_transform(numeric_features)
    selected_columns = numeric_features.columns[selector.get_support()]
    X = pd.DataFrame(selected_array, columns=selected_columns, index=ions_df.index)
    y = ions_df[TARGET_COLUMN]

    print("\nFeature summary:")
    print(X.describe())

    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(),
        y,
        random_state=104,
        test_size=0.25,
        shuffle=True,
        stratify=y,
    )

    ros = RandomOverSampler(random_state=42)
    X_train_over, y_train_over = ros.fit_resample(X_train, y_train)

    print("\nTraining class distribution after oversampling:")
    print(y_train_over.map(LABEL_NAMES).value_counts())
    print(f"Training set size: {X_train_over.shape[0]} (was {X_train.shape[0]})")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_over)
    X_test_scaled = scaler.transform(X_test)

    print(
        f"\nTrain/Test shapes: X_train={X_train_scaled.shape}, "
        f"X_test={X_test_scaled.shape}, y_train={y_train_over.shape}, y_test={y_test.shape}"
    )

    print("=" * 55)
    print("STEP 2: Machine Learning Workflow")
    print("=" * 55)
    # k-fold fit once iteratievely 
    model = LogisticRegression(random_state=42, max_iter=2000, solver="liblinear")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train_over, cv=kfold, scoring='accuracy')
    #fit to all of the data once and find predictions
    model.fit(X_train_scaled, y_train_over)
    y_pred = model.predict(X_test_scaled)


    print("=" * 55)
    print("Classification Results")
    print("=" * 55)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(
        "\nResults:\n"
        + classification_report(
            y_test,
            y_pred,
            labels=["g", "b"],
            target_names=["Good", "Bad"],
        )
    )
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred, labels=['g', 'b'])}")

    print("\nPredicted class distribution:")
    print(pd.Series(y_pred).map(LABEL_NAMES).value_counts())

   

