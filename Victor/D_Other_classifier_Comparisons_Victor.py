from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from A_LR_Victor import get_weighted_metrics


DATA_PATH = Path("/Users/adrianne/Desktop/AI_EX_1/Assignment_19_May/ionosphere.csv")
TARGET_COLUMN = "Class"


if __name__ == "__main__":
    ions_df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    X = ions_df.drop(columns=[TARGET_COLUMN])
    y = ions_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=42), True),
        "SVM": (SVC(kernel="rbf", random_state=42), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=5, random_state=42), False),
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), False),
    }

    results = []

    for model_name, (model, use_scaling) in models.items():
        X_tr = X_train_scaled if use_scaling else X_train
        X_te = X_test_scaled if use_scaling else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        print("=" * 55)
        print(model_name)
        print("=" * 55)
        print(
            classification_report(
                y_test,
                y_pred,
                labels=["g", "b"],
                target_names=["Good", "Bad"],
                zero_division=0,
            )
        )

        results.append(
            {"Model": model_name}
            | get_weighted_metrics(y_test, y_pred, ["g", "b"], ["Good", "Bad"])
        )

    print("=" * 55)
    print("5.1 ML Improvements Results Table")
    print("=" * 55)
    print(pd.DataFrame(results).to_string(index=False))

    print("\n5.3 Reflection")
    print(
        "The strongest improvement is the model that achieved the best weighted F1 score "
        "and accuracy on the tested tset after training. From the output it appears that SVM and the random forest did the best in " 
        "classification.  "
        "The deep learning improvement can then be compared against this table to judge whether "
        "adding layers helped more than changing the ML algorithm. The FNs do not change that much between each other but rank closely (a bit below)" \
        "the SVM and the random forest."
    )
