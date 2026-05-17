from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("/Users/adrianne/Desktop/AI_EX_1/Assignment_19_May/ionosphere.csv")
TARGET_COLUMN = "Class"


def get_weighted_metrics(y_true, y_pred, labels, target_names):
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(report["weighted avg"]["precision"], 4),
        "Recall": round(report["weighted avg"]["recall"], 4),
        "F1-score": round(report["weighted avg"]["f1-score"], 4),
    }


def logistic_regression_predictions(X_train_scaled, y_train, X_test_scaled):
    model = LogisticRegression(random_state=42, max_iter=2000, solver="liblinear")
    model.fit(X_train_scaled, y_train)
    return model.predict(X_test_scaled)


if __name__ == "__main__":
    print("=" * 55)
    print("STEP 1: Understand and Prepare the Data")
    print("=" * 55)

    ions_df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    dataset_size_mb = DATA_PATH.stat().st_size / (1024 ** 2)
    num_samples, num_columns = ions_df.shape
    num_features = num_columns - 1

    print(f"Dataset size: {dataset_size_mb} MB")
    print(f"Number of samples: {num_samples}")
    print(f"Number of features: {num_features}")
    print("Data types by column:")
    print(ions_df.dtypes)

    missing_values = ions_df.isna().sum().sum()
    print(f"Missing values: {missing_values}")

    X = ions_df.drop(columns=[TARGET_COLUMN])
    y = ions_df[TARGET_COLUMN]

    print("Feature summary:")
    print(X.describe())

    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(),
        y,
        random_state=104,
        test_size=0.25,
        shuffle=True,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("1.2 Data Understanding")
    print(
        "A radar return signal that is good is a signal pattern that matches a real target, while a bad signal "
        "return is more likely caused by noise and other things like interference. Usually, the "
        "priority is to detect good returns accurately so that signals that correspond to the objects of interest are not missed, "
        "while still keeping bad-return false alarms low."
    )

    print("1.3 Data Preprocessing")
    print(
        "Data cleaning checks the dataset for problems that can distort the model and make the data analysis less effective" 
        "or representative after training. Elements ssuch as missing values, duplicated rows, incorrect formats, or inconsistent values "
        "should usually be removed before data is processed."
    )
    print(
        "Feature scaling puts numeric features on a comparable scale so that large-valued variables do not dominate smaller ones. " \
        "StandardScaler does this by subtracting the training mean and dividing by the training standard deviation, which "
        "often helps models train more stably."
    )
    print(
        "The train-test split of the modelling separates the data into one part for learning patterns (i.e. this is the data" \
        "with which we make the model and define our parameters specifically to the data, in attempt to generalise the model to unseen data)"
        "patterns and another unseen part for evaluation. This gives a more honest estimate of "
        "how well the model generalizes."
    )

    print("=" * 55)
    print("STEP 2: Machine Learning Workflow")
    print("=" * 55)
    print("1. Load and split data")
    print("2. Preprocess features")
    print("3. Choose model: Logistic Regression")
    print("4. Train the model")
    print("5. Make predictions")
    print("6. Evaluate the model")

    model = LogisticRegression(random_state=42, max_iter=2000, solver="liblinear")
    k_value = 5
    kfold = StratifiedKFold(n_splits=k_value, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring="accuracy")

    print("2.2 Evaluation Methods")
    print(f"Train/test split: 75% training, 25% testing")
    print(f"{k_value}-fold cross-validation accuracies: {np.round(scores, 4)}")
    print(f"Cross-validation mean accuracy: {np.round(scores.mean(), 4)}")
    print(
        f"Why k={k_value}: Five folds is sensible for this data (different k's were tested and the accuracy etc. of the model evaluated). This is most likely due"
        "to this value of k striking a balance between not too large and not too small for this particular data set size. It is also a common "
        "default (in other models) that provides a stable estimate, more so than a single split while keeping "
        "computation simple."
    )

    y_pred = logistic_regression_predictions(X_train_scaled, y_train, X_test_scaled)

    print("=" * 55)
    print("Logistic Regression Results")
    print("=" * 55)
    print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 4)}")
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

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    print("=" * 55)
    print("Baseline Majority-Class Results")
    print("=" * 55)
    print(f"Accuracy: {round(accuracy_score(y_test, baseline_pred), 4)}")
    print(
        "\nResults:\n"
        + classification_report(
            y_test,
            baseline_pred,
            labels=["g", "b"],
            target_names=["Good", "Bad"],
            zero_division=0,
        )
    )
    print(
        f"Confusion Matrix:\n"
        f"{confusion_matrix(y_test, baseline_pred, labels=['g', 'b'])}"
    )

    print("\n2.4 Analysis")
    print(
        "A single train and test split is easy to understand and gives a direct estimate of test performance, but it depends on one random partition. "
        "Cross-validation is more reliable because it averages performance across several folds, "
        "so it usually gives a more stable idea of how the model generalizes."
    )
    print(
        " Performance could be improved by tuning Logistic Regression "
        "hyperparameters such as trying different class-weight settings, "
        "and comparing the model with other classifiers. Additional validation experiments and "
        "careful feature analysis could also help identify which variables contribute most to "
        "performance and improve generalization."
    )
    print(
        f"\nBaseline comparison: The majority-class baseline reached an accuracy of "
        f"{round(accuracy_score(y_test, baseline_pred), 4)}, while Logistic Regression reached "
        f"{round(accuracy_score(y_test, y_pred), 4)}. The baseline is useful because it shows how well "
        "a trivial classifier performs by always predicting the most common class. Logistic "
        "Regression should outperform it not just in accuracy, but also in recall and F1 for both of the classes."
    )
