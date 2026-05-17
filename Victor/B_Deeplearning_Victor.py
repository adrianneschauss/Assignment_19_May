from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Victor.A_LR_Victor import get_weighted_metrics, logistic_regression_predictions


DATA_PATH = Path("/Users/adrianne/Desktop/AI_EX_1/Assignment_19_May/ionosphere.csv")
TARGET_COLUMN = "Class"


class SimpleFNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    print("=" * 55)
    print("STEP 3: Deep Learning Workflow")
    print("=" * 55)

    print("1. Load and preprocess data")
    ions_df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    X = ions_df.drop(columns=[TARGET_COLUMN])
    y = ions_df[TARGET_COLUMN].map({"b": 0, "g": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("2. Convert data to tensors")
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

    print("3. Define neural network architecture")
    model = SimpleFNN(input_dim=X_train_tensor.shape[1])

    print("4. Define loss function and optimizer")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("5. Train model")
    epochs = []
    losses = []

    for epoch in range(100):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epochs.append(epoch)
        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {round(loss.item(), 4)}")

    print("6. Evaluate model")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predicted_classes = torch.argmax(test_outputs, dim=1)

    y_true = y_test_tensor.numpy()
    y_pred = predicted_classes.numpy()

    print(f"Accuracy: {round(accuracy_score(y_true, y_pred), 4)}")
    print(
        "\nResults:\n"
        + classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Bad", "Good"],
            zero_division=0,
        )
    )
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred, labels=[0, 1])}")

    print("7. Monitor performance")
    plt.plot(epochs, losses)
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.show()

    print("=" * 55)
    print("STEP 4: ML vs Deep Learning Comparison")
    print("=" * 55)

    y_test_ml = y_test.map({0: "b", 1: "g"})
    ml_pred = logistic_regression_predictions(X_train_scaled, y_train.map({0: "b", 1: "g"}), X_test_scaled)

    comparison_df = pd.DataFrame(
        [
            {"Model": "Logistic Regression"}
            | get_weighted_metrics(y_test_ml, ml_pred, ["g", "b"], ["Good", "Bad"]),
            {"Model": "Deep Learning"}
            | get_weighted_metrics(y_true, y_pred, [0, 1], ["Bad", "Good"]),
        ]
    )

    print("4.1 Results Table")
    print(comparison_df.to_string(index=False))
