from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from A_LR_Victor import get_weighted_metrics


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


class DeepFNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)


class DeeperFNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 12),
            nn.ReLU(),
            nn.Linear(12, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_and_evaluate(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(100):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predicted_classes = torch.argmax(test_outputs, dim=1)

    y_true = y_test_tensor.numpy()
    y_pred = predicted_classes.numpy()

    return y_true, y_pred


if __name__ == "__main__":
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

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

    models = [
        ("SimpleFNN", SimpleFNN(X_train_tensor.shape[1])),
        ("DeepFNN", DeepFNN(X_train_tensor.shape[1])),
        ("DeeperFNN", DeeperFNN(X_train_tensor.shape[1])),
    ]

    results = []

    for model_name, model in models:
        y_true, y_pred = train_and_evaluate(
            model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
        )

        print("=" * 55)
        print(model_name)
        print("=" * 55)
        print(f"Accuracy: {round(accuracy_score(y_true, y_pred), 4)}")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=["Bad", "Good"],
                zero_division=0,
            )
        )
        results.append(
            {"Model": model_name}
            | get_weighted_metrics(y_true, y_pred, [0, 1], ["Bad", "Good"])
        )

    print("=" * 55)
    print("Deep Learning Architecture Comparison")
    print("=" * 55)
    print(pd.DataFrame(results).to_string(index=False))
