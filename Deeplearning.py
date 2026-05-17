
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
    X, y, test_size=0.8, random_state=42
)

print(y_test.to_frame())
print(X_test)
print(type(X_train), type(X_test), type(y_test.to_frame()), type(y_train.to_frame()))


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

hot_encoded_train =[]
hot_encoded_test = [] 

for i in y_train:
    if i== 'b':
        hot_encoded_train.append(0)
    else:
        hot_encoded_train.append(1)
for j in y_test:
    if j== 'b':
        hot_encoded_test.append(0)
    else:
        hot_encoded_test.append(1)



X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(np.array(hot_encoded_train), dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(np.array(hot_encoded_test), dtype=torch.long)

# 5. Define model


class SimpleFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(33, 16),   # input → hidden1
            nn.ReLU(),
            nn.Linear(16, 8),   # hidden1 → hidden2
            nn.ReLU(),
            nn.Linear(8, 2)     # hidden2 → output
        )

    def forward(self, x):
        return self.net(x)


model = SimpleFNN()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 7. Training loop
for epoch in range(100):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


with torch.no_grad():
    predictions = model(X_test)
    predicted_classes = torch.argmax(predictions, dim=1)
    accuracy = (predicted_classes == y_test).float().mean()

print(f"Test Accuracy: {accuracy:.2f}")

with torch.no_grad():
    outputs = model(X_test)
    predicted_classes = torch.argmax(outputs, dim=1)


y_true = y_test.numpy()
y_pred = predicted_classes.numpy()

# Print precision, recall, f1-score for each class
print(classification_report(y_true, y_pred, target_names=["Good", "Bad"]))
