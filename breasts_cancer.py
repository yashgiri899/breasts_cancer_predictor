import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
def load_data(file_path):
    dataset = pd.read_csv(file_path)
    x = dataset.iloc[:, 1:-1].values  # Features
    y = dataset.iloc[:, -1].values  # Target
    return x, y

# Train model
def train_model(x_train, y_train):
    model = LogisticRegression(C=0.1, max_iter=500)
    model.fit(x_train, y_train)
    return model

# Plot confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix")
    plt.show()

# Main function
def main():
    # Load data
    file_path = "breast_cancer_dataset.csv" # Change if needed
    x, y = load_data(file_path)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Feature Scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Train model
    model = train_model(x_train, y_train)

    # Predictions
    y_pred = model.predict(x_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Visualization
    plot_confusion_matrix(cm)

if __name__ == "__main__":
    main()
