import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def update_index_html(accuracy):
    """Update index.html with model details, dataset, accuracy, and CI/CD deployment status."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML CI/CD Assignment</title>
</head>
<body style="font-family: Arial, sans-serif; max-width: 760px; margin: 50px auto; padding: 0 20px; line-height: 1.6;">
    <h1>Machine Learning CI/CD Pipeline</h1>
    <p><b>ML Model Name:</b> Logistic Regression</p>
    <p><b>Dataset Used:</b> Iris Dataset</p>
    <p><b>Model Accuracy:</b> {accuracy * 100:.2f}%</p>
    <p><b>CI/CD Deployment Status:</b> Passed / Deployed</p>
</body>
</html>
"""
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def train_and_evaluate():
    data = pd.read_csv("iris_dataset.csv")
    features = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
    X = data[features]
    encoder = LabelEncoder()
    y = encoder.fit_transform(data["species"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Model: Logistic Regression")
    print("Dataset: Iris")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    update_index_html(accuracy)

    return model, encoder, features, accuracy

if __name__ == "__main__":
    train_and_evaluate()
