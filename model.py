import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
    return model, encoder, features, accuracy

if __name__ == "__main__":
    train_and_evaluate()
