import pandas as pd
import joblib

MODEL_PATH = "model.pkl"
TEST_SIZE = 0.20
SEED = 42

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

df = pd.read_csv("data/train.csv")
_, unseen = __import__("sklearn.model_selection").model_selection.train_test_split(
    df, test_size=TEST_SIZE, random_state=SEED, stratify=df["Survived"]
)

model = joblib.load(MODEL_PATH)
record = unseen.iloc[[0]][FEATURES]
prediction = int(model.predict(record)[0])
label = "Survived" if prediction == 1 else "Did not survive"

print("ONLINE INFERENCE")
print(record.to_string(index=False))
print(f"\nPrediction: {prediction} ({label})")
