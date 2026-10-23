import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

MODEL_PATH = "model.pkl"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

df = pd.read_csv("data/train.csv")
_, unseen = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["Survived"]
)

records = unseen.iloc[:20].copy()
model = joblib.load(MODEL_PATH)
records["Prediction"] = model.predict(records[FEATURES]).astype(int)
records["Prediction_Label"] = records["Prediction"].map(
    {0: "Did not survive", 1: "Survived"}
)

records.to_csv("batch_predictions.csv", index=False)
print("BATCH INFERENCE")
print(records[["PassengerId", "Sex", "Pclass", "Prediction", "Prediction_Label"]].to_string(index=False))
print("\nSaved: batch_predictions.csv")
