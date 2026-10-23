import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

MODEL_PATH = "model.pkl"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

df = pd.read_csv("data/train.csv")
_, unseen = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["Survived"]
)

records = unseen.iloc[:10]
model = joblib.load(MODEL_PATH)

print("STREAMING INFERENCE")
for _, row in records.iterrows():
    item = row[FEATURES].to_frame().T
    prediction = int(model.predict(item)[0])
    label = "Survived" if prediction == 1 else "Did not survive"
    print(f"PassengerId={int(row['PassengerId'])} -> Prediction={prediction} ({label})")
    time.sleep(1)
