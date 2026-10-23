from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(
    title="Titanic ML Model API",
    description="FastAPI model serving for the CLA 4 Titanic assignment.",
    version="1.0.0"
)

model = joblib.load("model.pkl")

class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str = Field(..., pattern="^(male|female)$")
    Age: float = Field(..., ge=0, le=100)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    Embarked: str = Field(..., pattern="^(S|C|Q)$")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(passenger: Passenger):
    data = pd.DataFrame([passenger.model_dump()])
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][prediction])
    return {
        "prediction": prediction,
        "label": "Survived" if prediction == 1 else "Did not survive",
        "probability": round(probability, 4)
    }
