# CLA 4 – ML Model Deployment Strategies and Model Serving using FastAPI

## Dataset
Titanic – Machine Learning from Disaster (Kaggle), using `train.csv`.

Dataset page:
https://www.kaggle.com/competitions/titanic/data

## Tasks Completed
1. Model training and evaluation using accuracy.
2. Online inference using one unseen passenger.
3. Batch inference using 20 unseen passengers.
4. Streaming inference using 10 unseen passengers with a 1-second delay.
5. FastAPI model serving with `/health` and `/predict`.
6. Swagger UI validation and Python client testing.
7. Observation report.

## Project Structure
- `data/train.csv` – Titanic training dataset
- `train_model.py` – preprocessing, training, evaluation and model saving
- `online_inference.py` – one-record inference
- `batch_inference.py` – 20-record batch inference
- `streaming_inference.py` – 10-record streaming inference
- `app.py` – FastAPI application
- `client.py` – Python API client
- `model.pkl` – saved trained pipeline
- `batch_predictions.csv` – generated batch predictions
- `requirements.txt` – dependencies
- `screenshots/` – output evidence
- `report/CLA4_Report.pdf` – observations report

## Run
```bash
python -m pip install -r requirements.txt
python train_model.py
python online_inference.py
python batch_inference.py
python streaming_inference.py
uvicorn app:app --reload
```

Open Swagger UI:
`http://127.0.0.1:8000/docs`

## API
### GET /health
Checks whether the API is running and the model is loaded.

### POST /predict
Accepts passenger details and returns a survival prediction and probability.

Example:
```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 29,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 80.0,
  "Embarked": "C"
}
```

## Notes
The model is trained once and the same saved pipeline is reused for online, batch, streaming and FastAPI inference, as required by the assignment.
