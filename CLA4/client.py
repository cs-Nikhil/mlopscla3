import requests

URL = "http://127.0.0.1:8000/predict"

payload = {
    "Pclass": 1,
    "Sex": "female",
    "Age": 29,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 80.0,
    "Embarked": "C"
}

response = requests.post(URL, json=payload, timeout=10)
print("STATUS:", response.status_code)
print("RESPONSE:", response.json())
