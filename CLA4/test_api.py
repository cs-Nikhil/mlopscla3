import requests

base = "http://127.0.0.1:8000"

print("HEALTH TEST")
r = requests.get(base + "/health", timeout=10)
print(r.status_code, r.json())

print("\nVALID PREDICT TEST")
payload = {
    "Pclass": 1, "Sex": "female", "Age": 29,
    "SibSp": 0, "Parch": 0, "Fare": 80.0, "Embarked": "C"
}
r = requests.post(base + "/predict", json=payload, timeout=10)
print(r.status_code, r.json())

print("\nINVALID PREDICT TEST")
bad_payload = {
    "Pclass": 9, "Sex": "unknown", "Age": 29,
    "SibSp": 0, "Parch": 0, "Fare": 80.0, "Embarked": "X"
}
r = requests.post(base + "/predict", json=bad_payload, timeout=10)
print(r.status_code, r.json())
