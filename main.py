from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris ML Prediction API")

# Load trained model
try:
    saved_data = joblib.load("model.pkl")
    model = saved_data["model"]
    target_names = saved_data["target_names"]
    model_loaded = True
except Exception:
    model = None
    target_names = []
    model_loaded = False


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {
        "message": "Iris ML Prediction API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded
    }


@app.post("/predict")
def predict(data: IrisInput):
    if not model_loaded:
        return {
            "error": "Model is not loaded"
        }

    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction),
        "class": target_names[prediction]
    }