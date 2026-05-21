import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Services.inference_service import inference_service


API_MODEL_NAME = os.getenv("API_MODEL_NAME", "XGBoost")

app = FastAPI(title="Phishing URL Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://www.alexkazos.com",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    url: str


@app.get("/health")
def health():
    return {"status": "ok", "model": API_MODEL_NAME}


@app.post("/predict")
def predict(payload: PredictRequest):
    return inference_service(payload.url, model_name=API_MODEL_NAME)
