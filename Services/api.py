import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Services.inference_service import inference_service


def _allowed_origins() -> list[str]:
    origins = os.getenv("API_ALLOWED_ORIGINS", "http://localhost:5173")
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


API_MODEL_NAME = os.getenv("API_MODEL_NAME", "XGBoost")

app = FastAPI(title="Phishing URL Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
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
