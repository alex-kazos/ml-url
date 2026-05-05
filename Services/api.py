'''
this will later be used to call the model/application from the command line, e.g. with uvicorn:
uvicorn Services.api:app --host
'''
from fastapi import FastAPI
from pydantic import BaseModel

from Services.inference_service import inference_service

app = FastAPI(title="Phishing URL Detection API")


class PredictRequest(BaseModel):
    url: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest):
    return inference_service(payload.url)
