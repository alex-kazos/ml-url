'''
this will later be used to call the model/application from the command line, e.g. with uvicorn:
uvicorn Services.api:app --host
'''
from fastapi import FastAPI
from pydantic import BaseModel

from Services.inference_service import inference_service
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Phishing URL Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    url: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest):
    return inference_service(
        payload.url
    )
