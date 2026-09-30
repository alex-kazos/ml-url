"""MLflow pyfunc wrapper around the ml-url model bundle.

This file is logged with MLflow's "models from code" feature, so the registry
stores the *whole* bundle (estimator, feature order, character model, lookalike
reference domains) plus the exact prediction code. Anything loaded with
``mlflow.pyfunc.load_model("models:/ml-url@champion")`` therefore predicts the
same way the FastAPI service does.

Input : a DataFrame with a ``url`` column (or a list of URL strings)
Output: a DataFrame with ``url``, ``label``, ``probability``, ``verdict`` and
        ``risk_signals`` (comma-separated, empty when none fired)
"""
import pickle

import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel

OUTPUT_COLUMNS = ["url", "label", "probability", "verdict", "risk_signals"]


class URLPhishingClassifier(PythonModel):
    def load_context(self, context):
        # Only load bundles from your own registry: pickle can execute code.
        with open(context.artifacts["bundle"], "rb") as f:
            artifact = pickle.load(f)
        self.model = artifact["model"]
        self.metadata = {k: v for k, v in artifact.items() if k != "model"}

    def predict(self, context, model_input, params=None):
        from Services.inference_service import predict_with_bundle

        if isinstance(model_input, pd.DataFrame):
            urls = model_input["url"].astype(str).tolist()
        else:
            urls = [str(u) for u in model_input]

        rows = []
        for url in urls:
            result, _ = predict_with_bundle(url, self.model, self.metadata)
            rows.append(
                {
                    "url": result["url"],
                    "label": int(result["label"]),
                    "probability": float(result["probability"]),
                    "verdict": result["verdict"],
                    "risk_signals": ",".join(result.get("risk_signals", [])),
                }
            )
        return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


mlflow.models.set_model(URLPhishingClassifier())
