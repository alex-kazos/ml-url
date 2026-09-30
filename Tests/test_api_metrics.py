"""The API exposes the monitoring signals Prometheus scrapes."""
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import Services.api as api

client = TestClient(api.app)


def _value(name, labels=None):
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def _fake_predict(label, overridden=False, risk_signals=None):
    def _predict(url, model_name=None):
        result = {"url": url, "label": label, "probability": 0.85 if label else 0.1,
                  "verdict": "[!] SUSPICIOUS / PHISHING" if label else "[OK] LEGITIMATE"}
        if risk_signals:
            result["risk_signals"] = risk_signals
        return result, {"model_label": 0 if overridden else label, "overridden": overridden}
    return _predict


def test_predict_records_verdict_override_and_input_signals(monkeypatch):
    monkeypatch.setattr(
        api, "predict_url",
        _fake_predict(1, overridden=True, risk_signals=["domain-lookalike:paypa1->paypal.com"]),
    )
    before = {
        "phishing": _value("mlurl_predictions_total", {"verdict": "phishing"}),
        "overrides": _value("mlurl_lookalike_overrides_total"),
        "signals": _value("mlurl_risk_signals_total", {"signal_type": "domain-lookalike"}),
        "no_scheme": _value("mlurl_input_urls_total", {"has_scheme": "false"}),
        "latency": _value("mlurl_prediction_latency_seconds_count"),
    }

    response = client.post("/predict", json={"url": "paypa1.com"})

    assert response.status_code == 200
    assert response.json()["label"] == 1
    assert _value("mlurl_predictions_total", {"verdict": "phishing"}) == before["phishing"] + 1
    assert _value("mlurl_lookalike_overrides_total") == before["overrides"] + 1
    assert (
        _value("mlurl_risk_signals_total", {"signal_type": "domain-lookalike"})
        == before["signals"] + 1
    )
    assert _value("mlurl_input_urls_total", {"has_scheme": "false"}) == before["no_scheme"] + 1
    assert _value("mlurl_prediction_latency_seconds_count") == before["latency"] + 1


def test_public_response_shape_is_unchanged(monkeypatch):
    monkeypatch.setattr(api, "predict_url", _fake_predict(0))

    body = client.post("/predict", json={"url": "https://www.google.com"}).json()

    assert set(body) == {"url", "label", "probability", "verdict"}


def test_metrics_endpoint_requires_token_when_configured(monkeypatch):
    monkeypatch.setattr(api, "METRICS_TOKEN", "s3cret")

    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.get("/metrics", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    assert "mlurl_predictions_total" in ok.text


def test_metrics_endpoint_is_open_without_token(monkeypatch):
    monkeypatch.setattr(api, "METRICS_TOKEN", "")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "mlurl_model_info" in response.text


def test_health_reports_model_version():
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert "version" in body
