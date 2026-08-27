import importlib
from types import SimpleNamespace

import pytest


base_inference_module = importlib.import_module(
    "core.applications.model_switch_detection.inference_module.base_inference"
)


@pytest.mark.unit
def test_model_switch_queue_query_is_process_local(monkeypatch):
    calls = []
    monkeypatch.setattr(
        base_inference_module.Context,
        "get_parameter",
        staticmethod(lambda name: 9100 if name == "GUNICORN_PORT" else None),
    )
    monkeypatch.setattr(
        base_inference_module,
        "http_request",
        lambda url, method=None, timeout=None: calls.append((url, method, timeout)) or {
            "waiting_count": 3,
            "busy": True,
        },
    )

    instance = SimpleNamespace()
    assert base_inference_module.BaseInference.get_queue(instance) == 3
    assert calls == [("http://127.0.0.1:9100/queue_state", "GET", 5)]
    assert instance.processor_port == 9100
