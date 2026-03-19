import importlib

import pytest


generator_server_module = importlib.import_module("core.generator.generator_server")


@pytest.mark.unit
def test_generator_server_collects_context_parameters_and_runs_selected_generator(monkeypatch):
    calls = []

    class FakeGenerator:
        def run(self):
            calls.append(("run", None))

    values = {
        ("SOURCE_ID", False): 7,
        ("SOURCE_TYPE", True): "http_video",
        ("SOURCE_URL", True): "http://example.com/video.mp4",
        ("SOURCE_METADATA", False): {"fps": 25},
        ("DAG", False): {"detector": {"next_nodes": []}},
    }

    monkeypatch.setattr(
        generator_server_module.Context,
        "get_parameter",
        staticmethod(lambda name, direct=True: values[(name, direct)]),
    )
    monkeypatch.setattr(
        generator_server_module.Context,
        "get_algorithm",
        staticmethod(
            lambda algorithm, al_name=None, **kwargs: calls.append((algorithm, al_name, kwargs)) or FakeGenerator()
        ),
    )

    generator_server_module.GeneratorServer.run()

    assert calls == [
        (
            "GENERATOR",
            "http_video",
            {
                "source_id": 7,
                "source_url": "http://example.com/video.mp4",
                "source_metadata": {"fps": 25},
                "dag": {"detector": {"next_nodes": []}},
            },
        ),
        ("run", None),
    ]
