import importlib

import pytest


def build_source_list():
    return [
        {"dir": "camera-a", "url": "http://10.0.0.2:32000/stream-0"},
        {"dir": "camera-b", "url": "http://10.0.0.3:32001/stream-1"},
    ]


def configure_runtime(monkeypatch, module, tmp_path):
    monkeypatch.setenv("PLAY_MODE", "cycle")
    monkeypatch.setenv("REQUEST_INTERVAL", "1")
    monkeypatch.setenv("START_INTERVAL", "0")
    monkeypatch.setenv("GUNICORN_PORT", "19010")
    monkeypatch.setattr(module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-node"))
    monkeypatch.setattr(module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: hostname))
    monkeypatch.setattr(module.PortInfo, "get_component_port", staticmethod(lambda component: 9000))
    monkeypatch.setattr(module.Context, "get_file_path", staticmethod(lambda modal: str(tmp_path / modal)))


@pytest.mark.unit
def test_open_datasource_rewrites_source_urls_and_tracks_processes(monkeypatch, tmp_path):
    datasource_server_module = importlib.import_module("datasource_server")
    configure_runtime(monkeypatch, datasource_server_module, tmp_path)

    for source in build_source_list():
        (tmp_path / "video" / source["dir"] / "http_video").mkdir(parents=True, exist_ok=True)

    started_commands = []
    sleep_calls = []
    monkeypatch.setattr(
        datasource_server_module.ScriptHelper,
        "start_script",
        staticmethod(lambda command: started_commands.append(command) or f"process-{len(started_commands)}"),
    )
    monkeypatch.setattr(datasource_server_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    datasource = datasource_server_module.DataSource()
    datasource.open_datasource("video", "demo-source", "http_video", build_source_list())

    assert datasource.source_open is True
    assert datasource.source_label == "demo-source"
    assert datasource.process_list == ["process-1", "process-2"]
    assert len(started_commands) == 2
    assert "--address http://127.0.0.1:19010/stream-0" in started_commands[0]
    assert "--address http://127.0.0.1:19010/stream-1" in started_commands[1]
    assert str(tmp_path / "video" / "camera-a" / "http_video") in started_commands[0]
    assert str(tmp_path / "video" / "camera-b" / "http_video") in started_commands[1]
    assert sleep_calls == [0]


@pytest.mark.unit
def test_close_datasource_stops_all_started_processes(monkeypatch, tmp_path):
    datasource_server_module = importlib.import_module("datasource_server")
    configure_runtime(monkeypatch, datasource_server_module, tmp_path)

    stopped_processes = []
    monkeypatch.setattr(
        datasource_server_module.ScriptHelper,
        "stop_script",
        staticmethod(lambda process: stopped_processes.append(process)),
    )

    datasource = datasource_server_module.DataSource()
    datasource.source_open = True
    datasource.source_label = "demo-source"
    datasource.process_list = ["process-1", "process-2"]

    datasource.close_datasource()

    assert stopped_processes == ["process-1", "process-2"]
    assert datasource.source_open is False
    assert datasource.source_label == ""
    assert datasource.process_list == []


@pytest.mark.unit
def test_open_datasource_ignores_unknown_modes_and_missing_directories(monkeypatch, tmp_path):
    datasource_server_module = importlib.import_module("datasource_server")
    configure_runtime(monkeypatch, datasource_server_module, tmp_path)

    started_commands = []
    monkeypatch.setattr(
        datasource_server_module.ScriptHelper,
        "start_script",
        staticmethod(lambda command: started_commands.append(command) or "process-1"),
    )

    datasource = datasource_server_module.DataSource()
    datasource.open_datasource("video", "demo-source", "invalid-mode", build_source_list())
    datasource.open_datasource("video", "demo-source", "http_video", build_source_list())

    assert started_commands == []
    assert datasource.source_open is False
