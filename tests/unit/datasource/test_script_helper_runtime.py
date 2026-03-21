import importlib
import signal
from types import SimpleNamespace

import pytest


script_helper_module = importlib.import_module("script_helper")


@pytest.mark.unit
def test_script_helper_starts_process_with_shell_and_process_group(monkeypatch):
    popen_calls = []

    def fake_popen(command, shell, preexec_fn):
        popen_calls.append((command, shell, preexec_fn))
        return SimpleNamespace(pid=321)

    monkeypatch.setattr(script_helper_module.subprocess, "Popen", fake_popen)

    process = script_helper_module.ScriptHelper.start_script("python demo.py")

    assert process.pid == 321
    assert popen_calls == [("python demo.py", True, script_helper_module.os.setsid)]


@pytest.mark.unit
def test_script_helper_stops_process_group_with_sigterm(monkeypatch):
    kill_calls = []

    monkeypatch.setattr(script_helper_module.os, "getpgid", lambda pid: pid + 100)
    monkeypatch.setattr(
        script_helper_module.os,
        "killpg",
        lambda process_group, sig: kill_calls.append((process_group, sig)),
    )

    script_helper_module.ScriptHelper.stop_script(SimpleNamespace(pid=55))

    assert kill_calls == [(155, signal.SIGTERM)]
