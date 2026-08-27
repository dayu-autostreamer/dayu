import importlib
import threading

import pytest


client_module = importlib.import_module("core.lib.network.client")
utils_module = importlib.import_module("core.lib.network.utils")

http_request = client_module.http_request
connection_host = utils_module.connection_host
find_all_ips = utils_module.find_all_ips
merge_address = utils_module.merge_address


class FakeResponse:
    def __init__(self, status_code, payload=None, content=b"", text="", reason=None):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.text = text
        self.reason = reason
        self.url = "http://scheduler"

    def json(self):
        return self._payload


@pytest.mark.unit
def test_http_request_handles_http_error_request_error_and_generic_error(monkeypatch):
    monkeypatch.setattr(
        client_module,
        "_request",
        lambda **kwargs: (_ for _ in ()).throw(client_module.requests.exceptions.HTTPError("bad-request")),
    )
    assert http_request("http://scheduler") is None

    monkeypatch.setattr(
        client_module,
        "_request",
        lambda **kwargs: (_ for _ in ()).throw(client_module.requests.exceptions.RequestException("broken")),
    )
    assert http_request("http://scheduler") is None

    monkeypatch.setattr(
        client_module,
        "_request",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )
    assert http_request("http://scheduler") is None


@pytest.mark.unit
def test_http_request_cooperative_cancellation_skips_attempts_and_retry_backoff(monkeypatch):
    request_calls = []
    monkeypatch.setattr(
        client_module,
        "_request",
        lambda **kwargs: request_calls.append(kwargs) or FakeResponse(503),
    )

    cancelled = threading.Event()
    cancelled.set()
    assert http_request("http://scheduler", retry=3, cancel_event=cancelled) is None
    assert request_calls == []

    class CancelDuringBackoff:
        def __init__(self):
            self.cancelled = False
            self.waits = []

        def is_set(self):
            return self.cancelled

        def wait(self, timeout):
            self.waits.append(timeout)
            self.cancelled = True
            return True

    token = CancelDuringBackoff()
    assert http_request(
        "http://scheduler",
        retry=3,
        retry_interval=0.25,
        cancel_event=token,
    ) is None
    assert len(request_calls) == 1
    assert token.waits == [0.25]


@pytest.mark.unit
def test_http_request_without_cancellation_token_keeps_existing_contract(monkeypatch):
    request_calls = []
    monkeypatch.setattr(
        client_module,
        "_request",
        lambda **kwargs: request_calls.append(kwargs) or FakeResponse(
            200, {"ok": True},
        ),
    )

    assert http_request("http://scheduler") == {"ok": True}
    assert len(request_calls) == 1


@pytest.mark.unit
def test_http_request_or_raise_preserves_fastapi_error_detail(monkeypatch):
    request_calls = []
    monkeypatch.setattr(
        client_module,
        "_request",
        lambda **kwargs: request_calls.append(kwargs) or FakeResponse(
            422,
            {
                "detail": (
                    "deployment policy for service 'vehicle-trajectory-prediction' "
                    "selected non-candidate nodes: ['edgexn32']"
                )
            },
        ),
    )

    with pytest.raises(client_module.HTTPClientError) as exc_info:
        client_module.http_request_or_raise(
            "http://scheduler/initial_deployment",
            method="post",
            retry=3,
        )

    error = exc_info.value
    assert error.method == "POST"
    assert error.url == "http://scheduler/initial_deployment"
    assert error.status_code == 422
    assert error.detail == (
        "deployment policy for service 'vehicle-trajectory-prediction' "
        "selected non-candidate nodes: ['edgexn32']"
    )
    assert "HTTP 422" in str(error)
    assert len(request_calls) == 1


@pytest.mark.unit
def test_http_request_or_raise_sanitizes_structured_validation_detail(monkeypatch):
    validation_error = {
        "type": "missing",
        "loc": ["body", "node_set"],
        "msg": "Field required",
        "input": "x" * 4096,
    }
    monkeypatch.setattr(
        client_module,
        "_request",
        lambda **kwargs: FakeResponse(422, {"detail": [validation_error]}),
    )

    with pytest.raises(client_module.HTTPClientError) as exc_info:
        client_module.http_request_or_raise("http://scheduler/initial_deployment")

    detail = exc_info.value.detail
    assert '"loc":["body","node_set"]' in detail
    assert '"msg":"Field required"' in detail
    assert '"type":"missing"' in detail
    assert '"input"' not in detail
    assert "x" * 32 not in detail


@pytest.mark.unit
def test_http_request_or_raise_bounds_unstructured_error_detail(monkeypatch):
    monkeypatch.setattr(
        client_module,
        "_request",
        lambda **kwargs: FakeResponse(422, {"detail": "x" * 4096}),
    )

    with pytest.raises(client_module.HTTPClientError) as exc_info:
        client_module.http_request_or_raise("http://scheduler/initial_deployment")

    assert len(exc_info.value.detail) == client_module._MAX_ERROR_DETAIL_LENGTH
    assert exc_info.value.detail.endswith("...")


@pytest.mark.unit
def test_http_request_or_raise_retains_last_transient_failure(monkeypatch):
    responses = iter([
        FakeResponse(503, {"detail": "scheduler is starting"}),
        FakeResponse(503, {"detail": "scheduler policy is unavailable"}),
    ])
    request_calls = []
    monkeypatch.setattr(
        client_module,
        "_request",
        lambda **kwargs: request_calls.append(kwargs) or next(responses),
    )

    with pytest.raises(client_module.HTTPClientError) as exc_info:
        client_module.http_request_or_raise("http://scheduler", retry=2)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "scheduler policy is unavailable"
    assert len(request_calls) == 2


@pytest.mark.unit
def test_http_request_or_raise_keeps_cooperative_cancellation_semantics(monkeypatch):
    request_calls = []
    monkeypatch.setattr(
        client_module,
        "_request",
        lambda **kwargs: request_calls.append(kwargs) or FakeResponse(200, {"ok": True}),
    )
    cancelled = threading.Event()
    cancelled.set()

    assert client_module.http_request_or_raise(
        "http://scheduler",
        retry=3,
        cancel_event=cancelled,
    ) is None
    assert request_calls == []


@pytest.mark.unit
def test_http_request_reuses_one_session_per_process_thread(monkeypatch):
    created_sessions = []
    closed_sessions = []

    class FakeSession:
        def close(self):
            closed_sessions.append(self)

    def create_session():
        session = FakeSession()
        created_sessions.append(session)
        return session

    monkeypatch.setattr(client_module.requests, "Session", create_session)
    monkeypatch.setattr(client_module, "_HTTP_SESSION_LOCAL", threading.local())

    first = client_module._get_http_session()
    second = client_module._get_http_session()
    other_thread_sessions = []
    thread = threading.Thread(
        target=lambda: other_thread_sessions.append(client_module._get_http_session())
    )
    thread.start()
    thread.join()

    assert first is second
    assert other_thread_sessions[0] is not first
    assert created_sessions == [first, other_thread_sessions[0]]

    monkeypatch.setattr(client_module.os, "getpid", lambda: 999999)
    after_fork = client_module._get_http_session()

    assert after_fork is not first
    assert closed_sessions == [first]


@pytest.mark.unit
def test_network_utils_merge_address_and_find_all_ips_cover_optional_path_and_multiple_ips():
    assert merge_address("10.0.0.8", port=9000, path="/health") == "http://10.0.0.8:9000/health"
    assert merge_address("10.0.0.8", protocol="https", port=None, path=None) == "https://10.0.0.8"
    assert find_all_ips("edge=10.0.0.8 cloud=192.168.1.2") == ["10.0.0.8", "192.168.1.2"]
    assert find_all_ips("invalid 256.0.0.1") == []


@pytest.mark.unit
@pytest.mark.parametrize(("value", "expected"), (
    ("scheduler.dayu.svc.cluster.local", "scheduler.dayu.svc.cluster.local."),
    ("Scheduler.Dayu.SVC.CLUSTER.LOCAL", "Scheduler.Dayu.SVC.CLUSTER.LOCAL."),
    ("scheduler.dayu.svc.cluster.local.", "scheduler.dayu.svc.cluster.local."),
    ("10.0.0.1", "10.0.0.1"),
    ("localhost", "localhost"),
    ("api.example.com", "api.example.com"),
    ("", ""),
))
def test_connection_host_only_absolutizes_kubernetes_service_fqdns(value, expected):
    assert connection_host(value) == expected
