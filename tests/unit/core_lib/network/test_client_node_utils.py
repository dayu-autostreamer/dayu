import importlib

import pytest


client_module = importlib.import_module("core.lib.network.client")
utils_module = importlib.import_module("core.lib.network.utils")

http_request = client_module.http_request
find_all_ips = utils_module.find_all_ips
merge_address = utils_module.merge_address


@pytest.mark.unit
def test_http_request_handles_http_error_request_error_and_generic_error(monkeypatch):
    monkeypatch.setattr(
        client_module.requests,
        "request",
        lambda **kwargs: (_ for _ in ()).throw(client_module.requests.exceptions.HTTPError("bad-request")),
    )
    assert http_request("http://scheduler") is None

    monkeypatch.setattr(
        client_module.requests,
        "request",
        lambda **kwargs: (_ for _ in ()).throw(client_module.requests.exceptions.RequestException("broken")),
    )
    assert http_request("http://scheduler") is None

    monkeypatch.setattr(
        client_module.requests,
        "request",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )
    assert http_request("http://scheduler") is None


@pytest.mark.unit
def test_network_utils_merge_address_and_find_all_ips_cover_optional_path_and_multiple_ips():
    assert merge_address("10.0.0.8", port=9000, path="/health") == "http://10.0.0.8:9000/health"
    assert merge_address("10.0.0.8", protocol="https", port=None, path=None) == "https://10.0.0.8"
    assert find_all_ips("edge=10.0.0.8 cloud=192.168.1.2") == ["10.0.0.8", "192.168.1.2"]
    assert find_all_ips("invalid 256.0.0.1") == []
