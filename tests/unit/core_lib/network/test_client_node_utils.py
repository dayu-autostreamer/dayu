import importlib
from types import SimpleNamespace

import pytest


client_module = importlib.import_module("core.lib.network.client")
node_module = importlib.import_module("core.lib.network.node")
utils_module = importlib.import_module("core.lib.network.utils")

NodeInfo = node_module.NodeInfo
http_request = client_module.http_request
find_all_ips = utils_module.find_all_ips
merge_address = utils_module.merge_address


@pytest.fixture(autouse=True)
def reset_node_cache():
    NodeInfo._NodeInfo__node_info_hostname = None
    NodeInfo._NodeInfo__node_info_ip = None
    NodeInfo._NodeInfo__node_info_role = None
    yield
    NodeInfo._NodeInfo__node_info_hostname = None
    NodeInfo._NodeInfo__node_info_ip = None
    NodeInfo._NodeInfo__node_info_role = None


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


@pytest.mark.unit
def test_node_info_reports_lookup_errors_and_missing_cluster_state(monkeypatch):
    NodeInfo._NodeInfo__node_info_hostname = {"edge-a": "10.0.0.2"}
    NodeInfo._NodeInfo__node_info_ip = {"10.0.0.2": "edge-a"}
    NodeInfo._NodeInfo__node_info_role = {"edge-a": "edge"}

    with pytest.raises(AssertionError, match='Hostname "cloud-a" not exists'):
        NodeInfo.hostname2ip("cloud-a")

    with pytest.raises(AssertionError, match='Ip "10.0.0.9" not exists'):
        NodeInfo.ip2hostname("10.0.0.9")

    with pytest.raises(AssertionError, match='contains none or more than one legal ip'):
        NodeInfo.url2hostname("http://no-ip-here")

    with pytest.raises(AssertionError, match='Hostname "cloud-a" not exists'):
        NodeInfo.get_node_role("cloud-a")

    with pytest.raises(Exception, match="No cloud node identified"):
        NodeInfo.get_cloud_node()

    monkeypatch.setattr(node_module.Context, "get_parameter", staticmethod(lambda key: None))
    with pytest.raises(AssertionError, match='Node Config is not found'):
        NodeInfo.get_local_device()


@pytest.mark.unit
def test_node_info_extracts_cluster_metadata_and_validates_non_empty_nodes(monkeypatch):
    nodes = [
        SimpleNamespace(
            metadata=SimpleNamespace(
                name="cloud-a",
                labels={"node-role.kubernetes.io/master": ""},
            ),
            status=SimpleNamespace(
                addresses=[SimpleNamespace(type="InternalIP", address="10.0.0.1")]
            ),
        ),
        SimpleNamespace(
            metadata=SimpleNamespace(
                name="edge-a",
                labels={"node-role.kubernetes.io/edge": ""},
            ),
            status=SimpleNamespace(
                addresses=[SimpleNamespace(type="InternalIP", address="10.0.0.2")]
            ),
        ),
    ]

    monkeypatch.setattr(node_module.config, "load_incluster_config", lambda: None)
    monkeypatch.setattr(
        node_module.client,
        "CoreV1Api",
        lambda: SimpleNamespace(list_node=lambda: SimpleNamespace(items=nodes)),
    )

    hostname_map, reverse_map, role_map = NodeInfo._NodeInfo__extract_node_info()

    assert hostname_map == {"cloud-a": "10.0.0.1", "edge-a": "10.0.0.2"}
    assert reverse_map == {"10.0.0.1": "cloud-a", "10.0.0.2": "edge-a"}
    assert role_map == {"cloud-a": "cloud", "edge-a": "edge"}

    monkeypatch.setattr(
        node_module.client,
        "CoreV1Api",
        lambda: SimpleNamespace(list_node=lambda: SimpleNamespace(items=[])),
    )
    with pytest.raises(AssertionError, match="Invalid node config"):
        NodeInfo._NodeInfo__extract_node_info()


@pytest.mark.unit
def test_node_info_refreshes_missing_cached_maps_and_skips_non_internal_addresses(monkeypatch):
    extracted = []

    def fake_extract():
        extracted.append("called")
        return (
            {"edge-a": "10.0.0.2"},
            {"10.0.0.2": "edge-a"},
            {"edge-a": "edge"},
        )

    NodeInfo._NodeInfo__node_info_hostname = {"stale": "10.0.0.9"}
    NodeInfo._NodeInfo__node_info_ip = None
    NodeInfo._NodeInfo__node_info_role = None
    monkeypatch.setattr(NodeInfo, "_NodeInfo__extract_node_info", staticmethod(fake_extract))

    assert NodeInfo.get_node_info_reverse() == {"10.0.0.2": "edge-a"}
    assert NodeInfo.get_node_info_role() == {"edge-a": "edge"}
    assert extracted == ["called"]

    nodes = [
        SimpleNamespace(
            metadata=SimpleNamespace(name="edge-a", labels={"node-role.kubernetes.io/edge": ""}),
            status=SimpleNamespace(
                addresses=[
                    SimpleNamespace(type="ExternalIP", address="1.1.1.1"),
                    SimpleNamespace(type="InternalIP", address="10.0.0.2"),
                ]
            ),
        )
    ]
    monkeypatch.setattr(node_module.config, "load_incluster_config", lambda: None)
    monkeypatch.setattr(
        node_module.client,
        "CoreV1Api",
        lambda: SimpleNamespace(list_node=lambda: SimpleNamespace(items=nodes)),
    )

    hostname_map, reverse_map, role_map = NodeInfo._NodeInfo__extract_node_info()
    assert hostname_map == {"edge-a": "10.0.0.2"}
    assert reverse_map == {"10.0.0.2": "edge-a"}
    assert role_map == {"edge-a": "edge"}
