import re
from typing import Union


_KUBERNETES_SERVICE_DNS_SUFFIX = ".svc.cluster.local"


def connection_host(host: str) -> str:
    """Return a connection-safe host without changing endpoint identity.

    Kubernetes Pods commonly inherit ``ndots:5``. A service name such as
    ``scheduler.ns.svc.cluster.local`` has only four dots and may therefore be
    tried against every DNS search suffix before its absolute lookup. A
    trailing dot makes the name absolute at the network boundary.

    IP addresses, localhost and external DNS names are deliberately left
    untouched. The function is idempotent so callers may safely apply it at
    every connection boundary.
    """

    value = str(host or "").strip()
    if not value or value.endswith("."):
        return value
    if value.lower().endswith(_KUBERNETES_SERVICE_DNS_SUFFIX):
        return f"{value}."
    return value


def merge_address(ip: str, protocol: str = 'http', port: Union[int, str] = None, path: str = None):
    """
    merge address from {protocol, ip, port, path}
    eg: http://127.0.0.1:9000/submit
    """

    path = None if path is None else str(path).replace('/', '')

    port_divider = '' if port is None else ':'
    path_divider = '' if path is None else '/'

    port = '' if port is None else port
    path = '' if path is None else path

    return f'{protocol}://{ip}{port_divider}{port}{path_divider}{path}'


def find_all_ips(text: str) -> list:
    """
    :param text: text of address
    :return: list of ips
    """
    ips = re.findall(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
                     text)

    return ips
