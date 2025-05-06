import re
from typing import Union


def merge_address(ip: str, protocol: str = 'http', port: Union[int, str] = None, path: str = None):
    """
    merge address from {protocol, ip, port, path}
    eg: http://127.0.0.1:9000/submit
    """

    path = path.replace('/', '')

    port_divider = '' if port is None else ':'
    path_divider = '' if path is None else '/'

    port = '' if port is None else port
    path = '' if path is None else path

    return f'{protocol}://{ip}{port_divider}{port}{path_divider}{path}'


def find_all_ips(text: str) -> list:
    """
    :param text: 文本
    :return: 返回ip列表
    """
    ips = re.findall(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
                     text)

    return ips
