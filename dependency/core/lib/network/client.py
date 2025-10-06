from core.lib.common import LOGGER
import requests


def http_request(url,
                 method=None,
                 timeout=None,
                 binary=True,
                 no_decode=False,
                 **kwargs):
    _max_timeout = timeout if timeout else 1000
    _method = 'GET' if not method else method

    try:
        response = requests.request(method=_method, url=url, timeout=_max_timeout, **kwargs)
        if response.status_code == 200:
            if no_decode:
                return response
            else:
                return response.json() if binary else response.content.decode('utf-8')
        elif 200 < response.status_code < 400:
            LOGGER.info(f'Redirect URL: {response.url}')
        LOGGER.warning(f'Get invalid status code {response.status_code} in request {url}')
    except (ConnectionRefusedError, requests.exceptions.ConnectionError):
        LOGGER.warning(f'Connection refused in request {url}')
    except requests.exceptions.HTTPError as err:
        LOGGER.warning(f'Http Error in request {url}: {err}')
    except requests.exceptions.Timeout as err:
        LOGGER.warning(f'Timeout error in request {url}: {err}')
    except requests.exceptions.RequestException as err:
        LOGGER.warning(f'Error occurred in request {url}: {err}')
    except Exception as err:
        LOGGER.warning(f'Error occurred in request {url}: {err}')
