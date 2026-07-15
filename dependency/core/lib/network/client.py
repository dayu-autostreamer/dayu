import os
import threading
import time

import requests

from core.lib.common import LOGGER


DEFAULT_RETRY_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_HTTP_SESSION_LOCAL = threading.local()


def _get_http_session():
    """Return one connection pool per process and calling thread.

    ``requests.Session`` is not documented as thread-safe, while Dayu issues
    synchronous requests from both component loops and FastAPI worker threads.
    A process-aware thread-local session keeps connection reuse without sharing
    mutable request state across FastAPI threads or forked RTSP workers.
    """
    process_id = os.getpid()
    session = getattr(_HTTP_SESSION_LOCAL, "session", None)
    if session is None or getattr(_HTTP_SESSION_LOCAL, "process_id", None) != process_id:
        if session is not None:
            session.close()
        session = requests.Session()
        _HTTP_SESSION_LOCAL.session = session
        _HTTP_SESSION_LOCAL.process_id = process_id
    return session


def _request(**kwargs):
    return _get_http_session().request(**kwargs)


def _normalize_retry_count(retry):
    try:
        return max(1, int(retry))
    except (TypeError, ValueError):
        LOGGER.warning(f'Invalid retry count {retry}, fallback to 1')
        return 1


def _normalize_non_negative_float(value, default, name):
    try:
        return max(0, float(value))
    except (TypeError, ValueError):
        LOGGER.warning(f'Invalid {name} {value}, fallback to {default}')
        return default


def _normalize_retry_status_codes(retry_status_codes):
    if retry_status_codes is None:
        return DEFAULT_RETRY_STATUS_CODES
    try:
        if isinstance(retry_status_codes, int):
            return {retry_status_codes}
        if isinstance(retry_status_codes, str):
            return {
                int(status_code.strip())
                for status_code in retry_status_codes.split(',')
                if status_code.strip()
            }
        return {int(status_code) for status_code in retry_status_codes}
    except (TypeError, ValueError):
        LOGGER.warning(f'Invalid retry status codes {retry_status_codes}, fallback to defaults')
        return DEFAULT_RETRY_STATUS_CODES


def http_request(url,
                 method=None,
                 timeout=None,
                 binary=True,
                 no_decode=False,
                 retry=1,
                 retry_interval=0,
                 retry_backoff=1,
                 retry_status_codes=None,
                 cancel_event=None,
                 **kwargs):
    """
    Send an HTTP request and keep the previous return contract.

    retry is the maximum number of attempts. The default value 1 preserves the
    previous behavior. Retries are applied to transport exceptions and transient
    HTTP status codes such as 408, 429 and 5xx. ``cancel_event`` is cooperative:
    it prevents a new attempt and interrupts retry backoff, while the timeout of
    an already-running synchronous request remains its hard cancellation bound.
    """
    _max_timeout = timeout if timeout else 1000
    _method = 'GET' if not method else method
    _retry = _normalize_retry_count(retry)
    _retry_interval = _normalize_non_negative_float(retry_interval, default=0, name='retry interval')
    _retry_backoff = max(1, _normalize_non_negative_float(retry_backoff, default=1, name='retry backoff'))
    _retry_status_codes = _normalize_retry_status_codes(retry_status_codes)

    for attempt in range(1, _retry + 1):
        if cancel_event is not None and cancel_event.is_set():
            return None
        should_retry = False
        try:
            response = _request(method=_method, url=url, timeout=_max_timeout, **kwargs)
            if response.status_code == 200:
                if no_decode:
                    return response
                return response.json() if binary else response.content.decode('utf-8')

            if 200 < response.status_code < 400:
                LOGGER.info(f'Redirect URL: {response.url}')
            LOGGER.warning(f'Get invalid status code {response.status_code} in request {url}')
            should_retry = response.status_code in _retry_status_codes
        except (ConnectionRefusedError, requests.exceptions.ConnectionError):
            LOGGER.warning(f'Connection refused in request {url}')
            should_retry = True
        except requests.exceptions.HTTPError as err:
            LOGGER.warning(f'Http Error in request {url}: {err}')
            should_retry = True
        except requests.exceptions.Timeout as err:
            LOGGER.warning(f'Timeout error in request {url}: {err}')
            should_retry = True
        except requests.exceptions.RequestException as err:
            LOGGER.warning(f'Error occurred in request {url}: {err}')
            should_retry = True
        except Exception as err:
            LOGGER.warning(f'Error occurred in request {url}: {err}')
            should_retry = True

        if not should_retry or attempt >= _retry:
            return None

        LOGGER.warning(f'Retry request {url}, attempt {attempt + 1}/{_retry}')
        if _retry_interval > 0:
            if cancel_event is not None:
                if cancel_event.wait(_retry_interval):
                    return None
            else:
                time.sleep(_retry_interval)
            _retry_interval *= _retry_backoff

    return None
