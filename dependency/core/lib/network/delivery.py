import time
from pathlib import Path

from core.lib.common import LOGGER

from .client import http_request


_REQUEST_TIMEOUT_SECONDS = 5.0
_RETRY_INTERVAL_SECONDS = 0.5


def task_ack(task):
    """Return the exact ownership acknowledgement for one task delivery."""
    return {
        "accepted": True,
        "task_uuid": task.get_task_uuid(),
    }


def _is_matching_ack(response, task):
    return (
        isinstance(response, dict)
        and response.get("accepted") is True
        and response.get("task_uuid") == task.get_task_uuid()
    )


def deliver_task(
    *,
    url,
    method,
    task,
    file_path=None,
    file_content=None,
    persistent=False,
):
    """Transfer task ownership only after the receiver returns an exact ACK.

    Controller forwarding performs one attempt and propagates failure to the
    upstream owner. Generator and Processor use ``persistent=True`` so they
    retain their current task and apply backpressure until another component
    accepts it. File uploads reopen the immutable artifact for every attempt,
    replaying the complete payload without buffering a video in memory or
    reusing an exhausted file handle.
    """
    if file_path is not None and file_content is not None:
        raise ValueError("task delivery accepts either file_path or file_content, not both")

    file_name = Path(task.get_file_path() or file_path or "task.bin").name

    while True:
        request = {
            "url": url,
            "method": method,
            "timeout": _REQUEST_TIMEOUT_SECONDS,
            "retry": 1,
            "data": {"data": task.serialize()},
        }
        if file_path is not None:
            with Path(file_path).open("rb") as artifact:
                request["files"] = {
                    "file": (file_name, artifact, "application/octet-stream")
                }
                response = http_request(**request)
        else:
            if file_content is not None:
                request["files"] = {
                    "file": (file_name, file_content, "application/octet-stream")
                }
            response = http_request(**request)
        if _is_matching_ack(response, task):
            return True
        if not persistent:
            return False

        LOGGER.warning(
            f"[Task Delivery] Receiver did not acknowledge ownership; retrying. "
            f"source={task.get_source_id()} task={task.get_task_id()} "
            f"service={task.get_flow_index()} url={url}"
        )
        time.sleep(_RETRY_INTERVAL_SECONDS)
