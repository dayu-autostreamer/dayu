"""Scheduler-backed task leases for bounded RuntimeDirectory retirement.

The data plane never discovers Kubernetes resources.  A task carries the
immutable RuntimeDirectory revision that routed it, and all lease operations
are sent to the scheduler endpoint injected in ``DAYU_RUNTIME_BOOTSTRAP``.
"""

import json
import math
import threading
import time
from contextlib import contextmanager
from typing import Callable, Tuple

from core.lib.network import NetworkAPIMethod, NetworkAPIPath, http_request

from .context import RuntimeContext


class RuntimeLeaseError(RuntimeError):
    """Base class for lease operations that cannot be proven successful."""


class RuntimeLeaseIdentityError(RuntimeLeaseError):
    """The task or scheduler response does not contain an exact lease key."""


class RuntimeLeaseUnavailable(RuntimeLeaseError):
    """The scheduler did not confirm the requested lease operation."""


class RuntimeLeaseRetired(RuntimeLeaseError):
    """The task's immutable RuntimeDirectory revision has been fenced."""

    def __init__(self, revision, deadline=None):
        message = f"runtime directory revision {revision} is retired"
        if deadline is not None:
            message += f" at deadline {deadline}"
        super().__init__(message)
        self.revision = revision
        self.deadline = deadline


class RuntimeLeaseClient:
    """Strict client for the scheduler task-lease API.

    ``None`` or a malformed response is always an error.  This fail-closed
    contract is important during a RuntimeDirectory switch: without a
    confirmed lease, old RuntimeServices may be retired while the task is
    still running.
    """

    def __init__(
        self,
        runtime_context=None,
        requester: Callable = None,
        timeout_seconds: float = 5.0,
        clock: Callable = None,
    ):
        self.runtime_context = runtime_context or RuntimeContext.get_default()
        self.requester = requester or http_request
        try:
            timeout_seconds = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("task lease request timeout must be numeric") from exc
        if timeout_seconds <= 0:
            raise ValueError("task lease request timeout must be positive")
        self.timeout_seconds = timeout_seconds
        self.clock = clock or time.monotonic

    @property
    def ttl_seconds(self):
        return self.runtime_context.lease_ttl_seconds

    def _url(self):
        endpoint = self.runtime_context.resolve_static_endpoint("scheduler")
        return endpoint.url(NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY_TASK_LEASES)

    @staticmethod
    def _task_key(task) -> Tuple[int, str]:
        if task is None:
            raise RuntimeLeaseIdentityError("task is required for a runtime lease")
        revision_getter = getattr(task, "get_runtime_directory_revision", None)
        root_uuid_getter = getattr(task, "get_root_uuid", None)
        if not callable(revision_getter) or not callable(root_uuid_getter):
            raise RuntimeLeaseIdentityError(
                "task must expose runtime_directory_revision and root_uuid"
            )
        try:
            revision = int(revision_getter())
        except (TypeError, ValueError) as exc:
            raise RuntimeLeaseIdentityError(
                "task runtime_directory_revision must be an integer"
            ) from exc
        root_uuid = str(root_uuid_getter() or "").strip()
        if revision < 1:
            raise RuntimeLeaseIdentityError(
                "task runtime_directory_revision must be positive"
            )
        if not root_uuid:
            raise RuntimeLeaseIdentityError("task root_uuid is required")
        return revision, root_uuid

    def _operate(self, task, method, operation, include_ttl):
        revision, root_uuid = self._task_key(task)
        payload = {
            "revision": revision,
            "root_uuid": root_uuid,
        }
        if include_ttl:
            payload["ttl_seconds"] = self.ttl_seconds
        try:
            response = self.requester(
                url=self._url(),
                method=method,
                timeout=self.timeout_seconds,
                retry=1,
                data={"data": json.dumps(payload)},
            )
        except Exception as exc:
            raise RuntimeLeaseUnavailable(
                "runtime task lease {} request failed: {}".format(operation, exc)
            ) from exc
        if not isinstance(response, dict):
            raise RuntimeLeaseUnavailable(
                "runtime task lease {} was not confirmed by scheduler".format(operation)
            )
        try:
            response_revision = int(response.get("revision"))
        except (TypeError, ValueError) as exc:
            raise RuntimeLeaseUnavailable(
                "runtime task lease {} response has no valid revision".format(operation)
            ) from exc
        response_root_uuid = str(response.get("root_uuid") or "")
        if response_revision != revision or response_root_uuid != root_uuid:
            raise RuntimeLeaseUnavailable(
                "runtime task lease {} response identity mismatch".format(operation)
            )
        if response.get("retired") is True:
            raise RuntimeLeaseRetired(revision, response.get("deadline"))
        if operation == "release" and response.get("released") is not True:
            raise RuntimeLeaseUnavailable(
                "runtime task lease release response is not acknowledged"
            )
        if operation != "release":
            try:
                expires_at = float(response["expires_at"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeLeaseUnavailable(
                    "runtime task lease {} response has no valid expiry".format(
                        operation
                    )
                ) from exc
            try:
                valid_for_seconds = float(response["valid_for_seconds"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeLeaseUnavailable(
                    "runtime task lease {} response has no valid relative lifetime".format(
                        operation
                    )
                ) from exc
            if not math.isfinite(expires_at) or expires_at <= 0:
                raise RuntimeLeaseUnavailable(
                    "runtime task lease {} response has no valid expiry".format(
                        operation
                    )
                )
            if (
                    not math.isfinite(valid_for_seconds)
                    or valid_for_seconds <= 0
                    or valid_for_seconds > self.ttl_seconds
            ):
                raise RuntimeLeaseUnavailable(
                    "runtime task lease {} response has invalid relative lifetime".format(
                        operation
                    )
                )
            response = dict(response)
            response["expires_at"] = expires_at
            response["valid_for_seconds"] = valid_for_seconds
        return response

    def acquire(self, task):
        return self._operate(
            task,
            NetworkAPIMethod.SCHEDULER_ACQUIRE_TASK_LEASE,
            "acquire",
            include_ttl=True,
        )

    def renew(self, task):
        return self._operate(
            task,
            NetworkAPIMethod.SCHEDULER_RENEW_TASK_LEASE,
            "renew",
            include_ttl=True,
        )

    def release(self, task):
        return self._operate(
            task,
            NetworkAPIMethod.SCHEDULER_RELEASE_TASK_LEASE,
            "release",
            include_ttl=False,
        )

    @contextmanager
    def keepalive(self, task):
        """Renew a lease throughout one potentially long-running operation.

        Component-boundary renewals protect normal forwarding, but inference
        may legitimately run longer than a configured TTL.  A daemon heartbeat
        renews at most every 30 seconds. Transient failures are retried only
        while the last Scheduler-confirmed expiry remains valid. A retirement
        fence is terminal and fails closed when the operation returns.
        """

        initial = self.renew(task)
        stop = threading.Event()
        now = self.clock()
        state = {
            "valid_until": now + initial["valid_for_seconds"],
            "lost": None,
        }
        interval = max(
            0.05,
            min(30.0, initial["valid_for_seconds"] / 3.0),
        )

        def heartbeat():
            while not stop.wait(interval):
                try:
                    response = self.renew(task)
                    state["valid_until"] = (
                        self.clock() + response["valid_for_seconds"]
                    )
                    state["lost"] = None
                except RuntimeLeaseRetired as exc:
                    state["lost"] = exc
                    state["valid_until"] = self.clock()
                    return
                except RuntimeLeaseError as exc:
                    state["lost"] = exc
                    if self.clock() >= state["valid_until"]:
                        return

        thread = threading.Thread(
            target=heartbeat,
            name="dayu-runtime-lease-keepalive",
            daemon=True,
        )
        thread.start()
        body_error = None
        try:
            yield
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            stop.set()
            thread.join(timeout=max(1.0, min(self.timeout_seconds + 1.0, 10.0)))
            if (
                    body_error is None
                    and self.clock() >= state["valid_until"]
            ):
                if isinstance(state["lost"], RuntimeLeaseRetired):
                    raise state["lost"]
                raise RuntimeLeaseUnavailable(
                    "runtime task lease expired during a long-running operation"
                ) from state["lost"]
