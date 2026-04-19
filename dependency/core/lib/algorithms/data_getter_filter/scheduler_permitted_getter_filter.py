import abc
import json
import time

from .base_getter_filter import BaseDataGetterFilter

from core.lib.common import ClassFactory, ClassType, LOGGER, SystemConstant
from core.lib.network import (
    NetworkAPIPath,
    NetworkAPIMethod,
    NodeInfo,
    PortInfo,
    merge_address,
    http_request,
)

__all__ = ('SchedulerPermittedDataGetterFilter',)


@ClassFactory.register(ClassType.GEN_GETTER_FILTER, alias='scheduler_permitted')
class SchedulerPermittedDataGetterFilter(BaseDataGetterFilter, abc.ABC):
    """
    Ask the scheduler whether this generator should fetch and submit data now.

    The filter is intentionally scheduler-agent agnostic. The scheduler exposes
    a generic generation-admission endpoint, and the active scheduler agent can
    override `should_generate` to apply algorithm-specific backpressure.
    """

    def __init__(self, fail_open: bool = True, timeout_s: float = 1.0,
                 log_interval_s: float = 10.0, action_retry_interval_s: float = 5.0):
        self.fail_open = bool(fail_open)
        self.timeout_s = max(0.1, float(timeout_s))
        self.log_interval_s = max(1.0, float(log_interval_s))
        self.action_retry_interval_s = max(0.0, float(action_retry_interval_s))
        self._last_block_log_t = 0.0
        self._last_error_log_t = 0.0
        self._completed_action_targets = set()
        self._action_attempt_timestamps = {}

        scheduler_hostname = NodeInfo.get_cloud_node()
        scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
        self.controller_port = PortInfo.get_component_port(SystemConstant.CONTROLLER.value)
        self.scheduler_address = merge_address(
            NodeInfo.hostname2ip(scheduler_hostname),
            port=scheduler_port,
            path=NetworkAPIPath.SCHEDULER_GENERATION_ADMISSION,
        )

    def _log_throttled(self, message: str, *, is_error: bool = False):
        now = time.time()
        last_t = self._last_error_log_t if is_error else self._last_block_log_t
        if now - last_t < self.log_interval_s:
            return
        if is_error:
            self._last_error_log_t = now
            LOGGER.warning(message)
        else:
            self._last_block_log_t = now
            LOGGER.info(message)

    def _remember_completed_action_target(self, key):
        completed = getattr(self, "_completed_action_targets", None)
        if completed is None:
            completed = set()
            self._completed_action_targets = completed
        completed.add(key)
        if len(completed) > 256:
            for stale_key in list(completed)[:128]:
                completed.discard(stale_key)

    def _mark_action_attempt(self, key) -> bool:
        now = time.time()
        attempts = getattr(self, "_action_attempt_timestamps", None)
        if attempts is None:
            attempts = {}
            self._action_attempt_timestamps = attempts

        last_t = float(attempts.get(key, 0.0))
        if last_t > 0.0 and now - last_t < self.action_retry_interval_s:
            return False
        attempts[key] = now
        if len(attempts) > 256:
            stale_keys = sorted(attempts, key=attempts.get)[:128]
            for stale_key in stale_keys:
                attempts.pop(stale_key, None)
        return True

    def _clear_processor_queues_from_action(self, action: dict):
        command_id = str(action.get("command_id") or "")
        target_devices = action.get("target_devices") or action.get("devices") or []
        if isinstance(target_devices, str):
            target_devices = [target_devices]
        if not isinstance(target_devices, (list, tuple, set)):
            return

        request = action.get("request") or {}
        if not isinstance(request, dict):
            request = {}
        timeout_s = request.get("timeout_s", self.timeout_s)
        try:
            timeout_s = max(0.1, float(timeout_s))
        except (TypeError, ValueError):
            timeout_s = self.timeout_s

        for device in target_devices:
            device = str(device)
            dedupe_key = f"{command_id}:{device}" if command_id else f"clear_processor_queues:{device}"
            if dedupe_key in getattr(self, "_completed_action_targets", set()):
                continue
            if not self._mark_action_attempt(dedupe_key):
                continue

            try:
                controller_port = getattr(self, "controller_port", None)
                if controller_port is None:
                    controller_port = PortInfo.get_component_port(SystemConstant.CONTROLLER.value)
                controller_address = merge_address(
                    NodeInfo.hostname2ip(device),
                    port=controller_port,
                    path=NetworkAPIPath.CONTROLLER_CLEAR_PROCESSOR_QUEUES,
                )
                response = http_request(
                    url=controller_address,
                    method=NetworkAPIMethod.CONTROLLER_CLEAR_PROCESSOR_QUEUES,
                    timeout=timeout_s,
                    data={"data": json.dumps(request)},
                )
            except Exception as exc:
                self._log_throttled(
                    f"[Getter Filter] Failed to execute scheduler action {command_id} "
                    f"on controller {device}: {exc}",
                    is_error=True,
                )
                continue

            if not isinstance(response, dict) or response.get("ok") is False:
                self._log_throttled(
                    f"[Getter Filter] Scheduler action {command_id} on controller {device} failed: "
                    f"response={response}",
                    is_error=True,
                )
                continue

            self._remember_completed_action_target(dedupe_key)
            LOGGER.warning(
                f"[Getter Filter] Executed scheduler action {command_id} on controller {device}: "
                f"cleared={response.get('cleared_count')}, matched={response.get('matched_count')}, "
                f"remaining={response.get('remaining_count')}."
            )

    def _execute_scheduler_actions(self, response: dict):
        actions = response.get("actions") or response.get("commands") or []
        if isinstance(actions, dict):
            actions = [actions]
        if not isinstance(actions, list):
            return

        for action in actions:
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("type") or "").strip().lower()
            if action_type == "clear_processor_queues":
                self._clear_processor_queues_from_action(action)

    def __call__(self, system):
        payload = {
            "source_id": system.source_id,
            "source_device": system.local_device,
            "meta_data": system.meta_data,
            "raw_meta_data": system.raw_meta_data,
            "completed_action_targets": list(getattr(self, "_completed_action_targets", set())),
        }
        response = http_request(
            self.scheduler_address,
            method=NetworkAPIMethod.SCHEDULER_GENERATION_ADMISSION,
            timeout=self.timeout_s,
            data={'data': json.dumps(payload)},
        )

        if not isinstance(response, dict):
            self._log_throttled(
                f"[Getter Filter] Scheduler generation-admission response is unavailable; "
                f"{'allow' if self.fail_open else 'block'} getter by fail_open={self.fail_open}.",
                is_error=True,
            )
            return self.fail_open

        self._execute_scheduler_actions(response)

        generate = bool(response.get("generate", response.get("allow", True)))
        if not generate:
            reason = response.get("reason", "scheduler_blocked")
            self._log_throttled(
                f"[Getter Filter] Scheduler blocked getter for source {system.source_id}: reason={reason}."
            )
        return generate
