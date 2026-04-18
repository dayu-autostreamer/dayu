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
                 log_interval_s: float = 10.0):
        self.fail_open = bool(fail_open)
        self.timeout_s = max(0.1, float(timeout_s))
        self.log_interval_s = max(1.0, float(log_interval_s))
        self._last_block_log_t = 0.0
        self._last_error_log_t = 0.0

        scheduler_hostname = NodeInfo.get_cloud_node()
        scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
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

    def __call__(self, system):
        payload = {
            "source_id": system.source_id,
            "source_device": system.local_device,
            "meta_data": system.meta_data,
            "raw_meta_data": system.raw_meta_data,
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

        generate = bool(response.get("generate", response.get("allow", True)))
        if not generate:
            reason = response.get("reason", "scheduler_blocked")
            self._log_throttled(
                f"[Getter Filter] Scheduler blocked getter for source {system.source_id}: reason={reason}."
            )
        return generate
