import abc

from core.lib.common import ClassFactory, ClassType, LOGGER

from .base_config_extraction import BaseConfigExtraction

__all__ = ("DeepVAConfigExtraction",)


@ClassFactory.register(ClassType.SCH_CONFIG_EXTRACTION, alias="deepva")
class DeepVAConfigExtraction(BaseConfigExtraction, abc.ABC):
    """Config extraction for the DeepVA baseline."""

    def __init__(
        self,
        service_list,
        device_list,
        device_service_limits,
        num_services=None,
        num_devices=None,
    ):
        self.service_list_raw = service_list
        self.device_list_raw = device_list
        self.device_service_limits_raw = device_service_limits
        self.num_services = num_services
        self.num_devices = num_devices

    @staticmethod
    def _parse_list(value, name):
        if isinstance(value, str):
            parsed = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, (list, tuple)):
            parsed = [str(item).strip() for item in value if str(item).strip()]
        else:
            raise TypeError(f"DeepVA config `{name}` must be a comma-separated string or list.")
        if not parsed:
            raise ValueError(f"DeepVA config `{name}` must not be empty.")
        return parsed

    @staticmethod
    def _parse_int_list(value, name):
        if isinstance(value, str):
            raw = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, (list, tuple)):
            raw = list(value)
        else:
            raise TypeError(f"DeepVA config `{name}` must be a comma-separated string or list.")
        parsed = [int(item) for item in raw]
        if any(item <= 0 for item in parsed):
            raise ValueError(f"DeepVA config `{name}` values must be positive.")
        return parsed

    def __call__(self, scheduler):
        service_list = self._parse_list(self.service_list_raw, "service_list")
        device_list = self._parse_list(self.device_list_raw, "device_list")
        device_service_limits = self._parse_int_list(self.device_service_limits_raw, "device_service_limits")

        expected_services = len(service_list) if self.num_services is None else int(self.num_services)
        expected_devices = len(device_list) if self.num_devices is None else int(self.num_devices)
        if expected_services != len(service_list):
            raise ValueError("DeepVA `num_services` does not match `service_list` length.")
        if expected_devices != len(device_list):
            raise ValueError("DeepVA `num_devices` does not match `device_list` length.")
        if len(device_service_limits) != len(device_list):
            raise ValueError("DeepVA `device_service_limits` length must match `device_list` length.")
        if sum(device_service_limits) < len(service_list):
            raise ValueError("DeepVA total device capacity must cover all services.")

        scheduler.service_list = service_list
        scheduler.num_services = len(service_list)
        scheduler.device_list = device_list
        scheduler.num_devices = len(device_list)
        scheduler.device_service_limits = device_service_limits

        LOGGER.info(
            f"[DeepVA Config] services={service_list}, devices={device_list}, "
            f"limits={device_service_limits}"
        )
