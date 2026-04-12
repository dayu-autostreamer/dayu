import abc
import copy
from typing import Any, Dict, Optional

from core.lib.common import ClassFactory, ClassType, Context, YamlOps

from .base_config_extraction import BaseConfigExtraction

__all__ = ("HedgerConfigExtraction",)


@ClassFactory.register(ClassType.SCH_CONFIG_EXTRACTION, alias="hedger")
class HedgerConfigExtraction(BaseConfigExtraction, abc.ABC):
    """
    Load a single user-facing Hedger config file and apply one optional profile
    plus one optional overrides block.

    External interface:
        - config: required path or dict
        - profile: optional named preset under `profiles`
        - overrides: optional final deep-merge patch

    Final merged config is exposed as:
        scheduler.hedger_config
    """

    def __init__(
        self,
        config: Any,
        profile: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        self.config_ref = config
        self.profile = profile
        self.overrides = overrides

    @staticmethod
    def _deep_merge(base: Any, override: Any):
        if override is None:
            return None
        if base is None:
            return copy.deepcopy(override)
        if not isinstance(base, dict) or not isinstance(override, dict):
            return copy.deepcopy(override)

        merged = copy.deepcopy(base)
        for key, value in override.items():
            merged[key] = HedgerConfigExtraction._deep_merge(merged.get(key), value)
        return merged

    @staticmethod
    def _ensure_mapping(name: str, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError(f"Hedger config section '{name}' must be a mapping, got {type(value).__name__}.")
        return copy.deepcopy(value)

    def _load_yaml(self, config_ref: Any) -> Dict[str, Any]:
        if isinstance(config_ref, dict):
            return copy.deepcopy(config_ref)
        if not isinstance(config_ref, str):
            raise TypeError(f"Hedger config reference must be a path or mapping, got {type(config_ref).__name__}.")

        config_path = Context.get_file_path(config_ref)
        loaded = YamlOps.read_yaml(config_path)
        return {} if loaded is None else loaded

    def _resolve_merged_config(self) -> Dict[str, Any]:
        raw_config = self._ensure_mapping("config", self._load_yaml(self.config_ref))
        profiles = self._ensure_mapping("profiles", raw_config.get("profiles"))

        base_config = copy.deepcopy(raw_config)
        base_config.pop("profiles", None)
        default_profile = base_config.pop("default_profile", None)
        selected_profile_name = self.profile if self.profile is not None else default_profile

        merged = base_config
        if selected_profile_name is not None:
            if selected_profile_name not in profiles:
                raise ValueError(
                    f"Unknown Hedger profile '{selected_profile_name}'. "
                    f"Available profiles: {', '.join(sorted(profiles)) or '<none>'}."
                )
            merged = self._deep_merge(
                merged,
                self._ensure_mapping(f"profiles.{selected_profile_name}", profiles[selected_profile_name]),
            )

        if self.overrides is not None:
            merged = self._deep_merge(merged, self._ensure_mapping("overrides", self.overrides))

        mode = merged.get("mode")
        if mode not in {"train", "inference"}:
            raise ValueError("Hedger config requires top-level `mode` to be either 'train' or 'inference'.")
        if mode == "train":
            training = self._ensure_mapping("training", merged.get("training"))
            stage = training.get("stage")
            if stage not in {"offloading_warmup", "deployment_adaptation", "joint_finetune"}:
                raise ValueError(
                    "Hedger train mode requires `training.stage` to be one of "
                    "'offloading_warmup', 'deployment_adaptation', or 'joint_finetune'."
                )

        return merged

    def __call__(self, scheduler):
        scheduler.hedger_config = self._resolve_merged_config()
