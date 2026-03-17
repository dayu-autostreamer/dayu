import importlib
import warnings

__all__ = []


def _optional_import(module_name, symbol_name):
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("core."):
            raise
        warnings.warn(
            f"Skip loading optional processor '{symbol_name}' because dependency '{exc.name}' is unavailable.",
            RuntimeWarning,
        )
        return

    globals()[symbol_name] = getattr(module, symbol_name)
    __all__.append(symbol_name)


_optional_import("detector_processor", "DetectorProcessor")
_optional_import("detector_tracker_processor", "DetectorTrackerProcessor")
_optional_import("classifier_processor", "ClassifierProcessor")
_optional_import("roi_classifier_processor", "RoiClassifierProcessor")

from .processor_server import ProcessorServer

__all__.append("ProcessorServer")
