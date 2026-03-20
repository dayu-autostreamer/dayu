import pkgutil
import importlib
import warnings

__all__ = []

for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if not is_pkg:
        try:
            module = importlib.import_module(f'.{module_name}', __name__)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith('core.'):
                raise
            warnings.warn(
                f"Skip loading optional redeployment policy module '{module_name}' "
                f"because dependency '{exc.name}' is unavailable.",
                RuntimeWarning,
            )
            continue
        if hasattr(module, '__all__'):
            for item in module.__all__:
                globals()[item] = getattr(module, item)
                __all__.append(item)
