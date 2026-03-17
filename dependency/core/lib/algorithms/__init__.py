import pkgutil
import importlib
import warnings

__all__ = []


for _, name, is_pkg in pkgutil.iter_modules(__path__):
    if is_pkg:
        try:
            module = importlib.import_module(f'.{name}', __name__)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith('core.'):
                raise
            warnings.warn(
                f"Skip loading optional algorithm package '{name}' because dependency '{exc.name}' is unavailable.",
                RuntimeWarning,
            )
            continue
        globals()[name] = module
        __all__.append(name)
