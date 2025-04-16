import pkgutil
import importlib

__all__ = []


for _, name, is_pkg in pkgutil.iter_modules(__path__):
    if is_pkg:
        module = importlib.import_module(f'.{name}', __name__)
        globals()[name] = module
        __all__.append(name)