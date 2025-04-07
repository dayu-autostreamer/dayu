import importlib
import pkgutil

def auto_import_submodules(package_name):
    package = importlib.import_module(package_name)
    modules = {}
    if hasattr(package, '__path__'):
        for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
            full_name = f"{package_name}.{name}"
            sub_module = importlib.import_module(full_name)
            if hasattr(sub_module, '__all__'):
                for item in sub_module.__all__:
                    modules[item] = getattr(sub_module, item)
            else:
                modules[name] = sub_module
    return modules

_submodules = auto_import_submodules(__name__)
globals().update(_submodules)

__all__ = list(_submodules.keys())
