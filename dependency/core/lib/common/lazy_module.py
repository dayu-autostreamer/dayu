import importlib
import pkgutil
from types import ModuleType


class LazyModule(ModuleType):

    # Very heavily inspired by huggingface.diffusers._LazyModule
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/import_utils.py
    def __init__(self, name, package_path, auto_scan=True):
        super().__init__(name)
        self.__package_path__ = package_path
        self.__auto_scan__ = auto_scan
        self.__submodules__ = set()
        self.__classes__ = {}

        if self.__auto_scan__:
            self._scan_modules()

    def _scan_modules(self):
        for _, name, is_pkg in pkgutil.iter_modules(self.__package_path__):
            print(f'scan_modules name: {name}')
            if is_pkg:
                self.__submodules__.add(name)
            else:
                self.__classes__[name] = name

    def __getattr__(self, name):
        if name in self.__submodules__:
            module = importlib.import_module(f".{name}", self.__name__)
            setattr(self, name, module)
            return module

        if name in self.__classes__:
            module_name = self.__classes__[name]
            module = importlib.import_module(f".{module_name}", self.__name__)
            cls = getattr(module, name)
            setattr(self, name, cls)
            return cls

        raise AttributeError(f"module {self.__name__} has no attribute {name}")

    def __dir__(self):
        return list(self.__submodules__) + list(self.__classes__.keys())