"""Discovery helpers for registry-backed algorithm hooks."""

import importlib
import warnings
from pathlib import Path


def _entry_modules(package_name, package_paths):
    """Yield deterministic module paths for one hook family."""

    modules = set()
    for package_path in package_paths:
        root = Path(package_path)
        for entry in root.iterdir():
            if entry.is_file():
                if entry.suffix == ".py" and not entry.stem.startswith("_"):
                    modules.add((entry.name, f"{package_name}.{entry.stem}"))
                continue

            hook_path = entry / "hook.py"
            if entry.is_dir() and hook_path.is_file():
                modules.add((entry.name, f"{package_name}.{entry.name}.hook"))

    for _, module_name in sorted(modules):
        yield module_name


def load_hooks(
    package_name,
    package_paths,
    *,
    optional_dependencies=False,
    warning_subject="algorithm hook",
):
    """Import the public files and explicit package entry points of a hook family.

    Public direct ``.py`` files are imported unconditionally. Directories opt in
    by providing ``hook.py``; their other modules are owned by that entry point.
    Names exported by an imported module's ``__all__`` are returned for the hook
    package to re-export. ``warning_subject`` is used only to describe an entry
    skipped because an optional external dependency is unavailable.
    """

    exports = {}
    prefix = f"{package_name}."
    for module_name in _entry_modules(package_name, package_paths):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if (
                not optional_dependencies
                or (exc.name and exc.name.startswith("core."))
            ):
                raise
            entry_name = module_name[len(prefix):]
            warnings.warn(
                f"Skip loading optional {warning_subject} '{entry_name}' because "
                f"dependency '{exc.name}' is unavailable.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        for name in getattr(module, "__all__", ()):
            exports[name] = getattr(module, name)

    return exports
