from core.lib.algorithms.loader import load_hooks


_hooks = load_hooks(
    __name__,
    __path__,
    optional_dependencies=True,
    warning_subject="frame compressor",
)
globals().update(_hooks)
__all__ = list(_hooks)
