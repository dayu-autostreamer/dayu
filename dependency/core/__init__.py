import core.applications
import core.lib.algorithms

# Fix for different numpy versions
try:
    import numpy as _np
except ImportError:
    pass
else:
    _ALIASES = {
        'bool': bool,
    }
    for _name, _type in _ALIASES.items():
        if not hasattr(_np, _name):
            setattr(_np, _name, _type)