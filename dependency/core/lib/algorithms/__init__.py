import sys
from pathlib import Path
from core.lib.common import LazyModule


_path = [str(Path(__file__).parent.resolve())]
sys.modules[__name__] = LazyModule(__name__, _path)