import sys
from pathlib import Path
from core.lib.common.lazy_module import LazyModule

package_path = [str(Path(__file__).parent.resolve())]
lazy_module = LazyModule(__name__, package_path, auto_scan=True)
sys.modules[__name__] = lazy_module
