from importlib.metadata import PackageNotFoundError, version

from . import clustering as cl
from . import graph_construction as gc
from . import nhood_embedding as ne
from . import utils
from . import workflows as wf

__all__ = ["gc", "ne", "cl", "wf", "utils"]

try:
    __version__ = version("nichepca")
except PackageNotFoundError:  # pragma: no cover - fallback for local development
    __version__ = "0+unknown"
