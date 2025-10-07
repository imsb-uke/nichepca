from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib

from . import clustering as cl
from . import graph_construction as gc
from . import nhood_embedding as ne
from . import utils
from . import workflows as wf

__all__ = ["gc", "ne", "cl", "wf", "utils"]


def _local_version() -> str:
    """Return the project version declared in pyproject.toml."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data["project"]["version"]


def _get_version() -> str:
    try:
        return version("nichepca")
    except PackageNotFoundError:  # pragma: no cover - fallback for local development
        return _local_version()


__version__ = _get_version()
