"""
File Summary:
- Defines top-level package metadata and convenience imports for driftcast.
- Exposes the package version and a helper to configure the Loguru logger.
- See driftcast.cli for executable entry points that build on these utilities.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Optional

try:
    from loguru import logger as _logger
    _LOGURU_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when loguru missing
    import logging

    _LOGURU_AVAILABLE = False
    _base_logger = logging.getLogger("driftcast")
    _base_logger.setLevel(logging.INFO)

    class _FallbackLogger:
        def remove(self) -> None:
            for handler in list(_base_logger.handlers):
                _base_logger.removeHandler(handler)

        def add(self, sink, level: str = "INFO", **kwargs):
            if isinstance(sink, str):
                handler = logging.FileHandler(sink)
            elif callable(sink):
                class _CallableHandler(logging.Handler):
                    def emit(self, record):
                        sink(self.format(record))

                handler = _CallableHandler()
            else:
                handler = logging.StreamHandler(sink)
            handler.setLevel(getattr(logging, level.upper(), logging.INFO))
            handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
            _base_logger.addHandler(handler)

        def __getattr__(self, item):
            return getattr(_base_logger, item)

    _logger = _FallbackLogger()

logger = _logger

__all__ = ["__version__", "configure_logging"]


def _discover_version() -> str:
    """Return the installed package version or a development fallback."""
    try:
        return version("driftcast")
    except PackageNotFoundError:
        return "0.1.0-dev"


__version__ = _discover_version()


def configure_logging(level: str = "INFO", sink: Optional[str] = None) -> None:
    """Set up Loguru logging with sensible defaults for CLI and library use.

    Args:
        level: Global logging level (e.g., ``"INFO"`` or ``"DEBUG"``).
        sink: Optional file path to tee logs; console logging always enabled.

    Example:
        >>> configure_logging(level="DEBUG")
    """
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=level)
    if sink is not None:
        logger.add(sink, level=level, rotation="10 MB", retention="7 days")
