"""Cleaned MSB clustering workflow."""

try:
    from .config import Config
    from .clustering import prepare_clustering
except ImportError:  # pragma: no cover - allows direct import from package folder
    from config import Config
    from clustering import prepare_clustering

__all__ = ["Config", "prepare_clustering"]
