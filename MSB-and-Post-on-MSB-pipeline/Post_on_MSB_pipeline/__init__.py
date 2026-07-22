"""Post-on-MSB filopodia pipeline."""

try:
    from .config import Config
    from .pipeline import run_pipeline
except ImportError:  # pragma: no cover - allows direct import from package folder
    from config import Config
    from pipeline import run_pipeline

__all__ = ["Config", "run_pipeline"]
