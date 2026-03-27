"""FastAPI dependency injection — pipeline singleton."""

from src.config import ServiceConfig, load_config
from src.pipeline.pipeline import WatermarkRemovalPipeline

_pipeline: WatermarkRemovalPipeline | None = None
_config: ServiceConfig | None = None


def get_config() -> ServiceConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def init_pipeline() -> WatermarkRemovalPipeline:
    """Eagerly initialize the pipeline (called at startup)."""
    global _pipeline
    _pipeline = WatermarkRemovalPipeline(get_config())
    return _pipeline


def get_pipeline() -> WatermarkRemovalPipeline:
    """Get the pipeline singleton. Raises if not initialized."""
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized — server still starting?")
    return _pipeline
