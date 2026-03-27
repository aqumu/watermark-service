"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.dependencies import get_config, init_pipeline
from src.api.routes import router
from src.worker.job_manager import init_job_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models. Shutdown: cleanup."""
    config = get_config()

    logging.basicConfig(
        level=getattr(logging, config.server.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting watermark-service...")

    init_pipeline()
    init_job_manager(max_concurrent=config.batch.max_concurrent_jobs)

    logger.info("Service ready")
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Watermark Removal Service",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router, prefix="/api/v1")
    return app
