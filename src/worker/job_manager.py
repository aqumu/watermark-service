"""In-memory async job manager for large batch processing."""

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class JobState:
    job_id: str
    total: int
    status: str = "pending"           # pending | processing | completed | failed
    completed: int = 0
    failed: int = 0
    results: list[Any] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update_progress(self, done: int, total: int):
        with self._lock:
            self.completed = done


class JobManager:
    """Manages background batch processing jobs."""

    def __init__(self, max_concurrent: int = 4):
        self.jobs: dict[str, JobState] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._lock = threading.Lock()

    def submit(self, images: list[bytes], pipeline, fmt: str = "png",
               quality: int = 95) -> str:
        job_id = uuid.uuid4().hex[:16]
        state = JobState(job_id=job_id, total=len(images))

        with self._lock:
            self.jobs[job_id] = state

        self.executor.submit(self._run, state, images, pipeline, fmt, quality)
        logger.info("Job %s submitted: %d images", job_id, len(images))
        return job_id

    def get(self, job_id: str) -> JobState | None:
        return self.jobs.get(job_id)

    def _run(self, state: JobState, images: list[bytes], pipeline,
             fmt: str, quality: int):
        state.status = "processing"
        try:
            results = pipeline.process_batch(
                images,
                progress_callback=state.update_progress,
                fmt=fmt,
                quality=quality,
            )
            state.results = results
            state.failed = sum(1 for r in results if isinstance(r, str))
            state.completed = state.total
            state.status = "completed"
            logger.info("Job %s completed: %d ok, %d failed",
                        state.job_id, state.total - state.failed, state.failed)
        except Exception as e:
            state.status = "failed"
            state.results = [f"error: {e}"] * state.total
            state.failed = state.total
            logger.exception("Job %s failed: %s", state.job_id, e)


# Module-level singleton
_manager: JobManager | None = None


def init_job_manager(max_concurrent: int = 4) -> JobManager:
    global _manager
    _manager = JobManager(max_concurrent=max_concurrent)
    return _manager


def get_job_manager() -> JobManager:
    if _manager is None:
        raise RuntimeError("JobManager not initialized")
    return _manager
