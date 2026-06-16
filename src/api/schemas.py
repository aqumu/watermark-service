"""Pydantic request/response models for the API."""

from enum import Enum

from pydantic import BaseModel


class ProcessingMode(str, Enum):
    old = "old"
    new = "new"


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: bool
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None


class JobSubmitResponse(BaseModel):
    job_id: str
    total: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # pending | processing | completed | failed
    total: int
    completed: int
    failed: int


class ErrorResponse(BaseModel):
    detail: str
