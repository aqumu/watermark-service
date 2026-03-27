"""API endpoint definitions."""

import io
import logging
import zipfile

import torch
from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import Response

from src.api.dependencies import get_pipeline
from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    JobStatusResponse,
    JobSubmitResponse,
)
from src.pipeline.pipeline import WatermarkRemovalPipeline
from src.worker.job_manager import get_job_manager

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health(pipeline: WatermarkRemovalPipeline = Depends(get_pipeline)):
    gpu_used = gpu_total = None
    if pipeline.device.type == "cuda":
        gpu_used = torch.cuda.memory_allocated(pipeline.device) / 1024 / 1024
        gpu_total = torch.cuda.get_device_properties(pipeline.device).total_mem / 1024 / 1024

    return HealthResponse(
        status="ok",
        device=str(pipeline.device),
        models_loaded=True,
        gpu_memory_used_mb=round(gpu_used, 1) if gpu_used else None,
        gpu_memory_total_mb=round(gpu_total, 1) if gpu_total else None,
    )


# ── Single image ──────────────────────────────────────────────────────────────

@router.post("/process")
async def process_single(
    image: UploadFile = File(...),
    output_format: str = Query("png", pattern="^(png|jpeg|webp)$"),
    quality: int = Query(95, ge=1, le=100),
    feather: int | None = Query(None, ge=0),
    mask_expand: int | None = Query(None, ge=0),
    pipeline: WatermarkRemovalPipeline = Depends(get_pipeline),
):
    """Process a single image. Returns the cleaned image directly."""
    data = await image.read()
    try:
        result_bytes = pipeline.process_single(data, fmt=output_format, quality=quality)
    except (ValueError, RuntimeError) as e:
        return Response(
            content=ErrorResponse(detail=str(e)).model_dump_json(),
            status_code=400,
            media_type="application/json",
        )

    media_types = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
    return Response(content=result_bytes, media_type=media_types[output_format])


# ── Batch (sync) ──────────────────────────────────────────────────────────────

@router.post("/process/batch")
async def process_batch(
    images: list[UploadFile] = File(...),
    output_format: str = Query("png", pattern="^(png|jpeg|webp)$"),
    quality: int = Query(95, ge=1, le=100),
    pipeline: WatermarkRemovalPipeline = Depends(get_pipeline),
):
    """Process multiple images synchronously. Returns a ZIP archive."""
    image_data = [await f.read() for f in images]

    results = pipeline.process_batch(image_data, fmt=output_format, quality=quality)

    ext = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}[output_format]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for i, result in enumerate(results):
            name = f"result_{i:04d}"
            if isinstance(result, str):
                # Error — write a text file
                zf.writestr(f"{name}_error.txt", result)
            else:
                zf.writestr(f"{name}{ext}", result)
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=results.zip"},
    )


# ── Batch (async) ─────────────────────────────────────────────────────────────

@router.post("/process/batch/async", response_model=JobSubmitResponse)
async def process_batch_async(
    images: list[UploadFile] = File(...),
    output_format: str = Query("png", pattern="^(png|jpeg|webp)$"),
    quality: int = Query(95, ge=1, le=100),
    pipeline: WatermarkRemovalPipeline = Depends(get_pipeline),
):
    """Submit a large batch for background processing. Returns a job ID."""
    image_data = [await f.read() for f in images]

    manager = get_job_manager()
    job_id = manager.submit(image_data, pipeline, fmt=output_format, quality=quality)

    return JobSubmitResponse(job_id=job_id, total=len(image_data))


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str):
    """Poll the status and progress of an async batch job."""
    manager = get_job_manager()
    state = manager.get(job_id)
    if state is None:
        return Response(
            content=ErrorResponse(detail="Job not found").model_dump_json(),
            status_code=404,
            media_type="application/json",
        )
    return JobStatusResponse(
        job_id=state.job_id,
        status=state.status,
        total=state.total,
        completed=state.completed,
        failed=state.failed,
    )


@router.get("/jobs/{job_id}/results")
def job_results(
    job_id: str,
    output_format: str = Query("png", pattern="^(png|jpeg|webp)$"),
):
    """Download the results of a completed async job as a ZIP."""
    manager = get_job_manager()
    state = manager.get(job_id)
    if state is None:
        return Response(
            content=ErrorResponse(detail="Job not found").model_dump_json(),
            status_code=404,
            media_type="application/json",
        )
    if state.status != "completed":
        return Response(
            content=ErrorResponse(detail=f"Job status: {state.status}").model_dump_json(),
            status_code=409,
            media_type="application/json",
        )

    ext = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}[output_format]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for i, result in enumerate(state.results):
            name = f"result_{i:04d}"
            if isinstance(result, str):
                zf.writestr(f"{name}_error.txt", result)
            else:
                zf.writestr(f"{name}{ext}", result)
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=job_{job_id}.zip"},
    )
