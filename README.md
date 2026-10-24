# Watermark Removal Service

FastAPI microservice that removes watermarks from images. Processing modes:

- **"avito" mode** — Avito's current small watermark (bottom-right logo): a small ONNX model on CPU on the image's corner window; the mark sits at fixed pixels, so nothing has to be detected
- **"legacy_avito" mode** (alias `old`) — Avito's original (large) watermark, now out of circulation: segmentation, template alignment, residual removal, feather blending
- **"cian" mode** — Cian: the mark is located from the image size and the listing ID and removed by a small ONNX model on CPU (needs the listing ID of each image)
- **"new" mode** — the earlier treatment of the small Avito watermark (upscaling + fixed-region inpainting, Telea or LaMa), kept for existing callers; use `avito` instead

## Quick start

### Local

```bash
pip install -e ".[dev]"
uvicorn src.app:create_app --factory --host 0.0.0.0 --port 8000
```

For GPU support, install PyTorch with CUDA before pip install:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

### Docker

```bash
docker compose up
```

Models and config are mounted from the host (`./models/`, `./config/`). For GPU support, uncomment the `deploy.resources` block in `docker-compose.yml` and install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Checkpoints

Place checkpoints in `models/` using these naming conventions:

| Model | File pattern | Required? |
|-------|-------------|-----------|
| Segmentation | `model_seg_*.pth` | Yes |
| Removal | `model_rem_*.pth` | Yes |
| Watermark template | `watermark.png` | Yes (RGBA) |
| Real-ESRGAN upscaler | `RealESRGAN_x2plus.pth` | Only if upscale enabled |
| LaMa inpainter | `big-lama/**/*.ckpt` | Only if new.method=lama |
| Cian removal model | `model_cian_*.onnx` | Only if cian.enabled |
| Cian mark assets | `cian/mark.json`, `cian/mark.npz`, `cian/fonts/` | Only if cian.enabled |

The Cian model and mark assets are produced by `training/export_for_service.py` in the
watermark-removal repo (`--out <this repo>/models`). **Deploy them together**: the model is
trained on the mark model in `cian/mark.json` (logo shadow, text-line strength), and the hint
the service renders from it must match. Keep a single `model_cian_*.onnx` in `models/`.

Current release: `model_cian_c_v3.onnx` (2.3 MB, distilled student "C v3"). ~31 ms per image
on one 6-core desktop CPU with all threads, ~30-34 images/s for the whole machine. It removes
the logo, its shadow and the "ID" line at normal viewing; faint digit ghosts can remain,
visible only at strongly increased contrast. The full-resolution teacher (`model_cian_ft7.onnx`,
8.5 MB) is cleaner at high contrast but ~10-20x slower (~1.5-3 images/s); it uses the same
mark assets and can be swapped in by replacing the model file.

On startup, the service auto-detects the newest checkpoint for each prefix (by modification time). Set an explicit path in `config/default.yaml` to pin a version:

```yaml
model:
  seg_checkpoint: "./models/model_seg_2.0.pth"
  removal_checkpoint: "./models/model_rem_best.pth"
```

## API

All endpoints are under `/api/v1`.

### `GET /api/v1/health`

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "ok",
  "device": "cuda",
  "models_loaded": true,
  "gpu_memory_used_mb": 512.0,
  "gpu_memory_total_mb": 8192.0
}
```

### `POST /api/v1/process`

Process a single image. Returns the cleaned image directly.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | file | — | Input image (required) |
| `mode` | string | `old` | `"avito"`, `"legacy_avito"` (= `"old"`), `"cian"` or `"new"` |
| `listing_id` | string | — | Cian listing ID (required for `mode=cian`) |
| `output_format` | string | `png` | `"png"`, `"jpeg"`, or `"webp"` |
| `quality` | int | `95` | 1–100, for jpeg/webp |

```bash
curl -X POST -F "image=@photo.jpg" "http://localhost:8000/api/v1/process" -o result.png
curl -X POST -F "image=@photo.jpg" "http://localhost:8000/api/v1/process?mode=new&output_format=jpeg&quality=90" -o result.jpg
curl -X POST -F "image=@photo.jpg" "http://localhost:8000/api/v1/process?mode=cian&listing_id=333768830" -o result.png
```

### `POST /api/v1/process/batch`

Process multiple images synchronously. Returns a ZIP archive.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | files | — | Multiple image files (required) |
| `mode` | string | `old` | `"avito"`, `"legacy_avito"` (= `"old"`), `"cian"` or `"new"` |
| `listing_ids` | form field | — | `mode=cian`: comma-separated listing IDs, one per image, in upload order |
| `output_format` | string | `png` | `"png"`, `"jpeg"`, or `"webp"` |
| `quality` | int | `95` | 1–100 |

```bash
curl -X POST -F "images=@a.jpg" -F "images=@b.jpg" \
  "http://localhost:8000/api/v1/process/batch" -o results.zip
curl -X POST -F "images=@a.jpg" -F "images=@b.jpg" -F "listing_ids=333768830,329288428" \
  "http://localhost:8000/api/v1/process/batch?mode=cian" -o results.zip
```

### `POST /api/v1/process/batch/async`

Submit a large batch for background processing. Returns a job ID.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | files | — | Multiple image files (required) |
| `mode` | string | `old` | `"avito"`, `"legacy_avito"` (= `"old"`), `"cian"` or `"new"` |
| `listing_ids` | form field | — | `mode=cian`: comma-separated listing IDs, one per image, in upload order |
| `output_format` | string | `png` | `"png"`, `"jpeg"`, or `"webp"` |
| `quality` | int | `95` | 1–100 |

```bash
curl -X POST -F "images=@a.jpg" -F "images=@b.jpg" \
  "http://localhost:8000/api/v1/process/batch/async"
# → {"job_id": "abc123...", "total": 2}
```

### `GET /api/v1/jobs/{job_id}`

Poll job status. Status values: `pending`, `processing`, `completed`, `failed`.

```bash
curl http://localhost:8000/api/v1/jobs/abc123
# → {"job_id": "abc123", "status": "processing", "total": 2, "completed": 1, "failed": 0}
```

### `GET /api/v1/jobs/{job_id}/results`

Download completed job results as a ZIP.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_format` | string | `png` | `"png"`, `"jpeg"`, or `"webp"` |

```bash
curl http://localhost:8000/api/v1/jobs/abc123/results -o results.zip
```

## Processing modes

### "old" mode pipeline

```
Input → Segmentation → Alignment → Removal → Blend → Upscale (optional) → Output
```

1. **Segmentation** (GPU) — EfficientNet-B0 U-Net predicts a binary watermark probability map at 256×256, then upscales to original resolution.
2. **Alignment** (CPU) — Aligns the watermark template (`watermark.png`) to the probability map via FFT-based cross-correlation with subpixel refinement and scale search. Falls back to the raw seg mask if no template is available.
3. **Removal** (GPU) — Crops a fixed-aspect-ratio ROI around the watermark, runs MaskedUNet (residual prediction U-Net with GroupNorm), stores the inpainted ROI.
4. **Blend** (CPU) — Feather-blends the ROI prediction back into the original image with configurable Gaussian blur and mask expansion.
5. **Upscale** (GPU, optional) — Real-ESRGAN upscaling for images whose longest side is ≤ `resolution_threshold` (default 720px). Uses tiled inference for bounded VRAM.

### "new" mode pipeline

```
Input → Upscale → Inpaint → Output
```

1. **Upscale** (GPU, optional) — Same Real-ESRGAN step as above.
2. **Inpaint** (CPU/GPU) — Inpaints a fixed rectangle at the bottom-right corner of the image. Two methods:
   - `"telea"` (CPU) — OpenCV Telea inpainting, fast, no model needed.
   - `"lama"` (GPU) — LaMa FFCResNetGenerator, better quality, requires a checkpoint.

### "cian" mode pipeline

```
Input + listing ID → Window → Locate mark → Render hint → ONNX model → Paste back → Output
```

1. **Window** (CPU) — the Cian mark sits at a position that follows the image size; a fixed 352×224 window around it is cut out (aligned to the JPEG 16-px grid). The rest of the image is never touched.
2. **Locate** (CPU) — template matching finds the logo's exact position and size, then the ID line's style (bold / regular / none) and placement for the listing's ID.
3. **Hint** (CPU) — the exact mark for that ID is rendered there; the model input is the window, the exact algebraic inverse of the mark, and the mark itself.
4. **Model** (CPU, ONNX Runtime) — predicts a correction on top (JPEG damage around the letters, per-photo deviations of the mark); images are batched.
5. **Paste back** — only the window changes. No upscaling.

A photo re-used from another listing can carry that listing's ID; the model tolerates a wrong ID hint, but results are best with the right ID. A second, older mark from another listing (re-used photos, ~1%) is only partly removed: its ID is unknown.

Enable it with `cian.enabled: true` (see `config/default.yaml`); the other modes are unaffected.

## Configuration

All settings in `config/default.yaml` (overridable via `WM_CONFIG_PATH` env var).

### Model settings

| Key | Default | Description |
|-----|---------|-------------|
| `model.models_dir` | `./models` | Directory to scan for checkpoints |
| `model.seg_checkpoint` | `auto` | Path or `"auto"` (scans for newest `model_seg_*.pth`) |
| `model.removal_checkpoint` | `auto` | Path or `"auto"` (scans for newest `model_rem_*.pth`) |
| `model.seg_image_size` | `256` | Segmentation model input size (square) |
| `model.removal_image_width` | `512` | Removal model input width |
| `model.removal_image_height` | `256` | Removal model input height |
| `model.removal_crop_aspect_ratio` | `3.54` | Aspect ratio of the removal ROI crop |
| `model.removal_crop_margin_ratio` | `0.12` | Margin around the watermark bounding box |
| `model.seg_encoder` | `efficientnet-b0` | SMP encoder backbone |
| `model.watermark_path` | `./models/watermark.png` | RGBA watermark template for alignment |

### Inference settings

| Key | Default | Description |
|-----|---------|-------------|
| `inference.device` | `auto` | `"auto"` (CUDA if available), `"cuda"`, or `"cpu"` |
| `inference.mask_threshold` | `0.5` | Sigmoid threshold for binary segmentation mask |
| `inference.feather_radius` | `9` | Gaussian blur radius for feather blending (px) |
| `inference.mask_expand` | `0` | Extra dilation before blending (px) |
| `inference.amp` | `true` | Automatic mixed precision (CUDA only) |

### Upscale settings

| Key | Default | Description |
|-----|---------|-------------|
| `upscale.enabled` | `true` | Enable Real-ESRGAN upscaling |
| `upscale.model_name` | `RealESRGAN_x2plus` | Architecture variant |
| `upscale.model_path` | `auto` | `"auto"` resolves to `models/{model_name}.pth` |
| `upscale.tile` | `512` | Tile size for VRAM-bounded inference (0 = no tiling) |
| `upscale.half` | `true` | FP16 inference on CUDA |
| `upscale.resolution_threshold` | `720` | Upscale images whose longest side ≤ this (px). Set 0 to always upscale. |

Supported upscaler variants:

| `model_name` | Scale | Notes |
|---|---|---|
| `RealESRGAN_x4plus` | 4× | General images, 23 RRDB blocks |
| `RealESRGAN_x2plus` | 2× | General images, 23 RRDB blocks |
| `RealESRGAN_x4plus_anime_6B` | 4× | Anime/illustrations, 6 RRDB blocks |

### Inpainting settings ("new" mode)

| Key | Default | Description |
|-----|---------|-------------|
| `new.enabled` | `true` | Enable "new" processing mode |
| `new.method` | `lama` | Compatibility fallback only: `"telea"` or `"lama"` |
| `new.inpaint_width` | `114` | Width of the fixed inpaint rectangle |
| `new.inpaint_height` | `48` | Height of the fixed inpaint rectangle |
| `new.inpaint_radius` | `3` | Telea inpainting radius |
| `new.inpaint_size` | `256` | LaMa crop size (square) |
| `new.checkpoint` | `auto` | `"auto"` scans `models/` recursively for `*.ckpt` |

### Batch settings

| Key | Default | Description |
|-----|---------|-------------|
| `batch.max_batch_size` | `8` | Max images per GPU forward pass |
| `batch.io_workers` | `4` | Thread pool for I/O-bound operations |
| `batch.max_concurrent_jobs` | `4` | Max concurrent async batch jobs |

### Avito settings ("avito" mode)

Current release: `model_avito_v1.onnx` (8.5 MB) with `models/avito/` (mark.json,
mark.npz), both from the same `watermark-removal` export. Set `avito.enabled: true`.

| Key | Default | Description |
|---|---|---|
| `avito.enabled` | `false` | Load the small-logo ONNX model |
| `avito.onnx` | `auto` | Newest `model_avito_*.onnx`, or an explicit path |
| `avito.mark_dir` | `auto` | `models/avito/` (mark.json, mark.npz), exported with the model |
| `avito.threads` | `0` | ONNX Runtime intra-op threads |
| `avito.batch_size` | `8` | Images per model call |
| `avito.hint_scale` | `8.0` | Must match the training configuration |

### Cian settings ("cian" mode)

| Key | Default | Description |
|-----|---------|-------------|
| `cian.enabled` | `false` | Load the Cian model at startup and accept `mode=cian` |
| `cian.onnx` | `"auto"` | Newest `models_dir/model_cian_*.onnx`, or an explicit path |
| `cian.mark_dir` | `"auto"` | `models_dir/cian` (mark.json, mark.npz, fonts/) |
| `cian.threads` | `0` | ONNX Runtime threads (0 = runtime default) |
| `cian.batch_size` | `8` | Images per model call |
| `cian.hint_scale` | `8.0` | Must match the model's training config |

## Project structure

```
watermark-service/
├── config/default.yaml         # service configuration
├── models/                     # checkpoints (gitignored)
├── src/
│   ├── app.py                  # FastAPI app factory (lifespan, router)
│   ├── config.py               # YAML loader + checkpoint auto-detection
│   ├── api/
│   │   ├── routes.py           # 6 endpoints
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── dependencies.py     # DI singletons
│   ├── pipeline/
│   │   ├── pipeline.py         # orchestrator (old/new mode, sub-batching)
│   │   ├── segmentation.py     # EfficientNet-B0 U-Net mask prediction
│   │   ├── alignment.py        # template cross-correlation alignment
│   │   ├── removal.py          # MaskedUNet residual prediction
│   │   ├── blending.py         # feathered ROI paste-back
│   │   ├── upscale.py          # Real-ESRGAN with tiled inference
│   │   ├── inpaint.py          # Telea / LaMa inpainting
│   │   └── context.py          # per-image state dataclass
│   ├── models/
│   │   ├── checkpoint.py       # unified checkpointer (prefers EMA)
│   │   ├── seg_model.py        # SMP U-Net builder
│   │   ├── masked_unet.py      # residual U-Net with GroupNorm
│   │   └── lama.py             # LaMa FFCResNetGenerator
│   ├── processing/
│   │   ├── io.py               # BGR encode/decode
│   │   └── image_utils.py      # crop, resize, dilate, ROI helpers
│   └── worker/
│       └── job_manager.py      # in-memory async batch manager
├── scripts/
│   ├── batch_process.py        # Python CLI for async batch jobs
│   ├── batch_process.ps1       # PowerShell CLI for sync batch
│   ├── debug_single_image.py   # run and save all intermediate artifacts
│   ├── debug_dataset_pair.py   # compare against ground-truth mask
│   └── setup.sh                # automated local venv setup (Linux/macOS)
├── tests/
│   ├── test_api.py             # API integration tests (mocked pipeline)
│   └── test_pipeline.py        # component unit tests (no real weights)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
