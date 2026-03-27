# Watermark Removal Service

Microservice that detects and removes watermarks from images using two models in sequence:

1. **Segmentation** (EfficientNet-B0 U-Net) — predicts a binary watermark mask
2. **Removal** (MaskedUNet) — removes the watermark using the predicted mask
3. **Upscale** (placeholder) — reserved for a future upscaling model

Results are blended back to the original image resolution with feathered edges.

## Setup

### Local (using the training venv)

```bash
cd watermark-service
source ../watermark-removal/.venv/Scripts/activate
pip install fastapi uvicorn[standard] python-multipart pydantic-settings
```

### Local (standalone)

```bash
cd watermark-service
bash scripts/setup.sh
source .venv/bin/activate
```

The setup script auto-detects CUDA, installs the right PyTorch build, and symlinks the latest checkpoints from the training repo.

### Docker

```bash
docker compose up
```

GPU passthrough is configured automatically. Checkpoints are mounted from `models/`.

## Checkpoints

Place your checkpoints in `models/` using this naming convention:

```
models/model_seg_<version>.pth    # segmentation
models/model_rem_<version>.pth    # removal
```

Version can be anything: `1.0`, `2.1_finetune`, `3.0_large`, etc.

On startup the service auto-detects the **newest** file (by modification time) for each prefix. To pin a specific version, set the path explicitly in `config/default.yaml`:

```yaml
model:
  seg_checkpoint: "./models/model_seg_1.0.pth"
  removal_checkpoint: "./models/model_rem_1.0.pth"
```

## Running the server

```bash
uvicorn src.app:create_app --factory --host 0.0.0.0 --port 8000
```

## API

All endpoints are under `/api/v1`.

### Health check

```bash
curl http://localhost:8000/api/v1/health
```

Returns device info, GPU memory usage, and model load status.

### Single image

```bash
curl -X POST -F "image=@photo.jpg" http://localhost:8000/api/v1/process -o result.png
```

Optional query parameters:
- `output_format` — `png` (default), `jpeg`, or `webp`
- `quality` — 1–100, default 95 (for jpeg/webp)
- `feather` — blend softness radius in px, default 9
- `mask_expand` — extra mask dilation in px, default 0

Example with JPEG output:
```bash
curl -X POST -F "image=@photo.jpg" \
  "http://localhost:8000/api/v1/process?output_format=jpeg&quality=95" \
  -o result.jpg
```

### Batch (synchronous)

Processes multiple images and returns a ZIP archive:

```bash
curl -X POST \
  -F "images=@a.jpg" -F "images=@b.jpg" -F "images=@c.jpg" \
  http://localhost:8000/api/v1/process/batch -o results.zip
```

### Batch (asynchronous)

For large batches — submit, poll, then download:

```bash
# 1. Submit
curl -X POST \
  -F "images=@a.jpg" -F "images=@b.jpg" \
  http://localhost:8000/api/v1/process/batch/async
# → {"job_id": "abc123...", "total": 2}

# 2. Poll status
curl http://localhost:8000/api/v1/jobs/abc123
# → {"job_id": "abc123", "status": "processing", "total": 2, "completed": 1, "failed": 0}

# 3. Download results when completed
curl http://localhost:8000/api/v1/jobs/abc123/results -o results.zip
```

## Configuration

All settings live in `config/default.yaml` and can be overridden with environment variables (e.g. `WM_CONFIG_PATH` to point to a different config file).

Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `model.models_dir` | `./models` | Directory to scan for checkpoints |
| `model.seg_checkpoint` | `auto` | Seg checkpoint path or `auto` |
| `model.removal_checkpoint` | `auto` | Removal checkpoint path or `auto` |
| `inference.device` | `auto` | `auto`, `cuda`, or `cpu` |
| `inference.amp` | `true` | Mixed precision (CUDA only) |
| `batch.max_batch_size` | `8` | Max images per GPU forward pass |

## Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

## Project structure

```
watermark-service/
├── config/default.yaml          # service configuration
├── models/                      # checkpoints (gitignored)
│   ├── model_seg_1.0.pth
│   └── model_rem_1.0.pth
├── src/
│   ├── app.py                   # FastAPI app factory
│   ├── config.py                # config loading + checkpoint auto-detection
│   ├── api/                     # endpoints, schemas, dependency injection
│   ├── pipeline/                # segmentation → removal → upscale → blending
│   ├── models/                  # model architectures + checkpoint loader
│   ├── processing/              # image I/O and utilities
│   └── worker/                  # async job manager
├── tests/
├── Dockerfile
├── docker-compose.yml
├── scripts/setup.sh
└── pyproject.toml
```
