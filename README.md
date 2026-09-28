# Driver Monitoring System

Real-time driver-monitoring system for drowsiness, gaze diversion, facial occlusion, yawning, and phone distraction. It combines MediaPipe facial landmarks, a two-class MobileNet action classifier, and YOLO phone detection.

## Supported applications

The canonical dashboard is the root application:

```powershell
python dms_server.py
```

It listens only on `http://127.0.0.1:5000`. `python -m dms_server.app` exposes the same application for WSGI tooling; it is not a second implementation.

The original command-line viewer remains available:

```powershell
dms --webcam 0
dms --video path\to\drive.mp4 --save
```

## Installation

Use Python 3.9 or later and install the project with its development tools when running tests:

```powershell
python -m pip install -e ".[dev]"
```

The required inference file is `models/model_split.h5`. The action classifier uses a local torchvision MobileNetV2 checkpoint when present; it intentionally does not download one during application startup. YOLO is initialized on first video stream and may require an already-cached Torch Hub repository or network access.

## Dashboard operation

The dashboard initializes models on the first `/video_feed` request. Check `/health` to see which components loaded and any component-specific errors.

Runtime uploads and incident snapshots are stored in `uploads/` and `incidents/`; these directories are intentionally ignored by Git.

For deployments beyond the local machine, enable token checks for commands that change camera, speed, engine, uploaded video, or incident data:

```powershell
$env:DMS_REQUIRE_API_TOKEN = "true"
$env:DMS_API_TOKEN = "replace-with-a-long-random-value"
python dms_server.py
```

Authenticated API requests must send that value in the `X-API-Token` header. Configure `DMS_MAX_UPLOAD_BYTES` to change the default 1 GB upload limit.

## Training

Training uses PyTorch and exports the HDF5 classifier head format consumed by the runtime:

```powershell
python train.py --data-path path\to\DMD --trainer split --save-path models\model_split.h5
```

`--trainer random` performs a stratified random split; `--trainer split` holds out the original fifth subject.

## Quality checks

```powershell
ruff check .
ruff format --check .
pytest --cov=. --cov-report=term-missing
```

The CI workflow runs these checks on Python 3.9.
