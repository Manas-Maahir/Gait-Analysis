# Strider Setup Guide

Complete setup instructions for the development environment.

## Prerequisites

- Python 3.10 or higher
- Git (for version control)
- ~500 MB disk space for models + dependencies

## Initial Setup (Windows)

### 1. Create Virtual Environment

```bash
cd C:\Users\manas\Desktop\Projects\Stride

python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 3. Download ONNX Models

```bash
python scripts/download_models.py
```

Downloads:
- `rtmdet_nano_320-8bbb47ba.onnx` (~10 MB) — person detector
- `rtmpose-l_simcc-body7_wholebody_coco-384x288.onnx` (~90 MB) — 133-keypoint pose estimator

### 4. Verify Installation

```bash
python scripts/setup_env.py

# Quick smoke check
python -c "import onnxruntime, cv2, numpy, scipy, pydantic; print('OK')"
```

## Running the CLI

```bash
# Analyze a video
python -m stride analyze --video path/to/test.mp4 --output results.json

# With pathological gait preset (tuned for Parkinson's)
python -m stride analyze --video path/to/test.mp4 --output results.json --preset pathological

# Verbose logging
python -m stride analyze --video path/to/test.mp4 --output results.json --loglevel DEBUG
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only (fast, no video I/O)
pytest tests/unit/ -v

# Integration tests (synthetic gait, no video I/O)
pytest tests/integration/ -v

# Specific test file
pytest tests/unit/test_quartile_engine.py -v

# With coverage report
pytest tests/ --cov=stride --cov-report=html
# Open htmlcov/index.html to view
```

## Linting & Type Checking

```bash
# Type checking
mypy src/stride/

# Format
black src/stride/ tests/

# Lint
ruff check src/stride/ tests/
ruff check --fix src/stride/ tests/
```

## API Server (Phase 3+)

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Performance Profiling

```bash
# Benchmark pose inference
python scripts/benchmark.py --video test.mp4
```

## Project Structure

```
Stride/
├── src/stride/               # Main package (src layout)
│   ├── core/                 # Domain primitives (enums, protocols, keypoints)
│   ├── config/               # StriderConfig (Pydantic, no side effects)
│   ├── data/                 # Data models (events, metrics, clinical, result)
│   ├── pipeline/             # Orchestrator (run_pass1, run_pass2, GaitProcessor)
│   ├── pose/                 # RTMPose ONNX + OneEuro smoother
│   ├── tracking/             # ByteTrack + Kalman
│   ├── calibration/          # Homography (manual + SVD auto)
│   ├── segmentation/         # Phase detection, quartile engine, turn detector
│   ├── gait_events/          # Foot strike detector, step validator
│   ├── metrics/              # Per-quartile metrics
│   ├── clinical/             # Clinical flags (Phase 2+)
│   └── export/               # CSV/JSON/PDF (Phase 2+)
├── api/                      # FastAPI server (Phase 3+)
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   └── synthetic_gait.py # Parametric gait generator (no video I/O)
│   ├── unit/                 # Fast unit tests
│   └── integration/          # End-to-end pipeline tests
├── scripts/
│   ├── download_models.py
│   ├── setup_env.py
│   └── benchmark.py
├── models/                   # Downloaded ONNX weights (git-ignored)
├── pyproject.toml
└── requirements.txt
```

## Adding Test Videos

Place test videos in `tests/data/`:

```
tests/data/
├── patient_001.mp4
├── healthy_001.mp4
└── ...
```

## Configuration

```python
from stride.config import StriderConfig, get_pathological_gait_config

# Default config (no filesystem side effects until explicit call)
config = StriderConfig()
config.ensure_directories()  # creates output/ and models/ dirs

# Pathological gait preset (tuned for Parkinson's, FOG, shuffling)
config = get_pathological_gait_config()
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'stride'`

Activate the venv and install in editable mode:

```bash
venv\Scripts\activate
pip install -e .
```

### `onnxruntime` not found

```bash
pip install onnxruntime>=1.16.0
```

### Pytest discovers no tests

Test files must be named `test_*.py` or `*_test.py`. Test functions must start with `test_`. Test classes must start with `Test`.

## Next Steps

See [ROADMAP.md](ROADMAP.md) for phase-by-phase implementation plan, and [HANDOFF.md](HANDOFF.md) for current status and the next task.
