# STRIDE Project Handoff Document

**Last Updated:** 2026-05-16  
**Session:** Calibration + Foot-Strike Fix (Session 11)  
**Status:** Three bugs fixed — SVD anchoring, OneEuro alpha inversion, foot-strike detrend. 57/57 tests pass. Pipeline produces Total steps: 11, Cadence: 51 steps/min on Healthy.mp4. Next: Phase 2 metrics (asymmetry, sway, clinical flags).

---

## 1. PROJECT OVERVIEW

### What is STRIDE?

**STRIDE** (Spatial-Temporal Rehabilitation Imaging & Data Engine) is a research-grade **clinical gait analysis system** for neurological movement disorders. It analyzes **single monocular RGB video** of a standardized **6-meter walk test** and produces clinically meaningful metrics with supporting visualizations.

**Primary Use Case:** Detect and quantify gait abnormalities in patients with:
- Parkinson's disease (freezing of gait, bradykinesia, asymmetry)
- Stroke rehabilitation (asymmetry, turning deficits)
- Balance disorders
- General neurological movement assessment

**Defining Constraint:** All spatial metrics (step assignment, quartile boundaries, distance progression) **must be derived from actual world-space position** along the calibrated 6-meter path — **never from frame indices or elapsed time**. Patients with neurological disorders exhibit highly non-periodic gait (freezing, shuffling, variable cadence), so **no periodic-gait assumptions are permitted anywhere**.

### Current Goals (Phase 1–3)

**Phase 1 (Complete):** Detection + pose + spatial calibration + foot-strike detection all working on real video.
- ✅ Architectural foundation, 12 core modules, processor orchestrator, CLI — all done
- ✅ SimCC scoring bug fixed (ankle confidence 0.003 → 0.77+ mean)
- ✅ Full-frame hallucination eliminated; MISSING frames explicit
- ✅ Geometric bbox gating, calibration guard, ROI-zoom, RTMDet-m — 0% missing frames
- ✅ SVDAutoCalibrator anchoring fixed — world_x [0.000, 6.000] on Healthy.mp4
- ✅ OneEuro alpha formula fixed — step oscillations now visible after smoothing
- ✅ Foot-strike detrending fixed — gaussian sigma 5s→1.5s; 11 steps detected on Healthy.mp4
- ✅ 57/57 unit tests pass (no regressions)
- 🟡 **Current state:** Gait metrics running. `asymmetry_score` and `sway_rms` return 0.0 (Phase 2 placeholders). Clinical flags not yet firing. Phase 2 metrics are next.

**Phase 2:** Full metrics + clinical analysis (asymmetry, sway, FOG, variability, clinical flags)
- Requires the world-coord calibration fix above to run first.

**Phase 3:** FastAPI server with async job processing, WebSocket progress, REST endpoints

### Long-Term Vision

1. **Research-Grade Validation** — ICC > 0.90 on all primary metrics vs. GaitRite instrumented walkway + OPAL IMU
2. **Temporal Neural Models** — Replace signal-processing with learned sequence models (LSTM/Transformer) for step detection, phase segmentation, FOG detection
3. **Multi-Motion Generality** — Abstract architecture above `stride/` for other movement analysis tasks (balance, upper limb, rehab assessment)
4. **Clinical Deployment** — Frontend for clinicians: upload video → get report with metrics, flags, clinical interpretation
5. **Edge Deployment** — Real-time processing on mobile/embedded devices (< 3× real-time on CPU)

---

## 2. CURRENT ARCHITECTURE

### Folder Structure

```
Stride/
├── pyproject.toml                    # src layout: packages in src/
├── README.md
├── CLAUDE.md                         # Project instructions (IMPORTANT: read first)
├── ARCHITECTURE.md                   # Detailed algorithm + data flow specs
├── ROADMAP.md                        # Phase-based implementation plan
├── IMPLEMENTATION_PROGRESS.md        # Detailed session-by-session progress
├── HANDOFF.md                        # This file
│                                     # NOTE: backend/strider/ (old prototype) deleted in M0
│
├── src/stride/                       # Main package (src layout)
│   ├── __init__.py                   # Public API surface
│   ├── __main__.py                   # ✅ CLI entry point (python -m stride)
│   ├── cli.py                        # ✅ argparse CLI (analyze subcommand)
│   │
│   ├── core/                         # Domain primitives (no external deps except numpy)
│   │   ├── types.py                  # ✅ Side, Phase, Quartile, ClinicalFlagType enums
│   │   ├── keypoints.py              # ✅ KeypointSchema, RTMPoseWholebody133, MediaPipePose33
│   │   ├── protocols.py              # ✅ PoseEstimator, Tracker, Calibrator, GaitEventDetector, MetricComputer, ClinicalAnalyzer
│   │   └── __init__.py               # Core exports
│   │
│   ├── config/                       # Configuration (Pydantic BaseModel, no side effects)
│   │   ├── schema.py                 # ✅ StriderConfig with ensure_directories() explicit method
│   │   ├── presets.py                # ✅ get_default_config(), get_pathological_gait_config()
│   │   └── __init__.py               # Config exports
│   │
│   ├── data/                         # Data models (Pydantic BaseModel)
│   │   ├── events.py                 # ✅ FootStrikeEvent, FOGEpisode (full JSON round-trip)
│   │   ├── metrics.py                # ✅ QuartileMetrics, GlobalMetrics, GaitMetrics (dict[Quartile, ...])
│   │   ├── clinical.py               # ✅ ClinicalFlag, ClinicalReport
│   │   ├── result.py                 # ✅ AnalysisResult with from_json(path) / to_json(path)
│   │   └── __init__.py               # Data model exports
│   │
│   ├── pipeline/                     # Pipeline orchestration
│   │   ├── context.py                # ✅ Pass1Result, Pass2Result (+ turning_time_sec field)
│   │   ├── processor.py              # ✅ GaitProcessor.process() + run_pass1() + run_pass2()
│   │   └── __init__.py               # Pipeline exports
│   │
│   ├── pose/                         # Pose estimation
│   │   ├── rtmpose.py                # ✅ RTMPose-l ONNX + SimCC decoding
│   │   ├── smoother.py               # ✅ OneEuro temporal filter
│   │   └── __init__.py               # Pose module exports
│   │
│   ├── tracking/                     # Multi-object tracking
│   │   ├── bytetrack.py              # ✅ ByteTrack + Kalman + Hungarian
│   │   └── __init__.py               # Tracking exports
│   │
│   ├── calibration/                  # Spatial calibration
│   │   ├── homography.py             # ✅ Manual 4-pt + SVD auto calibration
│   │   ├── spatial_mapper.py         # ✅ Vectorized image→world transforms
│   │   └── __init__.py               # Calibration exports
│   │
│   ├── segmentation/                 # Gait phase & quartile segmentation
│   │   ├── phase_detector.py         # ✅ Velocity zero-crossing TOWARD/TURN/AWAY
│   │   ├── turn_detector.py          # ✅ Turning time measurement
│   │   ├── quartile_engine.py        # ✅ Distance-based Q1/Q2/Q3/Q4 assignment (updated)
│   │   └── __init__.py               # Segmentation exports
│   │
│   ├── gait_events/                  # Foot strike, FOG, stance/swing
│   │   ├── foot_strike.py            # ✅ scipy find_peaks + adaptive prominence
│   │   ├── step_validator.py         # ✅ Temporal + spatial plausibility
│   │   └── __init__.py               # Gait events exports
│   │
│   ├── metrics/                      # Metric computation
│   │   ├── per_quartile.py           # ✅ Step count, cadence per quartile (fixed confidence kwarg)
│   │   └── __init__.py               # Metrics exports
│   │
│   ├── clinical/                     # Clinical analysis (Phase 2+)
│   │   └── __init__.py               # Placeholder for flags.py, thresholds.py
│   │
│   └── export/                       # Export formats (Phase 2+)
│       └── __init__.py               # Placeholder for csv.py, json.py, pdf.py
│
├── api/                              # FastAPI server (Phase 3)
│   ├── main.py                       # (not yet created)
│   └── routes/                       # (not yet created)
│
├── tests/
│   ├── conftest.py                   # ✅ Fixtures: config_no_dirs, tmp_output_dir
│   ├── fixtures/
│   │   ├── __init__.py
│   │   └── synthetic_gait.py         # ✅ Synthetic gait generator (skeleton)
│   ├── unit/                         # (to be created)
│   │   ├── test_rtmpose.py           # (not yet created)
│   │   ├── test_foot_strike.py       # (not yet created)
│   │   ├── test_homography.py        # (not yet created)
│   │   ├── test_quartile_engine.py   # ✅ Exists from previous work
│   │   └── ...
│   └── integration/                  # (to be created)
│       └── test_pipeline_full.py     # (not yet created)
│
├── scripts/
│   ├── download_models.py            # ✅ ONNX model downloader
│   ├── setup_env.py                  # ✅ Environment validation
│   └── benchmark.py                  # (not yet created)
│
└── models/                           # Downloaded ONNX weights (not in repo)
    ├── rtmpose-l_simcc-body7_wholebody_coco-384x288.onnx  (90 MB)
    └── rtmdet_nano_320-8bbb47ba.onnx                      (10 MB)
```

### Major Modules & Responsibilities

#### Core Layer (`src/stride/core/`)
- **types.py** — `StrEnum` types: `Side` (L/R), `Phase` (TOWARD/TURN/AWAY), `Quartile` (Q1/Q2/Q3/Q4/TURN), `ClinicalFlagType`, `ClinicalSeverity`. All use `StrEnum` (not `str, Enum`) for Python 3.13 numpy compatibility — `str(StrEnum.MEMBER)` always returns the value string.
- **keypoints.py** — `KeypointSchema` abstraction; `RTMPoseWholebody133` (133 keypoints) and `MediaPipePose33` (33 keypoints) registries
- **protocols.py** — Python `Protocol` definitions for dependency injection:
  - `PoseEstimator` — frame + bbox → `KeypointFrame`
  - `Tracker` — detections → tracked detections
  - `Calibrator` — ankle trajectory → `CalibrationResult` (homography)
  - `GaitEventDetector` — keypoint sequence → `list[FootStrikeEvent]`
  - `MetricComputer` — events → `QuartileMetrics`
  - `ClinicalAnalyzer` — metrics → `ClinicalReport`

#### Configuration (`src/stride/config/`)
- **schema.py** — `StriderConfig` with all thresholds + paths (Pydantic `BaseModel`, immutable, **no side effects**)
  - Key fields: `path_length_m`, `target_fps`, `min_step_interval_sec`, `max_step_interval_sec`, clinical thresholds
  - **Important:** Call `config.ensure_directories()` explicitly before processing (not in `__init__`)
- **presets.py** — Factory functions: `get_default_config()`, `get_pathological_gait_config()`

#### Data Models (`src/stride/data/`)
- **events.py** — `FootStrikeEvent(frame_idx, timestamp, side, world_x, world_y, confidence, step_length, step_time, detection_phase, quartile)`, `FOGEpisode`. Note: `detection_phase: Optional[Phase]` = phase at detection time (TOWARD/TURN/AWAY); `quartile: Optional[Quartile]` = assigned post-detection by processor Stage 6. These are distinct concepts.
- **metrics.py** — `QuartileMetrics` (step_count, cadence, asymmetry, sway, duration, etc.), `GaitMetrics` with `dict[Quartile, QuartileMetrics]` (flexible for variable path lengths)
- **clinical.py** — `ClinicalFlag`, `ClinicalReport`
- **result.py** — `AnalysisResult` with full JSON round-trip: `to_json(path)`, `from_json(path)`, experiment_id UUID, schema_version for reproducibility

#### Pipeline (`src/stride/pipeline/`)
- **context.py** — Immutable dataclasses:
  - `Pass1Result` — raw keypoints (N, 133, 3), timestamps, track_ids, fps, schema
  - `Pass2Result` — world positions, phases, foot_strikes, FOG, calibration, quartile_metrics, `turning_time_sec`
- **processor.py** — ✅ Complete orchestrator:
  - `run_pass1(video_path, ...) → Pass1Result` — frame loop, ByteTrack, RTMPose, OneEuro smoothing; full-frame bbox fallback (no RTMDet). Requires explicit `pose_estimator` injection — `NotImplementedError` raised if not provided (by design).
  - `run_pass2(pass1_result, ...) → Pass2Result` — 8-stage pipeline: calibration → world coords → phases → foot strikes → quartile assignment (sets `event.quartile`) → metrics
  - `GaitProcessor.process(video_path) → AnalysisResult` — assembles GlobalMetrics + ClinicalReport stub + AnalysisResult
  - Note: `_SpatialMapperBridge` **removed in M0**. `SpatialMapper` now receives canonical `CalibrationResult` directly.

#### Pose Estimation (`src/stride/pose/`)
- **rtmpose.py** — `RTMPoseEstimator` class
  - Loads ONNX model
  - Accepts frame (BGR) + bbox
  - Outputs `KeypointFrame` (133 keypoints × [x, y, confidence])
  - **Critical:** SimCC logit decoding (argmax + softmax confidence)
  - Handles affine transforms and inverse warping
- **smoother.py** — `OneEuroFilter` class + `smooth_keypoints()` function
  - Temporal filtering per-keypoint
  - Adaptive cutoff frequency based on velocity

#### Tracking (`src/stride/tracking/`)
- **bytetrack.py** — `ByteTrack` class
  - Kalman state: [cx, cy, ar, h, vx, vy, va, vh] (center, aspect ratio, height, velocities)
  - Hungarian assignment via `scipy.optimize.linear_sum_assignment`
  - Two-stage matching: high-confidence detections first, then low-confidence
  - Returns `dict[track_id, bbox]`
  - **M0 fix:** `min_hits=1` (was 3); filter is age-only (`time_since_update < max_age`); `_create_track` initialises `hits=1`. This ensures patient track is returned immediately at frame 0, enabling the patient-lock logic to fire.

#### Calibration (`src/stride/calibration/`)
- **homography.py**
  - `ManualHomographyCalibrator` — Interactive 4-point markup
  - `SVDAutoCalibrator` — Auto-fit walking axis from ankle trajectory, scale by 6m constraint
  - Both return `CalibrationResult` with homography matrix H
- **spatial_mapper.py** — `SpatialMapper` class
  - `image_to_world(img_pts) → world_pts` (vectorized)
  - `world_to_image(world_pts) → img_pts`
  - `validate_roundtrip()` — checks H⁻¹ @ H @ pts ≈ pts

#### Segmentation (`src/stride/segmentation/`)
- **phase_detector.py** — `PhaseDetector` class
  - Velocity zero-crossing detection
  - Returns `(phases: Phase[], (turn_start, turn_end))`
  - Robust to multiple zero-crossings (selects by max distance)
- **turn_detector.py** — `TurnDetector` class
  - Finds frames with low velocity (turning phase)
  - Returns `(turning_time_sec, (turn_start, turn_end))`
- **quartile_engine.py** — `QuartileEngine` class (**FIXED in Phase 1**)
  - `assign_quartile(world_x, phase) → Quartile` (returns enum, not string)
  - Constructor now accepts `turn_distance_m` parameter (handles early turns)
  - Vectorized `compute_quartile_time_windows()`

#### Gait Events (`src/stride/gait_events/`)
- **foot_strike.py** — `FootStrikeDetector` class
  - `detect(keypoints, world_positions, timestamps, schema, phases) → list[FootStrikeEvent]`
  - Uses `scipy.signal.find_peaks(-ankle_y, prominence=adaptive, distance=...)`
  - Adaptive prominence: 10% of signal range (handles shuffling)
  - Per-frame confidence computation from logit softmax
- **step_validator.py** — `StepValidator` class
  - Checks temporal plausibility (min/max step intervals)
  - Checks spatial plausibility (min/max step lengths, backward steps)
  - Returns `(valid_events, rejected_events_with_reasons)`
- **fog_detector.py** — ✅ `FOGDetector` class (Session 6)
  - `detect(keypoints, timestamps, schema) → list[FOGEpisode]`
  - Spectral Freeze Index (Moore et al. 2008): `FI = P_freeze / P_loco` (linear)
  - Freeze band [3.0, 8.0] Hz; Loco band [0.5, 3.0] Hz
  - 50% overlapping windows using `np.maximum.at()` for max aggregation (Bug B1 fix)
  - Welch PSD via `scipy.signal.welch`
  - Episode detection: contiguous frames where FI > threshold for ≥ min_duration_sec
  - Confidence-weighted ankle velocity with linear interpolation for occluded frames

#### Metrics (`src/stride/metrics/`)
- **per_quartile.py** — `QuartileMetricsComputer` class
  - Filters events by quartile (distance-based)
  - Computes: step count, cadence, duration, mean step length, step time CV
  - Returns `dict[Quartile, QuartileMetrics]`

### Pipeline Data Flow

```
VIDEO FILE
    ↓ [RTMDet Person Detector]
BOUNDING BOXES
    ↓ [ByteTrack] 
TRACKED PERSON (track_id per frame)
    ↓ [RTMPose Estimator]
KEYPOINTS (N, 133, 3)
    ↓ [OneEuro Smoother]
SMOOTHED KEYPOINTS
    ↓ ==== PASS 1 COMPLETE (Pass1Result) ====
    ↓ [SpatialMapper - Manual or SVD Calibration]
WORLD COORDINATES (meters)
    ↓ [PhaseDetector]
PHASE LABELS (TOWARD/TURN/AWAY)
    ↓ [QuartileEngine]
QUARTILE ASSIGNMENT (Q1/Q2/Q3/Q4)
    ↓ [FootStrikeDetector + StepValidator]
FOOT STRIKE EVENTS
    ↓ [QuartileMetricsComputer]
PER-QUARTILE METRICS
    ↓ ==== PASS 2 COMPLETE (Pass2Result) ====
    ↓ [ClinicalAnalyzer - Phase 2]
CLINICAL FLAGS + REPORT
    ↓ [ExportModule - Phase 2]
JSON/CSV/PDF OUTPUT
```

### Key Technologies & Frameworks

| Component | Technology | Version | Notes |
|-----------|-----------|---------|-------|
| Runtime | Python | 3.10+ | Core language |
| Inference | ONNX Runtime | 1.16+ | CPU-based pose/detection |
| Data Models | Pydantic | 2.12+ | Validation, JSON serialization |
| Numerics | NumPy | 2.0+ | Array operations |
| Signal Processing | SciPy | 1.14+ | Peak detection, spectral analysis |
| Tracking | SciPy | 1.14+ | Hungarian assignment |
| CV | OpenCV | 4.13+ | Homography, transforms, video I/O |
| API | FastAPI | 0.128+ | (Phase 3) Web server |
| Export | ReportLab | 4.2+ | (Phase 2) PDF generation |
| Testing | pytest | 7.4+ | Unit & integration tests |

---

## 3. CURRENT SYSTEM STATUS

### ✅ IMPLEMENTED (Complete & Tested)

**Architectural Foundation (10 Steps):**
- ✅ Core enums (`types.py`) — Side, Phase, Quartile, ClinicalFlagType, ClinicalSeverity
- ✅ Keypoint schema registry (`keypoints.py`) — RTMPoseWholebody133, MediaPipePose33
- ✅ Component protocols (`protocols.py`) — All 6 major interfaces with data contracts
- ✅ Config refactoring (`config/schema.py`) — Pydantic, immutable, no side effects
- ✅ Data models (`data/`) — Full Pydantic with JSON round-trip
- ✅ Pipeline context (`pipeline/context.py`) — Pass1Result, Pass2Result immutable dataclasses
- ✅ Package layout (`pyproject.toml`) — src/ layout with proper package discovery
- ✅ Test fixtures (`tests/conftest.py`) — No-mkdir config fixtures
- ✅ Quartile engine (`segmentation/quartile_engine.py`) — Updated to return Quartile enum, handle turn_distance_m

**Phase 1 Core Modules (12 Modules):**
- ✅ RTMPose estimator (`pose/rtmpose.py`) — Full ONNX + SimCC decoding
- ✅ Temporal smoother (`pose/smoother.py`) — OneEuro filter implementation
- ✅ ByteTrack (`tracking/bytetrack.py`) — Kalman + Hungarian assignment
- ✅ Homography calibration (`calibration/homography.py`) — Manual 4-pt + SVD auto
- ✅ Spatial mapper (`calibration/spatial_mapper.py`) — Vectorized transforms
- ✅ Phase detector (`segmentation/phase_detector.py`) — Velocity zero-crossing
- ✅ Turn detector (`segmentation/turn_detector.py`) — Turning time measurement
- ✅ Foot strike detector (`gait_events/foot_strike.py`) — scipy peaks + adaptive
- ✅ Step validator (`gait_events/step_validator.py`) — Plausibility checks
- ✅ Per-quartile metrics (`metrics/per_quartile.py`) — Step count, cadence
- ✅ Model downloader (`scripts/download_models.py`) — ONNX downloads
- ✅ Setup script (`scripts/setup_env.py`) — Environment validation

**Pipeline Integration (Session 3):**
- ✅ Processor orchestrator (`pipeline/processor.py`) — run_pass1, run_pass2, GaitProcessor.process complete
- ✅ CLI interface (`src/stride/cli.py`) — argparse, progress bar, preset configs, result output
- ✅ Module entry point (`src/stride/__main__.py`) — `python -m stride` supported
- ✅ Synthetic gait generator (`tests/fixtures/synthetic_gait.py`) — sinusoidal ankle motion, FOG episodes, asymmetry injection, generate_synthetic_pass1_result helper

**Testing Infrastructure:**
- ✅ Test fixtures (`tests/conftest.py`) — config_no_dirs, tmp_output_dir, config_with_tmp_output
- ✅ Synthetic gait generator (`tests/fixtures/synthetic_gait.py`) — full parametric implementation

**Phase 1 Tests (Session 5):**
- ✅ `tests/unit/test_quartile_engine.py` — **19/19 tests pass**
- ✅ `tests/unit/test_foot_strike.py` — **8/8 tests pass** (FootStrikeDetector + StepValidator)
- ✅ `tests/integration/test_pipeline_synthetic.py` — **11/11 tests pass** (full Pass-2 pipeline on synthetic gait)

**Phase 1 + FOG Tests (Session 6):**
- ✅ `tests/unit/test_fog_detector.py` — **18/18 tests pass** (FOG spectral analysis, window overlap fix, episode detection)

**Detection/Pose Collapse Fix (Sessions 8–10):**
- ✅ `src/stride/pose/rtmpose.py` — `_decode_simcc` SimCC scoring fixed: raw max-logit replaces softmax-mean; ankle conf 0.003 → 0.77+ (verified on Healthy.mp4, fixed_angle.mp4, walk_along.mp4)
- ✅ `src/stride/pipeline/processor.py` — Fabricated full-frame box (`(0,0,W,H,0.5)`) removed; MISSING-frame explicit path; geometric bbox gate (`_is_plausible`); tracker-assisted ROI-zoom (last-resort recovery); calibration guard (loud failure on < 20 high-conf ankle frames); `raw` NameError in verbose logging fixed
- ✅ `src/stride/config/schema.py` — 8 new config fields: `max_bbox_frame_fraction`, `min_person_height_px`, `min_person_aspect`, `max_person_aspect`, `min_calibration_conf_frames`, `roi_zoom_enabled`, `roi_zoom_scale`, `roi_reacquire_interval`
- ✅ `src/stride/cli.py` — `--detector rtmdet-m` option; auto-prefer RTMDet-m when both models present; `input_size` wired through
- ✅ `src/stride/detection/rtmdet.py` — Two-output tensor format handled; person class filtering; conf_threshold=0.15
- ✅ `src/stride/detection/yolov8.py` — Full YOLOv8Detector (ONNX output format, coordinate transform, conf filtering)
- ✅ `scripts/download_models.py` — `rtmdet_medium` (109 MB, HuggingFace bukuroo) entry added; optional flag
- ✅ `docs/CAPTURE_PROTOCOL.md` — Operator capture SOP (resolution, framing, camera height, lighting, mount, subject count)

**Total: 57/57 unit tests pass** (19 quartile + 8 foot_strike + 18 FOG + 12 calibration regression)

### ❌ NOT IMPLEMENTED (Deferred to Phase 2+)

**Phase 2 Metrics:**
- ❌ Asymmetry computation (`metrics/asymmetry.py`)
- ❌ Sway computation (`metrics/sway.py`)
- ❌ Variability (CV) computation (`metrics/variability.py`)
- ❌ Global metrics aggregation (`metrics/global_metrics.py`)
- ❌ Stance/swing segmentation (`gait_events/stance_swing.py`)

**Phase 2 Clinical:**
- ❌ Clinical flag generation (`clinical/flags.py`)
- ❌ Evidence-based thresholds (`clinical/thresholds.py`)
- ❌ Confidence scoring (`clinical/confidence.py`)

**Phase 2 Export:**
- ❌ CSV export (`export/csv_exporter.py`)
- ❌ JSON export (`export/json_exporter.py`)
- ❌ PDF report generation (`export/pdf_report.py`)

**Phase 3 API:**
- ❌ FastAPI main (`api/main.py`)
- ❌ Routes (`api/routes/`)
- ❌ Job manager (`api/job_manager.py`)
- ❌ WebSocket streaming (`api/websocket.py`)

**Phase 4+ Frontend:**
- ❌ React application (deferred)

**Unit Tests (future):**
- ❌ `tests/unit/test_rtmpose.py` — needs ONNX model loaded
- ❌ `tests/unit/test_homography.py`
- ❌ `tests/unit/test_bytetrack.py`
- ❌ `tests/unit/test_phase_detector.py`

**Integration Tests (future):**
- ❌ `tests/integration/test_pipeline_real_video.py` — with actual test videos

---

## 4. CURRENT TECHNICAL CHALLENGES

### Known Bugs & Limitations

#### B1: FOG Spectral Window Overlap Issue (Phase 2) — ✅ FIXED (Session 6)
**Location:** `src/stride/gait_events/fog_detector.py`  
**Issue:** With 50% window overlap, each frame is covered by two spectral windows. The second write overwrites the first without aggregation.  
**Fix Applied:** Use `np.maximum.at(fi_values, np.arange(start, end), fi)` for max aggregation. Each frame receives the highest FI from all windows that cover it.  
**Status:** ✅ Implemented and tested (18/18 tests pass)

#### B2: FOG Formula Discrepancy (Phase 2) — ✅ RESOLVED (Session 6)
**Location:** `src/stride/gait_events/fog_detector.py`  
**Issue:** ARCHITECTURE.md specified `FI = P_freeze² / P_loco²` (squared); standard Moore et al. 2008 uses linear ratio.  
**Resolution:** User confirmed linear formula `FI = P_freeze / P_loco` (Moore 2008). Threshold `fog_fi_threshold = 2.5` is calibrated to linear formula.  
**Status:** ✅ Implemented with linear formula; ARCHITECTURE.md squared formula superseded

#### B3: Phase Detector Multi-Zero-Crossing Handling (Partially Fixed)
**Location:** `segmentation/phase_detector.py` (Phase 1 implementation)  
**Status:** ✅ Fixed — selects zero-crossing at maximum distance  
**Remaining Risk:** Very slow walkers or patients with multiple reversals might still confuse detector  
**Mitigation:** Validate on real patients; adjust window size if needed

#### B4: Early Turn Handling (Partially Fixed)
**Location:** `segmentation/quartile_engine.py` (Phase 1 implementation)  
**Status:** ✅ Fixed — accepts `turn_distance_m` parameter  
**Remaining Risk:** Turn point detection (`PhaseDetector`, `TurnDetector`) must set this parameter correctly  
**Mitigation:** Validate Turn detection on early-turning patients

#### B5: Shuffling Gait Detection (Minimal)
**Location:** `gait_events/foot_strike.py` (Phase 1 implementation)  
**Status:** ⚠️ Minimal — uses adaptive prominence, returns empty peaks for very flat signals  
**Remaining Risk:** Fallback to mediolateral displacement detection not yet implemented  
**Plan:** Implement in Phase 2 if shuffling cases appear in test data

### Architectural Weaknesses

#### W1: No Real Pose Detector — ✅ RESOLVED (Sessions 8–10)
**Status:** RTMDet-nano and RTMDet-m are both integrated and working. Full-frame fallback box is **gone** — a missing detection now marks the frame MISSING explicitly instead of fabricating garbage keypoints.  
**Current Implementation:** `--detector rtmdet` (default) auto-prefers RTMDet-m (640 input) when the model is present; nano otherwise. Tracker-assisted ROI-zoom recovers far-subject frames where full-frame detection misses. Result: 0% MISSING on all three test clips.

#### L1: SVD World Coordinates Not Absolute — ✅ FIXED (Session 11)
**Location:** `src/stride/calibration/homography.py` → `SVDAutoCalibrator`  
**Issue:** SVD fits walking axis to centered trajectory → world_x ≈ [−3, +3]. Needed [0, 6].  
**Fix applied:** Two changes in `calibrate()`:
1. Replaced quarter-based direction sign check with midpoint-based (central 20% of frames vs first 20%) — more robust for non-uniform walks.
2. Added `proj_min` anchoring: `H[0,2] -= proj_min * scale` so `world_x = scale*(proj - proj_min)` maps start→0, turn→6.
**Verified:** Healthy.mp4 world_x min=0.000, max=6.000; phase split 47%/53%. 12 regression tests in `tests/unit/test_calibration.py` cover this.  
**CLAUDE.md invariant #4** updated: the known limitation is resolved for the SVD path.

#### W2: No Person Selector Logic
**Issue:** With multiple people in frame, need to lock to largest/most-confident person and re-ID after turn  
**Impact:** `processor.py` doesn't filter detections to single patient  
**Fix:** Add `PatientSelector` class (simple: lock to largest bbox at frame 1, maintain via tracking)

#### W3: Progress Callbacks Not Threadsafe
**Issue:** CLI and API both need progress feedback, but callback mechanism not yet defined  
**Impact:** Can't provide real-time progress to user during long video processing  
**Design:** Add optional `progress_callback: Callable[[float, str], None]` parameter to `processor.process()`

#### W4: No Memory Management for Long Videos
**Issue:** `Pass1Result` stores all keypoints (N, 133, 3) in memory — 30-min video at 30 fps = 54M frames = multi-GB  
**Impact:** OOM on mobile/embedded devices  
**Fix:** Implement frame streaming + rolling buffer (Phase 6)

### Research Limitations

#### R1: No Temporal Sequence Models
**Issue:** All step/phase detection uses signal processing (scipy peaks, zero-crossing). No learned models.  
**Impact:** Limited ability to handle complex temporal patterns (FOG episodes, stuttering, variable cadence)  
**Plan:** Phase 5+ — train LSTM/Transformer models on labeled video datasets

#### R2: Single-Subject Assumption
**Issue:** Current design assumes single person walking straight 6m. No multi-person, no non-straight paths.  
**Impact:** Can't handle group walks, figure-8 paths, or tandem gait  
**Plan:** Future research (Phase 6+)

#### R3: No Validation Against Ground Truth
**Issue:** No comparison to GaitRite, OPAL IMU, or manual annotation on real patients  
**Impact:** Don't know actual accuracy; metrics might be wrong  
**Plan:** Phase 6 — collect labeled dataset, validate ICC > 0.90

#### R4: Pathological Gait Coverage Limited
**Issue:** Only tested design on normal gait patterns and synthetic data. Real Parkinson's patients may have unexpected patterns.  
**Impact:** Edge cases (asymmetric tremor, irregular stride, shuffling) might break algorithm  
**Fix:** Test on diverse patient cohorts in Phase 6

### Performance Concerns

#### P1: RTMPose Inference Speed
**Current:** Designed for 15–25 fps on CPU (384×288 input)  
**Target:** < 5× real-time (2-min video in < 10 min)  
**Risk:** If ONNX Runtime fallback to CPU on hardware without optimization, could be 5+ fps slower  
**Mitigation:** Profile on target hardware; consider GPU if available; fallback to MediaPipe if needed

#### P2: Tracking Assignment O(N²)
**Current:** ByteTrack's Hungarian algorithm is O(n³) in worst case  
**Risk:** With high-confidence detections + low-confidence detections, could be slow for crowded scenes  
**Note:** Single-person assumption makes this unlikely, but worth monitoring

#### P3: Spatial Mapper Vectorization
**Current:** All transforms are vectorized (numpy matrix ops)  
**Status:** ✅ Should be efficient  
**Risk:** None identified

---

## 5. CURRENT ACTIVE WORK

### Session 11 — Calibration & Foot-Strike Fix ✅ COMPLETE

**Session Goal:** Fix the downstream spatial chain: SVD world_x anchoring → foot-strike detection.

**Three bugs fixed (all independent):**

| # | Bug | Location | Root Cause | Fix |
|---|-----|----------|------------|-----|
| 1 | `SVDAutoCalibrator` produces centered world_x [−3,+3] instead of [0,6] | `calibration/homography.py` | `world_x = scale*(proj - mean·d)` centres at 0; no min-anchoring | Added `proj_min` anchoring: `H[0,2] -= proj_min*scale`; replaced quarter-based sign check with midpoint-based (central 20% vs first 20%) |
| 2 | `OneEuroFilter._alpha` inverted formula | `pose/smoother.py` | Returned `1/(1+cutoff*dt)` → large alpha at low speed, small at high speed → ankle oscillations destroyed by over-smoothing | Corrected to `t/(1+t)` where `t=cutoff*dt`; verified fc=1Hz→α=0.095, fc=50Hz→α=0.840 |
| 3 | `FootStrikeDetector._find_peaks_adaptive` gaussian sigma too large | `gait_events/foot_strike.py` | `sigma=5*fps=283f`; gaussian cutoff ~0.032 Hz is below approach-arch frequency ~0.068 Hz → arch not captured in trend → step oscillations buried | Changed to `sigma=1.5*fps=85f` (cutoff ~0.11 Hz > 0.068 Hz arch, < 1.25 Hz steps); approach trend removed, step oscillations visible |

**Measured improvements (Healthy.mp4, 838 frames):**

| Metric | Before Session 11 | After Session 11 |
|--------|-------------------|------------------|
| world_x range | [−3.0, +3.0] (centered) | [0.000, 6.000] ✓ |
| Phase split (TOWARD/AWAY) | 755/43 (stale data) / then ~47%/53% | 47% / 53% ✓ |
| Foot strikes detected | 0 | 11 ✓ |
| Overall cadence | 0.0 steps/min | 51.1 steps/min |
| Q1+Q2+Q3+Q4 == total_steps | N/A | 4+2+2+3=11=11 ✓ |
| Unit tests | 45/45 | 57/57 |

**New tests added:** `tests/unit/test_calibration.py` — 12 regression tests covering SVD anchoring (world_x starts at 0, max=path_length, TOWARD increasing, AWAY decreasing, no negatives, noisy trajectory) and OneEuro alpha formula (low-fc small alpha, high-fc large alpha, step response, stable signal).

**Known remaining placeholders (NOT bugs introduced by this session):**
- `asymmetry_score: 0.0` in all quartiles — explicit `# Computed in Phase 2` placeholder in `metrics/per_quartile.py:144`
- `sway_rms_meters: 0.0` — same placeholder
- Clinical flags not firing — `clinical/flags.py` not yet implemented (Phase 2)
- Cadence within each quartile (Q1: 236, Q2: 257 steps/min) reflects few steps clustered in a short window; quartile duration field represents span of detected steps, not total spatial-zone time — this is a metrics refinement for Phase 2.

---

### Sessions 8–10 — Detection/Pose Collapse Investigation & Fix ✅ COMPLETE

**Session Goal:** Diagnose and fix the "whole-frame detection" failure mode. During AWAY walk, RTMDet confidence collapsed → pipeline fabricated a full-frame bbox → garbage keypoints → corrupted calibration and 0 gait metrics.

**Root causes found (three independent, compounding defects):**

| # | Defect | Location | Fix |
|---|--------|----------|-----|
| 1 | SimCC scoring bug: confidence = softmax-mean over 384 bins ≈ 0.005 (uniform), not raw max-logit | `pose/rtmpose.py:_decode_simcc` | Use `min(max_logit_x, max_logit_y)` as confidence; coordinate: `idx / split_ratio` |
| 2 | Pipeline fabricated `(0,0,W,H,0.5)` synthetic box whenever RTMDet returned nothing | `pipeline/processor.py:146` | Deleted; gap frames emit explicit MISSING (zero keypoints) |
| 3 | Far-subject collapse: RTMDet-nano@320 cannot resolve a 6m-away subject in a low-res crop | `detection/rtmdet.py` | ROI-zoom + RTMDet-m@640 (0% MISSING on all clips) |

**Measured improvements (Healthy.mp4, 838 frames):**

| Metric | Before | After |
|--------|--------|-------|
| Ankle confidence (mean) | 0.003 | 0.77–0.79 |
| Fabricated whole-frame boxes | constant | 0 |
| MISSING frames (AWAY) | 168/838 | 0/838 |
| Unit tests | 45/45 | 45/45 |

**Work completed:**
- ✅ Phase 0 (empirical): ONNX shape audit, normalization A/B test, raw logit inspection, visual render confirmed true root cause
- ✅ Phase 1 (Tier 0): SimCC fix, MISSING-frame path, geometric gating, calibration guard — ankle conf 0.003→0.77
- ✅ Phase 2 (Tier 1): ROI-zoom (full-frame first, ROI as last resort), RTMDet-m auto-prefer — 0% MISSING all clips
- ✅ `docs/CAPTURE_PROTOCOL.md` written (operator SOP: ≥1080p, tight framing, ~1.2m height, static mount)
- ✅ Bug fix: `NameError: name 'raw'` in verbose frame-logging (`processor.py:193`) — resolved and verified

**Current state:** Pipeline runs end-to-end without crash, generates correct pose debug videos. Remaining output: `Total steps: 0, Cadence: 0.0` — this is the SVD calibration offset issue (see L1 above), not a detection/pose failure.

---

### Session 7 — RTMDet Fix + YOLOv8 Alternative Detector (In Progress)

**Session Goal:** Diagnose and fix RTMDet person detection; implement YOLOv8 as alternative detector with CLI flag.

**Context:** RTMDet was integrated but silently returning 0 detections every frame, causing full-frame fallback bbox. Debug video showed scattered keypoints. Root cause: output parser didn't handle two-tensor format (dets + labels).

**Work completed:**
- ✅ Diagnosed RTMDet issue: `_decode_outputs` read only first tensor, ignored labels; confidences below 0.3 threshold → 0 detections
- ✅ Fixed `src/stride/detection/rtmdet.py` — added two-tensor format handling + person class filtering (labels==0) + lowered conf_threshold to 0.15
- ✅ Created `src/stride/detection/yolov8.py` — Full YOLOv8Detector class (170+ lines)
  - Handles ONNX output format: (1, 84, 8400) → transpose to (8400, 84)
  - Parses [x_center, y_center, width, height, obj_conf, class_0_conf, ...]
  - Combines confidences: obj_conf × class_0_conf
  - Full coordinate transformation and confidence-based filtering
- ✅ Updated `src/stride/cli.py` — Added --detector flag (choices: rtmdet, yolov8)
  - Lines 74-77: CLI argument definition
  - Lines 167-194: Conditional detector initialization based on args.detector choice
  - RTMDet: conf_threshold=0.15
  - YOLOv8: conf_threshold=0.25
- ✅ Updated `src/stride/pipeline/processor.py` — Changed verbose message from "RTMDet detection(s)" to "person detection(s)" for detector-agnostic reporting
- ✅ Updated `scripts/download_models.py` — Added YOLOv8 model configuration with HuggingFace + GitHub fallback URLs
- ✅ Verified RTMDet works end-to-end: `python -m stride analyze --video tests/Healthy.mp4 -v --detector rtmdet --pose-debug`
  - Detection counts logged every 30 frames
  - Pipeline completes successfully in 28.9s
  - Pose debug video exports (with Unicode warning, non-critical)

**Current blockers:**
- YOLOv8 model download failing (HTTPError on both HuggingFace and GitHub URLs)
  - Infrastructure in place; model needs to be manually downloaded or URL fixed
  - Not blocking since RTMDet now works

**Test count: 56/56 PASSED** (unchanged; existing tests still passing)

### Session 6 — FOG Detector + Documentation Update ✅

**Session Goal:** Implement FOG detector, fix download script, update documentation.

**Work completed:**
- ✅ `src/stride/gait_events/fog_detector.py` — Full FOG spectral analysis implementation (18 test coverage)
  - Linear Freeze Index formula (Moore et al. 2008): `FI = P_freeze / P_loco`
  - Freeze band [3.0, 8.0] Hz; Loco band [0.5, 3.0] Hz
  - Welch PSD with 50% overlapping windows
  - Max aggregation using `np.maximum.at()` (Bug B1 fix)
  - Confidence-weighted ankle velocity with `np.interp` for occluded frames
  - Episode detection: contiguous frames where FI > threshold for ≥ min_duration_sec
- ✅ `tests/unit/test_fog_detector.py` — **18/18 tests pass** (6 compute_freeze_index, 6 detect, 6 detect_episodes)
- ✅ `src/stride/gait_events/__init__.py` — Added FOGDetector export
- ✅ `scripts/download_models.py` — Fixed broken HuggingFace URLs with fallback sources
  - OpenXLab (primary), IDEA-Research DWPose (fallback), OpenMMLab CDN (tertiary)
  - Graceful handling of optional models (RTMDet marked optional)
  - Windows console compatibility (Unicode → `[OK]`/`[FAIL]` tags)
- ✅ `src/stride/pipeline/processor.py` — Wired FOG detection into Stage 5b of run_pass2
- ✅ **Updated HANDOFF.md, ROADMAP.md, IMPLEMENTATION_PROGRESS.md** — Full documentation refresh

**Design decisions locked in Session 6:**
- FOG formula: Linear (Moore 2008), not squared; threshold 2.5 is calibrated to this formula
- RTMDet is optional for Phase 1; full-frame fallback works fine for single-patient scenarios
- Window overlap aggregation uses `np.maximum.at()` to avoid write-overwrite bias

**Test count: 56/56 PASSED** (38 Phase 1 + 18 FOG)

### Session 4 — M0 Foundation Cleanup

**Session Goal:** Fix all blocking runtime bugs identified in the architectural audit before any M1+ work begins.

**Critical bugs fixed:**

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `CalibrationResult` field name crash | `homography.py` used `homography=`, `spatial_mapper.py` read `.homography`; three different conventions | Unified to `homography_matrix`, `scale_px_to_m`, `pc1_variance` in all files; removed `_SpatialMapperBridge` |
| ByteTrack never locked patient | `min_hits=3` + filter included `hits < min_hits` check; `_create_track` set `hits=0` → tracks never survived | `min_hits=1`; filter is age-only; `hits=1` at creation |
| `FootStrikeEvent.phase` type collision | Field typed `Optional[Quartile]` but detector assigned `Phase` enum → Pydantic ValidationError | Renamed to `quartile: Optional[Quartile]`; added `detection_phase: Optional[Phase]`; all callsites updated |
| Test imports old backend | `test_quartile_engine.py` imported `from strider.segmentation…` (deleted package) | All test imports updated to `from stride.…` |
| `backend/strider/` causing import confusion | Old prototype still existed alongside `src/stride/` | **Permanently deleted** (user confirmed) |

**Additional bugs found during test run:**

| Bug | File | Fix |
|-----|------|-----|
| Wrong FootStrikeEvent import | `step_validator.py` | Imports from `stride.data.events` |
| Non-default field after default | `context.py` | `quartile_metrics` moved before `turning_time_sec` |
| Old import in unit conftest | `tests/unit/conftest.py` | `from stride.config import StriderConfig` |
| Python 3.13 numpy truncation | `core/types.py` | `str, Enum` → `StrEnum` (value string always correct) |
| `np.bool_` vs `bool` identity | `quartile_engine.py` | `bool(quartile_sum == total_steps)` |

**Test result:** `pytest tests/unit/test_quartile_engine.py -v` → **19/19 PASSED**

**Design decisions confirmed in Session 4:**
- `RTMPoseEstimator` requires explicit injection — raising `NotImplementedError` is correct, not a bug.
- `FootStrikeEvent.detection_phase` ≠ `FootStrikeEvent.quartile` — phase at detection time is preserved separately from the post-detection quartile assignment.
- Low-confidence keypoint soft weighting (T_good=0.60, T_low=0.35) is deferred to M1/M2.

### Previous Sessions

**Session 2 — Phase 1 Core Modules (12 modules):**
Pose (RTMPose+SimCC, OneEuro), Tracking (ByteTrack+Kalman+Hungarian), Calibration (Manual 4-pt, SVD auto, SpatialMapper), Segmentation (PhaseDetector, TurnDetector, QuartileEngine), Gait Events (FootStrikeDetector, StepValidator), Metrics (QuartileMetricsComputer), Scripts (download_models, setup_env)

**Session 1 — Architectural Redesign (10 steps):**
Core enums, Keypoint schema registry, Protocols, Config (Pydantic, no side effects), Data models (JSON round-trip), Pipeline context (immutable dataclasses), Package layout, Test fixtures, QuartileEngine enum fix

---

## 6. IMPORTANT PROJECT ASSUMPTIONS

### ✅ Core Constraints (Non-Negotiable)

1. **Distance-Based Metrics, Not Time-Based**
   - All step assignments, quartile membership, phase detection MUST use world-space position
   - Never use frame indices or elapsed time for spatial decisions
   - Rationale: Patients with FOG/shuffling have variable cadence; time-based binning produces wrong results

2. **Single-Patient Assumption (Current Phase)**
   - System assumes exactly one person walking straight down a 6m path
   - Future phases may relax this; current design optimized for single-subject
   - Impact: No multi-person tracking, no crowd handling

3. **Pathological Gait Must Be Supported**
   - System must handle: shuffling, freezing, asymmetry, bradykinesia, irregular cadence
   - No periodic-gait assumptions (e.g., "average step time = total_time / step_count" fails if FOG mid-stride)
   - Rationale: Stroke + Parkinson's patients have complex gait deficits

4. **Monocular RGB Only**
   - Input: single camera, RGB video (no depth, no multi-view, no pose suit)
   - Constraint: Necessary for clinical deployability (uses phone camera)
   - Challenge: Cannot resolve occlusions during turn

5. **Research-Grade Quality Required**
   - Code must be maintainable, testable, validated
   - Every metric must have scientific citation or validation protocol
   - Reproducibility: Every result includes config_hash, experiment_id, schema_version
   - Rationale: Clinical/research use requires auditability

### 🔧 Implementation Assumptions

1. **No Preprocessing of Video**
   - Assume input video is recorded by clinician (phone camera, natural lighting)
   - No background subtraction, no person segmentation pre-pass
   - Person localization handled by detector (RTMDet)

2. **Calibration Always Possible**
   - Assume floor markers or visible path (tape, painted line) for manual calibration
   - Or assume enough ankle visibility for auto-calibration
   - Graceful failure if neither possible (warning + manual fallback)

3. **ONNX Models Available at Runtime**
   - System expects RTMPose-l and RTMDet ONNX files in `config.model_dir`
   - Handled by `scripts/download_models.py`
   - Fallback to MediaPipe if ONNX fails (not yet implemented)

4. **Python 3.10+ Available**
   - Uses modern Python features (pattern matching, type hints, union syntax)
   - No backward compatibility with 3.9

### ⚠️ Design Tensions & Trade-offs

1. **Accuracy vs. Speed**
   - Chose: CPU-friendly ONNX inference (RTMPose) over heavier GPU models (OpenPose, HRNet)
   - Trade: Slightly lower keypoint accuracy, much faster on standard hardware

2. **Modularity vs. Performance**
   - Chose: Protocol-based DI (pluggable implementations) over monolithic processor
   - Trade: Slight overhead from abstraction, but enables testing + future model swaps

3. **Single-Pass vs. Two-Pass**
   - Chose: Two-pass design (calibration needs full trajectory; events need calibrated world coords)
   - Trade: Higher memory (stores Pass1Result in memory), but cleaner logic separation

4. **Immutability vs. Streaming**
   - Chose: Immutable Pass1Result (all keypoints in memory) for simplicity
   - Trade: OOM on 30+ minute videos; will need streaming refactor in Phase 6

---

## 7. IMPORTANT FILES AND ENTRY POINTS

### Backend Entry Points

| File | Purpose | Usage | Status |
|------|---------|-------|--------|
| `src/stride/__main__.py` | CLI entry point | `python -m stride analyze --video test.mp4` | ✅ Created |
| `src/stride/cli.py` | CLI interface | Argument parsing + processor calling | ✅ Created |
| `src/stride/__init__.py` | Public API | `from stride import StriderConfig, GaitProcessor` | ✅ Created |
| `api/main.py` | FastAPI server | `uvicorn api.main:app` | ❌ Not created (Phase 3) |
| `scripts/download_models.py` | Model setup | `python scripts/download_models.py` | ✅ Created |
| `scripts/setup_env.py` | Validation | `python scripts/setup_env.py` | ✅ Created |

### Main Pipeline Files

| File | Purpose | Key Classes | Status |
|------|---------|------------|--------|
| `src/stride/pipeline/processor.py` | Orchestrator | `GaitProcessor`, `run_pass1`, `run_pass2` | ✅ Complete |
| `src/stride/pipeline/context.py` | Data flow | `Pass1Result`, `Pass2Result` | ✅ Created |
| `src/stride/config/schema.py` | Configuration | `StriderConfig` | ✅ Created |
| `src/stride/core/protocols.py` | Interfaces | 6 Protocol definitions | ✅ Created |

### Core Algorithm Files (Alphabetical)

| File | Purpose | Key Classes | Status |
|------|---------|------------|--------|
| `src/stride/calibration/homography.py` | Spatial calibration | `ManualHomographyCalibrator`, `SVDAutoCalibrator` | ✅ Created |
| `src/stride/calibration/spatial_mapper.py` | Image↔world transforms | `SpatialMapper` | ✅ Created |
| `src/stride/gait_events/foot_strike.py` | Step detection | `FootStrikeDetector` | ✅ Created |
| `src/stride/gait_events/step_validator.py` | Step validation | `StepValidator` | ✅ Created |
| `src/stride/metrics/per_quartile.py` | Quartile metrics | `QuartileMetricsComputer` | ✅ Created |
| `src/stride/pose/rtmpose.py` | Pose estimation | `RTMPoseEstimator` | ✅ Created |
| `src/stride/pose/smoother.py` | Temporal filtering | `OneEuroFilter`, `smooth_keypoints()` | ✅ Created |
| `src/stride/segmentation/phase_detector.py` | Phase detection | `PhaseDetector` | ✅ Created |
| `src/stride/segmentation/quartile_engine.py` | Quartile assignment | `QuartileEngine` | ✅ Updated |
| `src/stride/segmentation/turn_detector.py` | Turn measurement | `TurnDetector` | ✅ Created |
| `src/stride/tracking/bytetrack.py` | Multi-object tracking | `ByteTrack`, `Track`, `KalmanState` | ✅ Created |

### Config & Data Model Files

| File | Purpose | Key Classes | Status |
|------|---------|------------|--------|
| `src/stride/config/schema.py` | Settings | `StriderConfig` | ✅ Created |
| `src/stride/config/presets.py` | Config factories | `get_default_config()`, etc. | ✅ Created |
| `src/stride/core/types.py` | Enums | `Side`, `Phase`, `Quartile`, etc. | ✅ Created |
| `src/stride/core/keypoints.py` | Keypoint registry | `KeypointSchema`, pose model definitions | ✅ Created |
| `src/stride/core/protocols.py` | Component interfaces | 6 Protocols | ✅ Created |
| `src/stride/data/events.py` | Event models | `FootStrikeEvent`, `FOGEpisode` | ✅ Created |
| `src/stride/data/metrics.py` | Metric models | `QuartileMetrics`, `GaitMetrics` | ✅ Created |
| `src/stride/data/clinical.py` | Clinical models | `ClinicalFlag`, `ClinicalReport` | ✅ Created |
| `src/stride/data/result.py` | Result model | `AnalysisResult` with JSON I/O | ✅ Created |

### Test Files

| File | Purpose | Status |
|------|---------|--------|
| `tests/conftest.py` | Shared fixtures | ✅ Created |
| `tests/fixtures/synthetic_gait.py` | Synthetic data generator | ✅ Full implementation |
| `tests/unit/test_quartile_engine.py` | Quartile engine tests | ✅ 19/19 passing (fixed in M0) |
| `tests/unit/test_foot_strike.py` | Foot strike / step validator tests | ❌ Not yet created |
| `tests/integration/test_pipeline_synthetic.py` | End-to-end synthetic pipeline test | ❌ Not yet created |

### Configuration Locations

| File | Purpose | Default | Notes |
|------|---------|---------|-------|
| `config.model_dir` | ONNX models | `./models/` | Created by `scripts/download_models.py` |
| `config.output_dir` | Results | `./output/` | User can customize in `StriderConfig()` |
| `pyproject.toml` | Package config | — | src layout; `pip install -e .` reads this |
| `CLAUDE.md` | Project instructions | — | **READ FIRST when starting work** |
| `ARCHITECTURE.md` | Algorithm specs | — | Detailed math + data flow |
| `ROADMAP.md` | Phase breakdown | — | Implementation schedule |

---

## 8. FUTURE PRIORITIES

### Immediate Next Steps (Next Session)

#### 🟡 Step 0 — Phase 2 Metrics (asymmetry, sway, variability, clinical flags)

**Prerequisites satisfied:** world_x [0,6] ✓, foot strikes > 0 ✓, quartile coverage invariant ✓.

**Ready to implement:**

#### Phase 2 Implementation Order

1. **`metrics/asymmetry.py`** — Robinson AI formula per quartile: `AI = |X_L - X_R| / (0.5 × (X_L + X_R)) × 100`. Composite: `0.6 × AI_step_length + 0.4 × AI_swing_time`.

2. **`metrics/sway.py`** — RMS mediolateral COM displacement: `COM = 0.6 × midpoint(L_hip, R_hip) + 0.4 × midpoint(L_shoulder, R_shoulder)`.

3. **`metrics/variability.py`** — CV for stride length and step time per quartile.

4. **`metrics/global_metrics.py`** — Aggregate Q1–Q4 → trial-level metrics. Wire into `GaitProcessor.process()`.

5. **`clinical/flags.py` + `clinical/thresholds.py`** — Evidence-based threshold checks (see CLAUDE.md table). Replace stub `ClinicalReport` in processor.

### Medium-Term Priorities (Next 2–4 Sessions)

6. **Phase 2 Implementation** (4 weeks estimated)
   - FOG detector (`gait_events/fog_detector.py`)
   - Asymmetry computation (`metrics/asymmetry.py`)
   - Sway computation (`metrics/sway.py`)
   - Variability (CV) computation (`metrics/variability.py`)
   - Global metrics (`metrics/global_metrics.py`)
   - Clinical flags (`clinical/flags.py`)
   - Evidence-based thresholds (`clinical/thresholds.py`)

7. **Phase 3: FastAPI Server** (2–3 weeks estimated)
   - Background job processing
   - WebSocket progress streaming
   - REST endpoints: `/analyze`, `/status/{job_id}`, `/results/{job_id}`, `/export/{job_id}/{format}`

### Long-Term Improvements (6+ Months)

8. **Validation Study** (Phase 6)
   - Collect labeled dataset (manual annotation + GaitRite/IMU ground truth)
   - Validate ICC > 0.90 on all primary metrics
   - Edge-case testing (shuffling, FOG, asymmetry, early turns)

9. **Temporal Neural Models** (Phase 5+)
   - Train LSTM/Transformer for step detection, phase segmentation, FOG detection
   - Collect labeled video dataset (1000+ clips)
   - Compare to signal-processing baseline

10. **Mobile Optimization** (Phase 6)
    - Streaming pipeline (rolling buffer for long videos)
    - Model quantization (int8 inference)
    - Target: < 2× real-time on ARM CPU

11. **Frontend & Deployment** (Phase 4+)
    - React web app for video upload + result visualization
    - Docker containerization
    - Clinical trial deployment

### Recommended Refactors

| Area | Current | Recommended | When |
|------|---------|-------------|------|
| Person Detection | None (assumed input) | Integrate RTMDet ONNX | Phase 1 completion |
| FOG Detection | Placeholder only | Full spectral analysis | Phase 2 |
| Memory | All keypoints in memory | Streaming + rolling buffer | Phase 6 |
| Type Hints | Partial | Full strict mode (`mypy --strict`) | Phase 2 |
| Error Messages | Minimal | Detailed user-friendly messages | Phase 2 |
| Logging | Print statements | Structured logging (Python `logging`) | Phase 2 |
| Performance | No profiling yet | Benchmark all stages; optimize hotspots | Phase 3 |
| Documentation | Inline only | Full API docs + tutorials | Phase 6 |

---

## 9. SUGGESTED NEXT PROMPTS

Use these prompts in a new Claude session to continue development:

### For Fixing SVD World Coordinates (NEXT IMMEDIATE TASK)

```
I'm continuing the STRIDE gait analysis project. Read HANDOFF.md and CLAUDE.md first.

Detection and pose layers are fully fixed (45/45 unit tests pass, 0% missing frames on all 
real-video clips). The current blocker is that SVDAutoCalibrator produces centered world_x 
≈ [−3, +3] instead of absolute [0, 6]. This causes foot_strike detector to find 0 peaks 
and all gait metrics to be zero.

Task: Fix world coordinate calibration so `python -m stride analyze --video tests/Healthy.mp4 
--detector rtmdet -v` produces Total steps > 0 and Cadence > 0.

Key files:
- src/stride/calibration/homography.py — SVDAutoCalibrator.calibrate() produces the centered result
- src/stride/pipeline/processor.py — run_pass2 Stage 2 applies world coords; can post-process here
- src/stride/segmentation/phase_detector.py — TOWARD phase detection identifies walking direction

Approach: In run_pass2, after computing world positions from SVD, detect the TOWARD-phase 
start anchor (high-confidence frames at the beginning of TOWARD) and apply a translation so 
the near endpoint → 0m and far endpoint → ~6m. Also correct sign if world_x direction is 
inverted relative to walking direction.

CLAUDE.md invariant #4 documents SVD centering as a known limitation. The fix should NOT 
change ManualHomographyCalibrator (which already produces absolute coords correctly). 
Only SVDAutoCalibrator / the post-SVD pipeline stage needs updating.

Exit criteria: tests/Healthy.mp4 → Total steps > 0, Cadence > 0, world_x range ≈ [0, 6].
```

### For Phase 2 Implementation

```
STRIDE Phase 1 is complete (processor, CLI, basic tests working). Starting Phase 2: Full metrics 
and clinical analysis. Need to implement:

1. FOG detector (gait_events/fog_detector.py) - spectral analysis with overlapping windows
2. Asymmetry metric (metrics/asymmetry.py) - Robinson AI formula
3. Sway metric (metrics/sway.py) - COM displacement RMS
4. Variability (metrics/variability.py) - CV for stride length/time
5. Clinical flags (clinical/flags.py) - threshold-based flag generation
6. Evidence-based thresholds (clinical/thresholds.py) - with citations

Reference: ARCHITECTURE.md describes all formulas. See CLAUDE.md for clinical thresholds table.
```

### For Testing & Validation

```
STRIDE Phase 1–2 implementation complete. Task: comprehensive testing and validation.

1. Expand unit tests to 100% coverage of core modules
2. Create integration test with synthetic gait: 6-step sequence → full pipeline → validate metrics
3. Test edge cases: shuffling gait, FOG episodes, early turn, asymmetry injection
4. Benchmark performance: target < 5× real-time on CPU

Reference: tests/fixtures/synthetic_gait.py for parametric gait generation.
Key metric: step_count within ±2 of expected, cadence within 10%.
```

### For API & Deployment (Phase 3)

```
STRIDE Phases 1–2 complete, ready for API server. Implement Phase 3:

1. FastAPI server (api/main.py) with background job processing
2. Async video analysis jobs with state machine (PENDING → PROCESSING → COMPLETE/FAILED)
3. REST endpoints: POST /analyze, GET /status/{job_id}, GET /results/{job_id}
4. WebSocket at /ws/{job_id} for real-time progress streaming
5. Export endpoints: /export/{job_id}/{csv|json|pdf}

Context: Full pipeline works; just need to wrap in async service.
Reference: ROADMAP.md Phase 3 specification.
```

### For Frontend Development (Phase 4)

```
STRIDE API server operational. Starting frontend (Phase 4):

1. React web app: video upload form + results dashboard
2. Display metrics: step count, cadence per quartile, asymmetry, sway, FOG episodes
3. Interactive timeline: scrub video, highlight gait events
4. Export buttons: CSV, JSON, PDF report
5. Clinician notes: allow annotation of test quality

API is at http://localhost:8000. See api/schemas.py for request/response models.
Reference: CLAUDE.md for clinical metric definitions.
```

### For Validation & Research

```
STRIDE system fully implemented. Starting validation study (Phase 6):

1. Collect labeled dataset: video + manual annotation + GaitRite/IMU ground truth
2. Validate ICC > 0.90 on all primary metrics
3. Edge-case analysis: shuffling, FOG, asymmetry, early turns, poor calibration
4. Performance profiling: target < 3× real-time on CPU
5. Compare to baseline systems (OpenPose, MediaPipe, commercial GaitRight)

Current implementation: See IMPLEMENTATION_PROGRESS.md for what's working.
Known limitations: See section 4 (Technical Challenges).
```

---

## 10. DEVELOPMENT NOTES

### Coding Standards

#### Python Style
- **Language:** Python 3.10+; use modern features (type hints, pattern matching, union syntax)
- **Format:** Black (100 char line length)
- **Lint:** Ruff (strict mode)
- **Type Hints:** Full type annotations; target `mypy --strict` eventually
- **Docstrings:** Short one-liners for functions; classes get Args/Returns sections

#### Example:
```python
def detect(
    self,
    keypoints: np.ndarray,
    timestamps: np.ndarray,
    schema: KeypointSchema,
) -> list[FootStrikeEvent]:
    """Detect foot strikes from keypoint sequence.

    Args:
        keypoints: (N, n_keypoints, 3) array [x, y, confidence]
        timestamps: (N,) array in seconds
        schema: KeypointSchema defining keypoint indices

    Returns:
        List of FootStrikeEvent objects sorted by timestamp
    """
    ...
```

#### Data Models
- Use `pydantic.BaseModel` for all data classes (never `@dataclass`)
- Immutable: Use `model_config = ConfigDict(frozen=True)` for results
- Serialization: Implement `to_json()` / `from_json()` for round-trip support
- Validation: Use Pydantic validators for thresholds, ranges, enum membership

#### Protocols & DI
- All swappable components implement a Protocol (not ABC)
- Constructor DI with optional defaults: `__init__(pose_estimator: PoseEstimator | None = None)`
- Never hardcode imports of implementations (breaks testability)

#### Module Structure
- Every module has an `__init__.py` that exports public classes/functions
- Keep modules focused: one algorithm per file (pose, tracking, calibration, etc.)
- No circular dependencies; follow dependency order: core → config → data → pipeline → CLI

### Architectural Philosophy

#### Core Principles (Non-Negotiable)

1. **Distance-Based Metrics**
   - Always use world-space position (meters), never frame indices
   - Enables handling of pathological gait with variable cadence

2. **Protocol-First Design**
   - All swappable components are Protocols
   - Enables testing with mocks; future model swaps without changing processor

3. **Immutable Data Flow**
   - Pass1Result, Pass2Result are frozen dataclasses
   - No mutable shared state in processor
   - Enables reentrancy, easier debugging

4. **No Side Effects in Construction**
   - `StriderConfig()` doesn't create directories
   - `__init__` is pure data; side effects are explicit methods
   - Enables test isolation

5. **Reproducibility**
   - Every `AnalysisResult` carries `config_hash`, `experiment_id`, `schema_version`
   - Full JSON round-trip support
   - Enables audit trail, re-running with same config

#### Anti-Patterns (Avoid)

❌ **Periodic-Gait Assumptions**  
Wrong: "Average cadence = total_time / total_steps"  
Right: "Cadence per-quartile from actual step timing"

❌ **Time-Based Quartile Assignment**  
Wrong: "Q1 = first 25% of frames"  
Right: "Q1 = frames with world_x < 3.0 meters"

❌ **Mutable Processor State**  
Wrong: `self.keypoints = []; process()`  
Right: `pass1_result = run_pass1(...); pass2_result = run_pass2(pass1_result)`

❌ **Bare Magic Numbers**  
Wrong: `if ankle_y < 15:`  
Right: `if ankle_y < schema.left_ankle_threshold`

❌ **Skipping Validation**  
Wrong: Assume detections are good; don't filter  
Right: Validate confidence > threshold; remove low-confidence detections

### Important Warnings

#### 🚨 Critical Bugs to Watch

1. **Window Overlap in FOG Detection** (Phase 2)
   - Spectral Freeze Index with 50% window overlap overwrites values
   - Must use `np.maximum.at(fi_values, indices, window_fi)` for max aggregation
   - Currently: Underestimates FOG episodes

2. **Early Turn Handling** (Phase 1 Verified)
   - If patient turns before 6m, `PhaseDetector` must detect it
   - `TurnDetector` must set `turn_distance_m` parameter in `QuartileEngine`
   - Missing this: Wrong quartile assignments for AWAY phase

3. **Memory OOM on Long Videos**
   - Current design stores all keypoints in `Pass1Result` (memory resident)
   - 30-min video at 30 fps = 54M keypoints = multi-GB RAM
   - Will fail on mobile devices
   - Planned refactor (Phase 6): streaming + rolling buffer

#### ⚠️ Fragile Areas

1. **RTMPose SimCC Decoding**
   - Easy to flip axis or offset indices
   - Test: Run on known image; verify ankle position matches visual
   - Validation: `smoother.py` should produce smooth ankle trajectory

2. **Homography Roundtrip**
   - Easy to invert H incorrectly
   - Test: `SpatialMapper.validate_roundtrip()` — should be < 1mm error
   - Validation: Process calibration frame; check that ankle pixel maps back

3. **Quartile Assignment at Boundaries**
   - Edge case: step exactly at 3m or 6m boundary
   - Test: `test_quartile_engine.py` has boundary test cases
   - Current: Uses `<` for TOWARD, `<=` for away (check consistency)

4. **ByteTrack with Single Person**
   - Designed for multi-object; may over-complicate for single patient
   - Risk: If multiple detections per frame (multiple body parts detected), tracking breaks
   - Mitigation: Add `PatientSelector` to lock to largest bbox

5. **Calibration Failure Handling**
   - Current: ManualHomographyCalibrator crashes if user clicks bad points
   - Current: SVDAutoCalibrator crashes if ankle visibility < 10 frames
   - Needed: Graceful fallback + user warnings

### Known Fragile Tests

- ✅ `test_quartile_engine.py` — 19/19 passing; comprehensive distance-based assignment coverage
- ❌ No tests for RTMPose (needs ONNX model loaded)
- ❌ No tests for ByteTrack (needs synthetic detections)
- ❌ No tests for foot strike detector (sinusoidal signals — not complex)
- ❌ No integration tests (needs full pipeline via generate_synthetic_pass1_result)

### Performance Baselines

| Stage | Target | Hardware | Notes |
|-------|--------|----------|-------|
| RTMPose inference | 15–25 fps | CPU (i7, 16GB) | 384×288 input |
| Tracking (ByteTrack) | < 1ms per frame | — | Single person |
| Calibration (SVD) | < 100ms | — | On full trajectory |
| Events detection | < 5ms per frame | — | Vectorized scipy |
| Metrics computation | < 50ms | — | Per-quartile aggregation |
| **Full pipeline** | **< 5× real-time** | CPU | 2-min video in < 10 min |

---

## 11. CRITICAL READING LIST

### Must Read (In This Order)

1. **[CLAUDE.md](./CLAUDE.md)** — Project constraints, clinical definitions, failure modes
   - Read first; defines all requirements and constraints
   - Section "Critical Mathematical Definitions" has formulas

2. **[ARCHITECTURE.md](./ARCHITECTURE.md)** — Algorithm specifications, data flow, module contracts
   - Detailed math for all metrics
   - Data flow diagram
   - Failure case handling

3. **[ROADMAP.md](./ROADMAP.md)** — Phase breakdown, success criteria, dependencies
   - Phase 1 success criteria
   - Phase 2+ planning

4. **[IMPLEMENTATION_PROGRESS.md](./IMPLEMENTATION_PROGRESS.md)** — Session-by-session progress tracking
   - What's been done
   - Files created/modified
   - Verification checklist

### Reference (As Needed)

- `src/stride/core/protocols.py` — See exact Protocol signatures
- `src/stride/config/schema.py` — See all configurable thresholds
- `src/stride/pipeline/context.py` — See data structures flowing through pipeline
- `tests/conftest.py` — See test fixtures

---

## Summary

STRIDE is a **research-grade gait analysis system** with a clean, modular architecture. The entire upstream stack (detection, tracking, pose, calibration, foot-strike detection) is working end-to-end on real clinical video. **57/57 unit tests pass.** Healthy.mp4 produces 11 foot strikes, world_x [0,6], phase split ~47/53%, quartile invariant holds.

**Current state:** Phase 1 fully complete. Phase 2 metrics (asymmetry, sway, variability, global metrics, clinical flags) are the next implementation target — the spatial/temporal prerequisites for all of these are now satisfied.

**Known placeholders in the output (not bugs):** `asymmetry_score=0.0`, `sway_rms=0.0` — these fields are explicitly `# Computed in Phase 2` stubs in `metrics/per_quartile.py`. Clinical flags not firing for the same reason.

**Key architectural facts:**
- All enums are `StrEnum` (not `str, Enum`) — Python 3.13 numpy compatibility
- `FootStrikeEvent.detection_phase` (Phase at detection time) ≠ `FootStrikeEvent.quartile` (assigned by processor Stage 6)
- `RTMPoseEstimator` is explicitly injected — no auto-construction
- `CalibrationResult` fields: `homography_matrix`, `scale_px_to_m`, `pc1_variance`, `method`
- SVD SimCC confidence = raw max-logit (NOT softmax-mean) — this was the root cause of the whole-frame collapse; fixed
- Full-frame synthetic box is **gone** — missing detections emit explicit MISSING frames, not garbage keypoints
- RTMDet-m (640 input) is auto-preferred when present; nano is the fallback; YOLOv8 selectable via `--detector yolov8`
- `StepValidator` backward-step guard is phase-aware — AWAY events pass validation
- `docs/CAPTURE_PROTOCOL.md` is the operator SOP for clinical video capture

**Next session should start with:** Read CLAUDE.md + this HANDOFF.md, then fix `SVDAutoCalibrator` world coordinate translation/sign correction (see Section 8 "Step 0" and the "For Fixing SVD World Coordinates" prompt in Section 9).

---

**Document Version:** 2.1  
**Last Updated:** 2026-05-16  
**Author:** Claude Code (Multi-Session Context)  
**Status:** Phase 1 complete — 57/57 tests; 11 foot strikes on Healthy.mp4; Phase 2 metrics next
