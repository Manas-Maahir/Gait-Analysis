# Strider — Clinical Gait Analysis System

**Advanced monocular video-based gait analysis for neurological movement disorders.**

Strider analyzes single RGB video of a standardized 6-meter walk test and produces clinically meaningful gait metrics with an interactive analysis dashboard. Designed for Parkinson's disease research, gait abnormality detection, and rehabilitation monitoring.

## Key Features

- **Distance-Based Metrics:** All spatial quantification derived from world coordinates along the calibrated 6m path, never frame/time-based
- **Pathological Gait Support:** Robust to shuffling, freezing, asymmetric stepping, variable cadence, and irregular gait patterns
- **Per-Quartile Analysis:** Gait metrics segmented into four 1.5m zones (Q1/Q2 toward, Q3/Q4 away) for detailed spatial progression analysis
- **Clinical Flags:** Evidence-based warnings for freezing of gait, abnormal asymmetry, reduced cadence, excessive sway, and other pathological indicators
- **Research-Grade:** SimCC pose estimation (RTMPose), spectral freeze detection, comprehensive metric validation

## Metrics Produced

### Per-Quartile (Q1, Q2, Q3, Q4)
- Step count (validated, distance-based assignment)
- Cadence (steps/min)
- Asymmetry score (Robinson AI formula, L/R imbalance)
- Sway metric (RMS mediolateral displacement)
- Step length variability (coefficient of variation)
- Freezing of gait episodes (spectral freeze index)

### Global
- Total trial time, toward-walk time, away-walk time
- Turning time (initiation to completion)
- Overall cadence, asymmetry, sway
- Clinical flags with severity levels
- Confidence scores per metric

## Quick Start

### Setup

```bash
# Clone and enter project
cd C:\Users\manas\Desktop\Projects\Stride

# Create Python environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Download ONNX pose models
python scripts/download_models.py

# Verify environment
python scripts/setup_env.py
```

### Process a Video

```bash
# Analyze a single 6m walk test video
python -m strider.cli analyze \
  --video path/to/test.mp4 \
  --output results.json \
  --calibration-mode auto

# With manual floor-point calibration
python -m strider.cli analyze \
  --video path/to/test.mp4 \
  --output results.json \
  --calibration-mode manual \
  --calibration-data calibration.json  # 4 floor points + world coords
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration test (synthetic gait)
pytest tests/integration/test_pipeline_synthetic.py -v

# With coverage
pytest tests/ --cov=strider --cov-report=html
```

## Input Requirements

- **Video:** Single monocular RGB video (any phone camera OK)
- **Resolution:** 480p–1080p (dynamically resized to fit)
- **Frame rate:** 24–60 fps (variable OK, resampled internally)
- **Duration:** ~2–5 minutes (6m walk + turn at any speed)
- **Calibration:** Bright tape markers on floor (for homography computation)

## Output

```json
{
  "metadata": {
    "video_path": "...",
    "fps": 30,
    "total_frames": 3600,
    "processing_time_sec": 45
  },
  "metrics": {
    "q1": {
      "step_count": 12,
      "cadence_steps_per_min": 94.5,
      "asymmetry_score": 8.2,
      "sway_rms_meters": 0.032,
      "duration_sec": 7.63,
      "fog_episodes": 0
    },
    "q2": { ... },
    "q3": { ... },
    "q4": { ... },
    "global": {
      "total_steps": 48,
      "overall_cadence": 93.2,
      "overall_asymmetry": 8.8,
      "overall_sway": 0.034,
      "turning_time_sec": 2.1,
      "toward_6m_time_sec": 30.5,
      "away_6m_time_sec": 31.2,
      "total_trial_time_sec": 63.8
    }
  },
  "clinical_report": {
    "flags": [
      {
        "flag": "HIGH_STEP_VARIABILITY",
        "severity": "WARNING",
        "quartile": "Q3",
        "value": 5.2,
        "threshold": 4.0,
        "description": "Stride length CV exceeds normal range"
      }
    ],
    "confidence_scores": { ... }
  },
  "annotated_video": "output_annotated.mp4"
}
```

## Architecture

**Pipeline (Backend):**
```
Video → Pose Estimation (RTMPose)
     → Tracking (ByteTrack)
     → Spatial Calibration (Homography)
     → Phase Detection (TOWARD/TURN/AWAY)
     → Quartile Assignment (Distance-based)
     → Foot Strike Detection (Ankle peaks)
     → FOG Detection (Spectral)
     → Metric Computation
     → Clinical Analysis
     → Output (JSON + Annotated Video)
```

**Frontend (Phase 4+):**
React + FastAPI dashboard with:
- Real-time video playback with pose overlay
- Interactive quartile visualization
- Step event timeline with clickable markers
- Dynamic metric cards + clinical flags
- Sway trajectory, cadence, and stride variability plots
- Clinician notes + CSV/JSON/PDF export

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design.

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Pose Estimation | RTMPose-l WholeBody (ONNX) |
| Person Detection | RTMDet-nano (ONNX) |
| Tracking | ByteTrack + Kalman Filter |
| Spatial Calibration | OpenCV Homography |
| Backend | FastAPI (Phase 3+) |
| Frontend | React + Vite + TypeScript (Phase 4+) |
| Testing | pytest + synthetic gait generators |

## Development Status

**Current Phase:** 1 (Core Pipeline MVP)  
**Status:** Scaffolding and initialization  

See [ROADMAP.md](ROADMAP.md) for detailed phase timelines and progress.

## Documentation

- [CLAUDE.md](CLAUDE.md) — Development guidelines and module reference
- [ARCHITECTURE.md](ARCHITECTURE.md) — System design and data flow
- [ROADMAP.md](ROADMAP.md) — Implementation phases and progress tracker

## Citations & References

Metrics and thresholds based on:
- Bohannon RW (1997) — Normative cadence data
- Robinson RO (1987) — Asymmetry Index formula
- Brach JS (2005) — Stride variability thresholds
- Nieuwboer A (2004) — Freezing of Gait spectral analysis
- Moe-Nilssen R (2004) — Sway measurement
- Salarian A (2010) — Turn duration normative data
- Patterson KK (2010) — Asymmetry in neurological gait

## License

[To be determined]

## Contact

Manas Maahir (manasmaahir27@gmail.com)
