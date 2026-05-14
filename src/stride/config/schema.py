"""Strider configuration schema using Pydantic with no side effects.

Configuration is pure data — directory creation is explicit via ensure_directories().
This enables tests to instantiate configs without polluting the filesystem.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class StriderConfig(BaseModel):
    """Central configuration for Strider gait analysis pipeline.

    All processing parameters, model paths, thresholds, and calibration settings
    are defined here. This enables reproducibility and easy tuning.

    Side effects (mkdir) are NOT performed here. Call ensure_directories() explicitly.
    """

    # ========== Model Configuration ==========

    pose_model: Literal["rtmpose-l", "mediapipe"] = "rtmpose-l"
    """Primary pose estimation model."""

    detector_model: str = "rtmdet-nano"
    """Person detection model (ONNX)."""

    model_dir: Path = Path("models")
    """Directory containing ONNX model weights."""

    device: Literal["cpu", "cuda"] = "cpu"
    """Inference device. GPU support planned for Phase 2+."""

    # ========== Video Processing ==========

    target_fps: float = 30.0
    """Resample video to this frame rate for consistent processing."""

    frame_skip: int = 1
    """Process every Nth frame. 1 = all frames, 2 = every 2nd frame."""

    max_dim: int = 640
    """Resize video's longest dimension to this value."""

    # ========== Spatial Configuration ==========

    path_length_meters: float = 6.0
    """Total walking distance (6-meter walk test standard)."""

    calibration_mode: Literal["auto", "manual", "auto_with_fallback"] = "auto_with_fallback"
    """How to calibrate: auto SVD, manual 4-point, or auto with fallback."""

    calibration_pc1_threshold: float = 0.85
    """Min explained variance for PC1 in auto calibration."""

    # ========== Gait Event Detection ==========

    # Foot strike detection
    foot_strike_prominence_factor: float = 0.1
    """Adaptive peak prominence factor for foot strike detection."""

    foot_strike_min_interval_sec: float = 0.15
    """Minimum physiologically plausible step interval (seconds)."""

    foot_strike_max_interval_sec: float = 3.0
    """Maximum step interval (handles pauses)."""

    foot_strike_distance_frames: int | None = None
    """Min frames between peaks. Auto-computed if None."""

    # FOG detection
    fog_fi_threshold: float = 2.5
    """Spectral Freeze Index threshold for FOG episode."""

    fog_window_sec: float = 2.0
    """Sliding window duration for spectral analysis."""

    fog_min_duration_sec: float = 0.5
    """Minimum FOG episode duration to be flagged."""

    # ========== Quartile & Phase Detection ==========

    phase_velocity_threshold: float = 0.01
    """Velocity threshold for turn phase detection (m/s)."""

    quartile_boundary_tolerance_m: float = 0.1
    """Tolerance for quartile boundary assignment (±m from 3m/6m)."""

    # ========== Clinical Thresholds (Evidence-Based) ==========

    normal_cadence_min: float = 80.0
    """Normal minimum cadence (steps/min)."""

    normal_cadence_max: float = 130.0
    """Normal maximum cadence (steps/min)."""

    abnormal_asymmetry_warning: float = 10.0
    """Asymmetry Index > this (%) → WARNING flag."""

    abnormal_asymmetry_critical: float = 20.0
    """Asymmetry Index > this (%) → CRITICAL flag."""

    abnormal_sway_rms: float = 0.05
    """Sway RMS > this (meters) → WARNING flag."""

    abnormal_stride_cv: float = 4.0
    """Stride length CV > this (%) → WARNING flag."""

    abnormal_turning_time: float = 3.5
    """Turning time > this (seconds) → WARNING flag."""

    # ========== Confidence & Filtering ==========

    pose_confidence_threshold: float = 0.3
    """Keypoint confidence below this is treated as unreliable."""

    detection_score_threshold: float = 0.3
    """Person detection score below this is discarded."""

    tracking_lost_threshold_frames: int = 30
    """Remove track if lost for this many frames."""

    # ========== Output & Paths ==========

    output_dir: Path = Path("output")
    """Directory for results (JSON, annotated video, etc.)."""

    save_annotated_video: bool = True
    """Whether to render and save annotated video with pose overlay."""

    save_intermediate_data: bool = False
    """Whether to save intermediate results for debugging."""

    verbose: bool = False
    """Enable verbose logging."""

    model_config = {
        "validate_assignment": True,
    }

    @field_validator("target_fps")
    @classmethod
    def validate_target_fps(cls, v: float) -> float:
        """Ensure target_fps is reasonable."""
        if v < 10:
            raise ValueError(f"target_fps must be >= 10, got {v}")
        return v

    @field_validator("path_length_meters")
    @classmethod
    def validate_path_length(cls, v: float) -> float:
        """Ensure path_length_meters is positive."""
        if v <= 0:
            raise ValueError(f"path_length_meters must be > 0, got {v}")
        return v

    @field_validator("pose_confidence_threshold")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is in [0, 1]."""
        if not (0 <= v <= 1):
            raise ValueError(f"pose_confidence_threshold must be in [0, 1], got {v}")
        return v

    def ensure_directories(self) -> None:
        """Explicitly create required directories.

        This is NOT called automatically during __init__ — tests can instantiate
        config without side effects. Call this method before processing starts.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def foot_strike_distance_frames_computed(self) -> int:
        """Get computed distance threshold for foot strike detection."""
        if self.foot_strike_distance_frames is None:
            return int(self.target_fps * 0.2)
        return self.foot_strike_distance_frames

    @property
    def config_hash(self) -> str:
        """Compute SHA256 hash of config for reproducibility tracking.

        Used in AnalysisResult to tag which config produced the results.
        """
        data = self.model_dump(exclude_none=True, mode="json")
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return self.model_dump(exclude_none=False, mode="json")
