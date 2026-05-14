"""Immutable pipeline stage result containers.

Instead of mutable shared state in GaitProcessor, each stage produces immutable results
that flow to the next stage. This enables:
- Reentrancy (process() can be called multiple times safely)
- Testability (each stage can be unit tested with known inputs)
- Debugging (full trace of data transformation through pipeline)
"""

from dataclasses import dataclass, field
from typing import Sequence, Optional

import numpy as np

from ..core.types import Phase
from ..core.keypoints import KeypointSchema
from ..data.events import FootStrikeEvent, FOGEpisode
from ..data.metrics import QuartileMetrics
from ..core.protocols import CalibrationResult


@dataclass(frozen=True)
class Pass1Result:
    """Immutable output of Pass 1: all raw per-frame data.

    Pass 1 is responsible for: video I/O → person detection → tracking →
    pose estimation → keypoint smoothing → store raw keypoints in memory.
    """

    keypoints: np.ndarray
    """(N, n_keypoints, 3) array where 3 = [x, y, confidence]"""

    timestamps: np.ndarray
    """(N,) seconds from video start"""

    track_ids: np.ndarray
    """(N,) integer track ID per frame"""

    fps: float
    """Frames per second of input video"""

    total_frames: int
    """Total number of frames processed"""

    schema: KeypointSchema
    """Keypoint schema of the pose model used"""

    video_path: str
    """Path to input video"""

    bboxes: Optional[np.ndarray] = field(default=None)
    """(N, 4) array [x1, y1, x2, y2] of bboxes fed to pose estimator per frame (debug)"""


@dataclass(frozen=True)
class Pass2Result:
    """Immutable output of Pass 2: all computed analytical outputs.

    Pass 2 is responsible for: spatial calibration (homography) → phase detection →
    quartile assignment → gait event detection → metric computation → clinical analysis.
    """

    world_positions: np.ndarray
    """(N, 2) meters [x, y] along calibrated path"""

    phases: np.ndarray
    """(N,) Phase enum values (TOWARD, TURN, AWAY)"""

    foot_strikes: tuple[FootStrikeEvent, ...]
    """All detected foot strike events (left and right)"""

    fog_episodes: tuple[FOGEpisode, ...]
    """All detected Freezing of Gait episodes"""

    calibration: CalibrationResult
    """Calibration result: homography matrix, scale factor, variance"""

    actual_turn_distance_m: float
    """Measured turn distance (may differ from config path_length_m)"""

    phase_start_frames: dict[Phase, int]
    """Frame indices where each phase begins"""

    phase_end_frames: dict[Phase, int]
    """Frame indices where each phase ends"""

    quartile_metrics: dict[str, QuartileMetrics]  # Key: "Q1", "Q2", "Q3", "Q4"
    """Per-quartile metrics (step count, cadence, etc.)"""

    turning_time_sec: float = 0.0
    """Measured time spent in TURN phase (seconds)"""
