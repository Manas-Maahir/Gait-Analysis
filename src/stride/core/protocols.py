"""Protocol (interface) definitions for swappable pipeline components.

Using Python's typing.Protocol enables duck-typing with full type-checking support.
All pipeline components implement these protocols, enabling runtime substitution
(e.g., swapping RTMPose for MediaPipe, or injecting mocks for testing).
"""

from typing import Protocol, Sequence
from dataclasses import dataclass

import numpy as np

from .types import Phase, Quartile
from .keypoints import KeypointSchema


@dataclass
class BoundingBox:
    """Person detection bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass
class Detection:
    """Person detection result."""

    bbox: BoundingBox
    track_id: int | None = None


@dataclass
class KeypointFrame:
    """Per-frame keypoint data from pose estimator."""

    keypoints: np.ndarray  # (n_keypoints, 3) where 3 = [x, y, confidence]
    confidence: float      # Overall frame confidence


@dataclass
class CalibrationResult:
    """Output of calibration (homography + metadata)."""

    homography_matrix: np.ndarray  # 3×3 transformation matrix
    scale_px_to_m: float           # Pixels to meters conversion factor
    pc1_variance: float            # PCA variance for auto-calibration (0–1)
    method: str                    # "manual" or "auto"


@dataclass
class FrameData:
    """Intermediate frame-level data for pipeline stages."""

    frame_idx: int
    timestamp: float
    keypoints: np.ndarray          # (n_keypoints, 3)
    world_position: np.ndarray | None = None  # (2,) meters after calibration
    phase: Phase | None = None
    confidence: float = 1.0


class PersonDetector(Protocol):
    """Protocol for person detection models."""

    def detect(self, frame: np.ndarray) -> Sequence[BoundingBox]:
        """Detect persons in a frame.

        Args:
            frame: BGR image (H, W, 3) uint8

        Returns:
            List of BoundingBox objects, sorted by area descending
        """
        ...


class PoseEstimator(Protocol):
    """Protocol for pose estimation models."""

    @property
    def keypoint_schema(self) -> KeypointSchema:
        """Return the keypoint index registry for this model."""
        ...

    def estimate(self, frame: np.ndarray, bbox: BoundingBox) -> KeypointFrame:
        """Estimate pose keypoints from a frame and bounding box.

        Args:
            frame: BGR image (H, W, 3) uint8
            bbox: Person detection bounding box

        Returns:
            KeypointFrame with (n_keypoints, 3) array
        """
        ...


class Tracker(Protocol):
    """Protocol for multi-object tracking."""

    def update(
        self,
        detections: Sequence[Detection],
        frame: np.ndarray,
    ) -> Sequence[Detection]:
        """Update tracks with new detections.

        Args:
            detections: List of Detection objects with bounding boxes
            frame: Current video frame (for optional feature extraction)

        Returns:
            Detections with assigned track_id values
        """
        ...


class Calibrator(Protocol):
    """Protocol for spatial calibration (image → world coordinates)."""

    def calibrate(
        self,
        trajectory: np.ndarray,  # (N, 2) image coordinates
        metadata: dict | None = None,
    ) -> CalibrationResult:
        """Compute calibration from ankle trajectory.

        Args:
            trajectory: (N, 2) ankle positions in image space
            metadata: Optional hints (manual 4-pt references, etc.)

        Returns:
            CalibrationResult with homography matrix
        """
        ...

    def image_to_world(
        self,
        points: np.ndarray,
        result: CalibrationResult,
    ) -> np.ndarray:
        """Transform image coordinates to world space (meters).

        Args:
            points: (..., 2) points in image space
            result: CalibrationResult from calibrate()

        Returns:
            (..., 2) points in world space (meters)
        """
        ...


class GaitEventDetector(Protocol):
    """Protocol for detecting gait events (foot strikes, FOG, etc.)."""

    def detect(
        self,
        frames: Sequence[FrameData],
        schema: KeypointSchema,
    ) -> Sequence:  # List[FootStrikeEvent] or similar
        """Detect gait events from frame sequence.

        Args:
            frames: Sequence of FrameData with keypoints and world positions
            schema: KeypointSchema to resolve keypoint indices

        Returns:
            List of event objects (FootStrikeEvent, FOGEpisode, etc.)
        """
        ...


class MetricComputer(Protocol):
    """Protocol for computing gait metrics from events."""

    def compute(
        self,
        events: Sequence,  # List[FootStrikeEvent]
        context: dict,    # Pipeline context with world_positions, phases, etc.
    ) -> dict:
        """Compute metrics from detected gait events.

        Args:
            events: List of gait events
            context: Pipeline context with all auxiliary data

        Returns:
            Dictionary of computed metrics
        """
        ...


class ClinicalAnalyzer(Protocol):
    """Protocol for generating clinical flags from metrics."""

    def analyze(self, metrics: dict) -> Sequence:  # List[ClinicalFlag]
        """Generate clinical flags from computed metrics.

        Args:
            metrics: Dictionary of computed metrics

        Returns:
            List of ClinicalFlag objects with type, severity, description
        """
        ...
