"""Strider core: domain types, protocols, and schemas."""

from .types import (
    ClinicalFlagType,
    ClinicalSeverity,
    Phase,
    Quartile,
    Side,
)
from .keypoints import (
    KeypointSchema,
    RTMPoseWholebody133,
    MediaPipePose33,
    get_keypoint_schema,
)
from .protocols import (
    PoseEstimator,
    Tracker,
    Calibrator,
    GaitEventDetector,
    MetricComputer,
    ClinicalAnalyzer,
    BoundingBox,
    Detection,
    KeypointFrame,
    CalibrationResult,
    FrameData,
)

__all__ = [
    # Types
    "Side",
    "Phase",
    "Quartile",
    "ClinicalFlagType",
    "ClinicalSeverity",
    # Keypoints
    "KeypointSchema",
    "RTMPoseWholebody133",
    "MediaPipePose33",
    "get_keypoint_schema",
    # Protocols
    "PoseEstimator",
    "Tracker",
    "Calibrator",
    "GaitEventDetector",
    "MetricComputer",
    "ClinicalAnalyzer",
    # Data types
    "BoundingBox",
    "Detection",
    "KeypointFrame",
    "CalibrationResult",
    "FrameData",
]
