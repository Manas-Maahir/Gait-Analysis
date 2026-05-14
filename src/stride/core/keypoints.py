"""Keypoint schema registry for model-agnostic index references.

Each pose model (RTMPose, MediaPipe, etc.) has different keypoint indices.
This module centralizes the mapping so downstream code never uses bare integers like `15`.
Instead, code uses `schema.left_ankle` regardless of which model is active.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class KeypointSchema:
    """Immutable registry of keypoint indices and metadata for a pose model."""

    model_name: str
    n_keypoints: int

    # Lower body (primary for gait analysis)
    left_ankle: int
    right_ankle: int
    left_knee: int
    right_knee: int
    left_hip: int
    right_hip: int

    # Upper body (secondary, for COM/sway)
    left_shoulder: int
    right_shoulder: int
    left_elbow: int
    right_elbow: int

    # Head (optional, for orientation)
    nose: int
    left_eye: int
    right_eye: int

    def validate(self) -> None:
        """Ensure all indices are within valid range [0, n_keypoints)."""
        indices = [
            self.left_ankle, self.right_ankle,
            self.left_knee, self.right_knee,
            self.left_hip, self.right_hip,
            self.left_shoulder, self.right_shoulder,
            self.left_elbow, self.right_elbow,
            self.nose, self.left_eye, self.right_eye,
        ]
        for idx in indices:
            if not (0 <= idx < self.n_keypoints):
                raise ValueError(
                    f"Invalid keypoint index {idx} for model {self.model_name} "
                    f"with {self.n_keypoints} keypoints"
                )


# RTMPose-l WholeBody: 133 keypoints
# Body: 17 keypoints (COCO-style)
# Foot: 2×6=12 keypoints (left/right × 6 foot parts)
# Hand: 2×21=42 keypoints (left/right hands)
# Face: 2×68=136 keypoints, but only subset used
# Total: ~133
RTMPoseWholebody133 = KeypointSchema(
    model_name="rtmpose-l-wholebody133",
    n_keypoints=133,
    # Body keypoints (COCO-style indices 0–16)
    left_ankle=15,
    right_ankle=16,
    left_knee=13,
    right_knee=14,
    left_hip=11,
    right_hip=12,
    left_shoulder=5,
    right_shoulder=6,
    left_elbow=7,
    right_elbow=8,
    nose=0,
    left_eye=1,
    right_eye=2,
)
RTMPoseWholebody133.validate()


# MediaPipe Pose: 33 keypoints
# Primary landmarks: 0–32 (body + hand + face points)
MediaPipePose33 = KeypointSchema(
    model_name="mediapipe-pose33",
    n_keypoints=33,
    # MediaPipe indices (COCO-compatible but different numbering)
    left_ankle=27,
    right_ankle=28,
    left_knee=25,
    right_knee=26,
    left_hip=23,
    right_hip=24,
    left_shoulder=11,
    right_shoulder=12,
    left_elbow=13,
    right_elbow=14,
    nose=0,
    left_eye=2,
    right_eye=5,
)
MediaPipePose33.validate()


# Registry for runtime lookup
SCHEMA_REGISTRY = {
    "rtmpose-l-wholebody133": RTMPoseWholebody133,
    "mediapipe-pose33": MediaPipePose33,
}


def get_keypoint_schema(model_name: str) -> KeypointSchema:
    """Retrieve keypoint schema by model name.

    Args:
        model_name: Key from SCHEMA_REGISTRY (e.g., "rtmpose-l-wholebody133")

    Returns:
        KeypointSchema for the model

    Raises:
        KeyError: If model_name not found in registry
    """
    if model_name not in SCHEMA_REGISTRY:
        available = ", ".join(SCHEMA_REGISTRY.keys())
        raise KeyError(
            f"Unknown model '{model_name}'. Available: {available}"
        )
    return SCHEMA_REGISTRY[model_name]
