"""Parametric synthetic gait generator for testing.

Generates synthetic gait trajectories with configurable parameters:
- Cadence (steps/min)
- Stride length (meters)
- Asymmetry (L/R differences)
- FOG episodes
- Lateral sway

Output: sequences of FrameData or keypoint arrays matching RTMPose format.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from stride.core import FrameData, KeypointSchema, Phase
from stride.pipeline.context import Pass1Result


@dataclass
class SyntheticGaitParams:
    """Configuration for synthetic gait generation."""

    cadence_steps_per_min: float = 100.0
    """Target cadence (steps/minute)."""

    stride_length_m: float = 1.3
    """Stride length (meters)."""

    stride_length_asymmetry_percent: float = 0.0
    """Left-right stride length asymmetry (%)."""

    step_height_m: float = 0.05
    """Peak ankle height during swing (meters)."""

    lateral_sway_m: float = 0.02
    """Mediolateral sway amplitude (meters)."""

    walking_speed_m_s: float = 1.3
    """Average walking speed (meters/second)."""

    fog_episodes: list[tuple[float, float]] = None
    """List of (start_sec, end_sec) for FOG episodes."""

    path_length_m: float = 6.0
    """Total walking distance (meters)."""

    def __post_init__(self):
        if self.fog_episodes is None:
            self.fog_episodes = []


def _generate_ankle_trajectory(
    duration_sec: float,
    fps: float,
    params: SyntheticGaitParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic ankle pixel trajectories.

    Returns:
        left_ankle_xy:  (N, 2) pixel coordinates
        right_ankle_xy: (N, 2) pixel coordinates
        world_x:        (N,)   world-space X position in meters
    """
    total_frames = int(duration_sec * fps)
    timestamps = np.arange(total_frames, dtype=np.float32) / fps

    path_length = params.path_length_m
    half_duration = duration_sec / 2.0  # toward vs away

    # World X position: ramp toward then ramp away (linear progression)
    toward_speed = path_length / half_duration  # m/s
    world_x = np.where(
        timestamps < half_duration,
        timestamps * toward_speed,                          # TOWARD: 0 → 6m
        (2 * path_length) - timestamps * toward_speed,      # AWAY:  6 → 0m (linear)
    )
    world_x = np.clip(world_x, 0.0, path_length)

    # Pixel mapping: project world_x (0..6m) → pixel_x in image (100..540 px)
    img_width_pixels = 640
    x_margin = 50
    pixels_per_meter = (img_width_pixels - 2 * x_margin) / path_length
    pixel_x_center = x_margin + world_x * pixels_per_meter  # (N,)

    # Person is at y_center ≈ 400 px (from top), ankles ~100px below hip
    img_height_pixels = 480
    ankle_y_center = 400.0  # px, approximately constant for lateral-view camera

    # Step cycle: sinusoidal ankle vertical motion (pixel space)
    # One complete gait cycle = 2 steps = cadence / 60 * 2 cycles per second
    step_freq_hz = params.cadence_steps_per_min / 60.0  # steps/sec
    stride_freq_hz = step_freq_hz / 2.0                  # strides/sec (L+R = 1 stride)

    # Ankle swing amplitude in pixels (approximate: step_height_m * pixels_per_meter)
    swing_amplitude_px = params.step_height_m * pixels_per_meter

    # Left ankle: sin wave; Right ankle: sin wave offset by half-cycle (π)
    left_ankle_y = (
        ankle_y_center + swing_amplitude_px *
        np.sin(2 * np.pi * stride_freq_hz * timestamps)
    )
    right_ankle_y = (
        ankle_y_center + swing_amplitude_px *
        np.sin(2 * np.pi * stride_freq_hz * timestamps + np.pi)
    )

    # Lateral separation (ankles offset from center by ~0.1m each = ~10 px)
    lateral_offset_px = 0.1 * pixels_per_meter
    left_ankle_x = pixel_x_center - lateral_offset_px
    right_ankle_x = pixel_x_center + lateral_offset_px

    # Apply asymmetry: scale one side's amplitude
    asym = params.stride_length_asymmetry_percent / 100.0
    left_ankle_y = ankle_y_center + (1 + asym) * (left_ankle_y - ankle_y_center)

    # Apply lateral sway (sinusoidal at same frequency)
    sway_amplitude_px = params.lateral_sway_m * pixels_per_meter
    sway = sway_amplitude_px * np.sin(2 * np.pi * step_freq_hz * timestamps)
    left_ankle_x += sway
    right_ankle_x += sway

    # Simulate FOG episodes (zero ankle displacement in those windows)
    for (fog_start_sec, fog_end_sec) in params.fog_episodes:
        fog_start_f = int(fog_start_sec * fps)
        fog_end_f = int(fog_end_sec * fps)
        fog_start_f = max(0, min(fog_start_f, total_frames - 1))
        fog_end_f = max(0, min(fog_end_f, total_frames - 1))
        # During FOG: freeze Y at neutral (ankle not moving)
        left_ankle_y[fog_start_f:fog_end_f] = ankle_y_center
        right_ankle_y[fog_start_f:fog_end_f] = ankle_y_center

    left_ankle_xy = np.stack([left_ankle_x, left_ankle_y], axis=1).astype(np.float32)
    right_ankle_xy = np.stack([right_ankle_x, right_ankle_y], axis=1).astype(np.float32)

    return left_ankle_xy, right_ankle_xy, world_x.astype(np.float32)


def _generate_synthetic_keypoints(
    duration_sec: float,
    fps: float,
    params: SyntheticGaitParams,
    schema: KeypointSchema,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic (N, n_keypoints, 3) keypoint array.

    Returns:
        keypoints: (N, n_kpts, 3) — [x, y, confidence]
        timestamps: (N,)
        world_x: (N,) — ground-truth world position in meters
    """
    total_frames = int(duration_sec * fps)
    timestamps = np.arange(total_frames, dtype=np.float32) / fps

    left_ankle_xy, right_ankle_xy, world_x = _generate_ankle_trajectory(
        duration_sec, fps, params
    )

    # Allocate keypoints array — all zeros initially (confidence = 0 by default)
    keypoints = np.zeros((total_frames, schema.n_keypoints, 3), dtype=np.float32)

    # Set ankle keypoints with confidence=1.0
    keypoints[:, schema.left_ankle, 0] = left_ankle_xy[:, 0]   # x
    keypoints[:, schema.left_ankle, 1] = left_ankle_xy[:, 1]   # y
    keypoints[:, schema.left_ankle, 2] = 1.0                    # conf

    keypoints[:, schema.right_ankle, 0] = right_ankle_xy[:, 0]
    keypoints[:, schema.right_ankle, 1] = right_ankle_xy[:, 1]
    keypoints[:, schema.right_ankle, 2] = 1.0

    # Derive hip positions from world_x (hips follow center of mass)
    # Hips ~200px above ankles in lateral view (approximate body height ~150px)
    img_margin = 50
    pixels_per_meter = (640 - 2 * img_margin) / params.path_length_m
    pixel_x_center = img_margin + world_x * pixels_per_meter
    hip_y = 300.0  # fixed y for hips (above ankles at ~400)
    lateral_hip_offset = 0.08 * pixels_per_meter

    keypoints[:, schema.left_hip, 0] = pixel_x_center - lateral_hip_offset
    keypoints[:, schema.left_hip, 1] = hip_y
    keypoints[:, schema.left_hip, 2] = 1.0

    keypoints[:, schema.right_hip, 0] = pixel_x_center + lateral_hip_offset
    keypoints[:, schema.right_hip, 1] = hip_y
    keypoints[:, schema.right_hip, 2] = 1.0

    # Shoulders ~80px above hips
    shoulder_y = hip_y - 80.0
    keypoints[:, schema.left_shoulder, 0] = pixel_x_center - lateral_hip_offset * 1.2
    keypoints[:, schema.left_shoulder, 1] = shoulder_y
    keypoints[:, schema.left_shoulder, 2] = 1.0
    keypoints[:, schema.right_shoulder, 0] = pixel_x_center + lateral_hip_offset * 1.2
    keypoints[:, schema.right_shoulder, 1] = shoulder_y
    keypoints[:, schema.right_shoulder, 2] = 1.0

    return keypoints, timestamps, world_x


def generate_synthetic_gait(
    duration_sec: float,
    fps: float,
    params: SyntheticGaitParams | None = None,
    keypoint_schema: KeypointSchema | None = None,
) -> Sequence[FrameData]:
    """Generate a synthetic gait trajectory.

    Args:
        duration_sec: Duration of synthetic gait (seconds)
        fps: Frame rate (frames/second)
        params: SyntheticGaitParams configuration (uses defaults if None)
        keypoint_schema: KeypointSchema for output (defaults to RTMPoseWholebody133)

    Returns:
        Sequence of FrameData with synthetic keypoints and positions
    """
    if params is None:
        params = SyntheticGaitParams()

    from stride.core import RTMPoseWholebody133
    if keypoint_schema is None:
        keypoint_schema = RTMPoseWholebody133

    total_frames = int(duration_sec * fps)
    half_duration = duration_sec / 2.0

    keypoints, timestamps, world_x = _generate_synthetic_keypoints(
        duration_sec, fps, params, keypoint_schema
    )

    # Convert to FrameData sequence
    frames = []
    for frame_idx in range(total_frames):
        t = timestamps[frame_idx]
        phase = Phase.TOWARD if t < half_duration else Phase.AWAY

        frame = FrameData(
            frame_idx=frame_idx,
            timestamp=float(t),
            keypoints=keypoints[frame_idx],
            world_position=np.array([world_x[frame_idx], 0.0], dtype=np.float32),
            phase=phase,
            confidence=1.0,
        )
        frames.append(frame)

    return frames


def generate_synthetic_pass1_result(
    duration_sec: float = 10.0,
    fps: float = 30.0,
    params: SyntheticGaitParams | None = None,
) -> Pass1Result:
    """Generate a Pass1Result directly for pipeline integration tests.

    Useful for testing run_pass2 without video I/O.

    Args:
        duration_sec: Duration of synthetic gait (seconds)
        fps: Frame rate (frames/second)
        params: SyntheticGaitParams configuration

    Returns:
        Pass1Result suitable for run_pass2() input
    """
    if params is None:
        params = SyntheticGaitParams()

    from stride.core import RTMPoseWholebody133
    keypoints, timestamps, _ = _generate_synthetic_keypoints(
        duration_sec, fps, params, RTMPoseWholebody133
    )

    return Pass1Result(
        keypoints=keypoints,
        timestamps=timestamps,
        track_ids=np.zeros(len(timestamps), dtype=np.int32),
        fps=fps,
        total_frames=len(timestamps),
        schema=RTMPoseWholebody133,
        video_path="synthetic://test",
    )


def generate_symmetric_gait(duration_sec: float, fps: float) -> Sequence[FrameData]:
    """Generate perfectly symmetric gait (zero asymmetry).

    Useful for testing that asymmetry metrics return ~0%.
    """
    return generate_synthetic_gait(
        duration_sec,
        fps,
        params=SyntheticGaitParams(stride_length_asymmetry_percent=0.0),
    )


def generate_pathological_gait(duration_sec: float, fps: float) -> Sequence[FrameData]:
    """Generate pathological gait with FOG and asymmetry.

    Useful for testing clinical flag generation.
    """
    return generate_synthetic_gait(
        duration_sec,
        fps,
        params=SyntheticGaitParams(
            cadence_steps_per_min=70.0,
            stride_length_m=1.0,
            stride_length_asymmetry_percent=15.0,
            lateral_sway_m=0.06,
            fog_episodes=[(15.0, 17.0), (45.0, 47.5)],
        ),
    )
