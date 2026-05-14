"""Turn detection: compute turning time from velocity and orientation changes.

Identifies the turn point and computes how long the turning took (from slowing down
to resuming speed in opposite direction).
"""

from typing import Tuple

import numpy as np


class TurnDetector:
    """Detects and measures turning behavior.

    Turning time = time spent in TURN phase (near zero velocity).
    """

    def __init__(
        self,
        fps: float,
        velocity_threshold: float = 0.1,
    ):
        """Initialize turn detector.

        Args:
            fps: Frame rate (frames/second)
            velocity_threshold: Velocity magnitude below which we're turning
        """
        self.fps = fps
        self.velocity_threshold = velocity_threshold

    def detect_turn(
        self,
        world_positions: np.ndarray,
        timestamps: np.ndarray,
    ) -> Tuple[float, Tuple[int, int]]:
        """Detect turn and compute turning time.

        Args:
            world_positions: (N, 2) array of [x, y] in meters
            timestamps: (N,) array of timestamps in seconds

        Returns:
            Tuple of (turning_time_seconds, (turn_start_frame, turn_end_frame))
        """
        n_frames = len(world_positions)

        # Compute velocity along X axis
        x_pos = world_positions[:, 0]
        dt = np.diff(timestamps, prepend=timestamps[0])
        dt[0] = 1.0 / self.fps
        velocity_x = np.diff(x_pos, prepend=x_pos[0]) / (dt + 1e-8)

        # Find frames with low velocity (turning phase)
        slow_mask = np.abs(velocity_x) < self.velocity_threshold

        # Find continuous segments of slow velocity
        # (Connect small gaps < 0.5 seconds)
        gap_threshold = int(0.5 * self.fps)
        slow_segments = self._find_segments(slow_mask, gap_threshold)

        if len(slow_segments) == 0:
            return 0.0, (n_frames // 2, n_frames // 2)

        # Find the longest slow segment (likely the turn)
        longest_seg = max(slow_segments, key=lambda seg: seg[1] - seg[0])
        turn_start, turn_end = longest_seg

        turning_time = timestamps[turn_end] - timestamps[turn_start]

        return float(turning_time), (int(turn_start), int(turn_end))

    @staticmethod
    def _find_segments(
        mask: np.ndarray,
        max_gap: int = 10,
    ) -> list[Tuple[int, int]]:
        """Find contiguous segments in a boolean mask, allowing small gaps.

        Args:
            mask: Boolean array
            max_gap: Maximum gap size to bridge

        Returns:
            List of (start, end) tuples for each segment
        """
        # Find True regions
        true_indices = np.where(mask)[0]

        if len(true_indices) == 0:
            return []

        segments = []
        start = true_indices[0]

        for i in range(1, len(true_indices)):
            gap = true_indices[i] - true_indices[i - 1]
            if gap > max_gap:
                # End of segment
                segments.append((start, true_indices[i - 1]))
                start = true_indices[i]

        # Add last segment
        segments.append((start, true_indices[-1]))

        return segments
