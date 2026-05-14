"""Phase detection: classifying motion as TOWARD / TURN / AWAY based on velocity.

Detects phase transitions using velocity sign changes along the walking axis.
Robust to multiple zero-crossings (hesitations) by selecting the one at maximum distance.
"""

from typing import Tuple

import numpy as np

from stride.core import Phase


class PhaseDetector:
    """Detects gait phases (TOWARD/TURN/AWAY) from world-space trajectories.

    Algorithm:
    1. Compute velocity along walking axis (X direction, assuming straight 6m path)
    2. Find zero-crossings (velocity sign changes)
    3. Classify frames as TOWARD (positive velocity), AWAY (negative velocity), or TURN (near zero-crossing)
    """

    def __init__(
        self,
        fps: float,
        turn_window_sec: float = 0.5,
        velocity_zero_threshold: float = 0.1,
    ):
        """Initialize phase detector.

        Args:
            fps: Frame rate (frames/second)
            turn_window_sec: Time window around zero-crossing classified as TURN
            velocity_zero_threshold: Velocity magnitude below which we're in TURN phase
        """
        self.fps = fps
        self.turn_window_frames = int(turn_window_sec * fps)
        self.velocity_zero_threshold = velocity_zero_threshold

    def detect(
        self,
        world_positions: np.ndarray,
        timestamps: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Detect phases from world-space trajectory.

        Args:
            world_positions: (N, 2) array of [x, y] in meters
            timestamps: (N,) array of timestamps in seconds

        Returns:
            Tuple of (phase_array, (turn_start_frame, turn_end_frame))
            where phase_array is (N,) array of Phase values
        """
        n_frames = len(world_positions)

        # Compute velocity along X axis
        x_pos = world_positions[:, 0]
        dt = np.diff(timestamps, prepend=timestamps[0])  # First dt is 0, then real dts
        dt[0] = 1.0 / self.fps  # Set first dt
        velocity_x = np.diff(x_pos, prepend=x_pos[0]) / (dt + 1e-8)

        # Find zero-crossings (velocity sign changes)
        sign_changes = np.where(np.diff(np.sign(velocity_x)))[0]

        if len(sign_changes) == 0:
            # No turn detected; classify by velocity sign
            phases = np.where(velocity_x > 0, Phase.TOWARD, Phase.AWAY)
            return phases, (n_frames // 2, n_frames // 2)

        # Find the zero-crossing at maximum forward distance
        # (This filters out hesitations/reversals)
        zero_x_values = x_pos[sign_changes]
        turn_idx = sign_changes[np.argmax(zero_x_values)]

        # Classify frames
        phases = np.empty(n_frames, dtype=object)

        # TOWARD phase: before turn, positive velocity
        toward_mask = np.arange(n_frames) < turn_idx
        phases[toward_mask] = Phase.TOWARD

        # AWAY phase: after turn, negative velocity
        away_mask = np.arange(n_frames) > turn_idx
        phases[away_mask] = Phase.AWAY

        # TURN phase: near turn point
        turn_start = max(0, turn_idx - self.turn_window_frames)
        turn_end = min(n_frames - 1, turn_idx + self.turn_window_frames)
        phases[turn_start : turn_end + 1] = Phase.TURN

        # Convert to Phase enum if needed
        phases = np.array([p if isinstance(p, Phase) else Phase.TOWARD for p in phases])

        return phases, (turn_start, turn_end)
