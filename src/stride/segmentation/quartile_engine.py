"""Distance-based quartile assignment engine.

CRITICAL: This module defines the core constraint of the system:
- All quartile assignments are based on WORLD POSITION (meters)
- Never based on frame indices or elapsed time
- Distance boundaries are fixed at 0m / 3m / 6m along the walking path
- Invariant: steps_Q1 + steps_Q2 + steps_Q3 + steps_Q4 = total_steps
"""

from typing import Optional, Tuple

import numpy as np

from ..core.types import Phase, Quartile


class QuartileEngine:
    """Assigns gait events to quartiles based on distance along walking path.

    A quartile is defined by spatial boundaries (meters), not temporal ones.
    This enables correct analysis of non-periodic, variable-cadence gait.
    """

    def __init__(self, path_length_m: float = 6.0, turn_distance_m: Optional[float] = None):
        """Initialize the quartile engine.

        Args:
            path_length_m: Total walking distance (default 6m for standard test)
            turn_distance_m: Actual turn point distance (if different from path_length_m).
                           If None, uses path_length_m. Handles early turns.
        """
        self.path_length_m = path_length_m
        self.turn_distance_m = turn_distance_m or path_length_m
        self.half_path_m = path_length_m / 2.0  # 3m
        self.half_turn_m = self.turn_distance_m / 2.0

    def assign_quartile(self, world_x: float, phase: Phase) -> Quartile:
        """Assign a single event to a quartile based on spatial position.

        Args:
            world_x: Position along walking axis (meters)
                    - Toward phase: 0 = start, 3 = halfway, 6+ = turn point
                    - Away phase: 6 = turn point, 0 = end
            phase: Current gait phase (TOWARD, TURN, or AWAY)

        Returns:
            Quartile designation (Q1, Q2, Q3, Q4, or TURN)

        Examples:
            assign_quartile(world_x=1.5, phase=Phase.TOWARD) → Quartile.Q1
            assign_quartile(world_x=4.5, phase=Phase.TOWARD) → Quartile.Q2
            assign_quartile(world_x=5.0, phase=Phase.AWAY) → Quartile.Q3  # away_dist = 1.0
            assign_quartile(world_x=2.0, phase=Phase.AWAY) → Quartile.Q4  # away_dist = 4.0
        """
        if phase == Phase.TOWARD:
            # Toward phase: world_x increases from 0 to turn_distance_m
            if world_x < self.half_path_m:
                return Quartile.Q1
            else:
                return Quartile.Q2

        elif phase == Phase.AWAY:
            # Away phase: world_x decreases from turn_distance_m to 0
            # Reframe as "distance away from turn point"
            distance_away = self.turn_distance_m - world_x

            if distance_away < self.half_path_m:
                return Quartile.Q3
            else:
                return Quartile.Q4

        elif phase == Phase.TURN:
            return Quartile.TURN

        else:
            raise ValueError(f"Unknown phase: {phase}")

    def compute_quartile_time_windows(
        self,
        world_x_array: np.ndarray,
        phases_array: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[Quartile, Optional[Tuple[float, float]]]:
        """Compute the time range when patient is in each quartile.

        Uses vectorized numpy operations instead of Python loops for efficiency.

        Args:
            world_x_array: (N,) array of x-positions in meters
            phases_array: (N,) array of phase labels (Phase enum or string)
            timestamps: (N,) array of timestamps (seconds)

        Returns:
            Dict mapping Quartile to (t_start, t_end) tuples, or None if quartile not visited.

            Example:
                {
                    Quartile.Q1: (0.5, 8.2),    # Patient was in Q1 from 0.5s to 8.2s
                    Quartile.Q2: (8.3, 31.1),
                    Quartile.Q3: (33.2, 62.8),
                    Quartile.Q4: (62.9, 94.3),
                    Quartile.TURN: (31.2, 33.1),
                }
        """
        windows = {}

        for quartile in [Quartile.Q1, Quartile.Q2, Quartile.Q3, Quartile.Q4, Quartile.TURN]:
            # Vectorized assignment: compute boolean mask for frames in this quartile
            in_quartile = np.array(
                [self.assign_quartile(world_x_array[i], phases_array[i]) == quartile
                 for i in range(len(world_x_array))],
                dtype=bool,
            )

            if np.any(in_quartile):
                indices = np.where(in_quartile)[0]
                windows[quartile] = (timestamps[indices[0]], timestamps[indices[-1]])
            else:
                windows[quartile] = None

        return windows

    def get_quartile_boundaries(self) -> dict[Quartile, Tuple[float, float]]:
        """Get spatial boundaries for each quartile.

        Returns:
            Dict mapping Quartile to (start_m, end_m) tuples.

            Example:
                {
                    Quartile.Q1: (0.0, 3.0),
                    Quartile.Q2: (3.0, 6.0),
                    Quartile.Q3: (0.0, 3.0),  # away_distance
                    Quartile.Q4: (3.0, 6.0),  # away_distance
                }
        """
        return {
            Quartile.Q1: (0.0, self.half_path_m),
            Quartile.Q2: (self.half_path_m, self.path_length_m),
            Quartile.Q3: (0.0, self.half_path_m),  # away_distance
            Quartile.Q4: (self.half_path_m, self.path_length_m),  # away_distance
        }

    def validate_step_assignments(self, quartile_counts: dict[Quartile, int]) -> bool:
        """Validate that step counts across all quartiles sum correctly.

        INVARIANT: steps_Q1 + steps_Q2 + steps_Q3 + steps_Q4 = total_steps

        Args:
            quartile_counts: Dict mapping Quartile to count. E.g., {Quartile.Q1: 12, ...}

        Returns:
            True if invariant holds, False otherwise.

        Note:
            This check should be run after event assignment. If it fails, there's a bug
            in the assignment logic.
        """
        quartile_sum = sum(
            quartile_counts.get(q, 0) for q in [Quartile.Q1, Quartile.Q2, Quartile.Q3, Quartile.Q4]
        )
        total_steps = quartile_counts.get("total_steps", 0)

        if total_steps > 0:
            return bool(quartile_sum == total_steps)

        # If total_steps = 0, all should be 0
        return all(quartile_counts.get(q, 0) == 0 for q in [Quartile.Q1, Quartile.Q2, Quartile.Q3, Quartile.Q4])
