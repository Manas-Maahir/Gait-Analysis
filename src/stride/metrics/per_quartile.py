"""Per-quartile metrics computation: step count, cadence, and timing.

Computes metrics for each quartile (Q1, Q2, Q3, Q4) of the 6-meter walk test,
including step counts, cadence (steps/min), and durations.
"""

from typing import Dict

import numpy as np

from stride.core import Quartile, Side
from stride.data.events import FootStrikeEvent
from stride.data.metrics import QuartileMetrics
from stride.segmentation.quartile_engine import QuartileEngine


class QuartileMetricsComputer:
    """Computes per-quartile gait metrics from foot strike events."""

    def __init__(
        self,
        path_length_m: float = 6.0,
        quartile_boundaries_m: tuple = (3.0, 6.0),
    ):
        """Initialize metrics computer.

        Args:
            path_length_m: Total walking distance (meters)
            quartile_boundaries_m: Quartile segment boundaries
        """
        self.path_length_m = path_length_m
        self.quartile_engine = QuartileEngine(path_length_m)

    def compute(
        self,
        events: list[FootStrikeEvent],
        timestamps: np.ndarray,
    ) -> Dict[Quartile, QuartileMetrics]:
        """Compute metrics for each quartile.

        Args:
            events: List of FootStrikeEvent objects
            timestamps: (N,) array of timestamps in seconds

        Returns:
            Dictionary mapping Quartile -> QuartileMetrics
        """
        metrics_dict = {}

        # Initialize metrics for all quartiles
        for quartile in [Quartile.Q1, Quartile.Q2, Quartile.Q3, Quartile.Q4]:
            metrics_dict[quartile] = self._compute_quartile_metrics(
                events, quartile, timestamps
            )

        return metrics_dict

    def _compute_quartile_metrics(
        self,
        events: list[FootStrikeEvent],
        quartile: Quartile,
        timestamps: np.ndarray,
    ) -> QuartileMetrics:
        """Compute metrics for a specific quartile.

        Args:
            events: All foot strike events
            quartile: Quartile to compute metrics for
            timestamps: Frame timestamps

        Returns:
            QuartileMetrics object
        """
        # Filter events by quartile (already assigned by processor before metrics run)
        quartile_events = [
            e for e in events
            if e.quartile == quartile
        ]

        if len(quartile_events) == 0:
            # No steps in this quartile
            return QuartileMetrics(
                quartile=quartile,
                step_count=0,
                cadence_steps_per_min=0.0,
                asymmetry_score=0.0,
                sway_rms_meters=0.0,
                duration_seconds=0.0,
                mean_step_length_m=0.0,
                step_length_cv_percent=0.0,
                mean_step_time_sec=0.0,
                step_time_cv_percent=0.0,
                left_step_count=0,
                right_step_count=0,
                fog_episodes_count=0,
                fog_total_duration_sec=0.0,
            )

        # Count steps by side
        left_steps = [e for e in quartile_events if e.side == Side.L]
        right_steps = [e for e in quartile_events if e.side == Side.R]

        step_count = len(quartile_events)
        left_count = len(left_steps)
        right_count = len(right_steps)

        # Compute duration
        if len(quartile_events) > 1:
            duration = quartile_events[-1].timestamp - quartile_events[0].timestamp
        else:
            duration = 0.0

        # Compute cadence (steps/min)
        if duration > 0:
            cadence = (step_count / duration) * 60
        else:
            cadence = 0.0

        # Compute step lengths and times
        step_lengths = [e.step_length for e in quartile_events if e.step_length > 0]
        step_times = [e.step_time for e in quartile_events if e.step_time > 0]

        if len(step_lengths) > 0:
            mean_step_length = np.mean(step_lengths)
            step_length_cv = (np.std(step_lengths) / mean_step_length * 100) if mean_step_length > 0 else 0.0
        else:
            mean_step_length = 0.0
            step_length_cv = 0.0

        if len(step_times) > 0:
            mean_step_time = np.mean(step_times)
            step_time_cv = (np.std(step_times) / mean_step_time * 100) if mean_step_time > 0 else 0.0
        else:
            mean_step_time = 0.0
            step_time_cv = 0.0

        # Compute mean confidence
        confidence = np.mean([e.confidence for e in quartile_events]) if quartile_events else 0.0

        return QuartileMetrics(
            quartile=quartile,
            step_count=step_count,
            cadence_steps_per_min=float(cadence),
            asymmetry_score=0.0,  # Computed in Phase 2
            sway_rms_meters=0.0,  # Computed in Phase 2
            duration_seconds=float(duration),
            mean_step_length_m=float(mean_step_length),
            step_length_cv_percent=float(step_length_cv),
            mean_step_time_sec=float(mean_step_time),
            step_time_cv_percent=float(step_time_cv),
            left_step_count=left_count,
            right_step_count=right_count,
            fog_episodes_count=0,  # Computed in Phase 2
            fog_total_duration_sec=0.0,  # Computed in Phase 2
        )
