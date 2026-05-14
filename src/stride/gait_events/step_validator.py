"""Step event validation: plausibility checks on detected foot strikes."""

from typing import Optional

import numpy as np

from stride.core.types import Phase
from stride.data.events import FootStrikeEvent


class StepValidator:
    """Validates detected foot strike events for temporal and spatial plausibility.

    Removes impossible steps: overlapping, backwards steps, very short steps, etc.
    """

    def __init__(
        self,
        min_step_interval_sec: float = 0.2,
        max_step_interval_sec: float = 2.0,
        min_step_length_m: float = 0.1,
        max_step_length_m: float = 2.5,
    ):
        """Initialize validator.

        Args:
            min_step_interval_sec: Minimum time between consecutive steps
            max_step_interval_sec: Maximum time between consecutive steps
            min_step_length_m: Minimum forward distance for a valid step
            max_step_length_m: Maximum forward distance for a valid step
        """
        self.min_interval = min_step_interval_sec
        self.max_interval = max_step_interval_sec
        self.min_length = min_step_length_m
        self.max_length = max_step_length_m

    def validate(
        self,
        events: list[FootStrikeEvent],
    ) -> tuple[list[FootStrikeEvent], list[tuple[int, str]]]:
        """Validate foot strike events.

        Args:
            events: List of FootStrikeEvent objects

        Returns:
            Tuple of (valid_events, rejected_events)
            where rejected_events is a list of (event_index, reason) tuples
        """
        if len(events) == 0:
            return [], []

        valid = []
        rejected = []

        for i, event in enumerate(events):
            reason = self._check_event_validity(event, i, events)

            if reason is None:
                valid.append(event)
            else:
                rejected.append((i, reason))

        return valid, rejected

    def _check_event_validity(
        self,
        event: FootStrikeEvent,
        event_idx: int,
        all_events: list[FootStrikeEvent],
    ) -> Optional[str]:
        """Check if a single event is valid.

        Args:
            event: Event to check
            event_idx: Index in events list
            all_events: All events for context

        Returns:
            None if valid, or a reason string if invalid
        """
        # Check interval from previous event
        if event_idx > 0:
            prev_event = all_events[event_idx - 1]
            interval = event.timestamp - prev_event.timestamp

            if interval < self.min_interval:
                return f"too_soon ({interval:.3f}s < {self.min_interval}s)"

            if interval > self.max_interval:
                return f"too_long ({interval:.3f}s > {self.max_interval}s)"

        # Check step length
        if event.step_length < self.min_length:
            return f"too_short ({event.step_length:.3f}m < {self.min_length}m)"

        if event.step_length > self.max_length:
            return f"too_long ({event.step_length:.3f}m > {self.max_length}m)"

        # Check backward step — only applies to TOWARD phase.
        # During AWAY, world_x legitimately decreases (walking back to start).
        if event_idx > 0:
            prev_x = all_events[event_idx - 1].world_x
            is_away = event.detection_phase is Phase.AWAY
            if not is_away and event.world_x < prev_x and event.step_length > 0.05:
                return "backward_step"

        return None
