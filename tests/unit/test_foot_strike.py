"""Unit tests for FootStrikeDetector and StepValidator.

Uses synthetic sinusoidal ankle trajectories — no video I/O required.
"""

import numpy as np
import pytest

from stride.core import RTMPoseWholebody133
from stride.data.events import FootStrikeEvent
from stride.gait_events.foot_strike import FootStrikeDetector
from stride.gait_events.step_validator import StepValidator


# ── Module-level constants ──────────────────────────────────────────────────

SCHEMA = RTMPoseWholebody133
FPS = 30.0


# ── Signal-generation helpers ───────────────────────────────────────────────

def _make_kp(
    left_y: np.ndarray,
    right_y: np.ndarray,
    left_conf: np.ndarray | None = None,
    right_conf: np.ndarray | None = None,
) -> np.ndarray:
    """Build (N, 133, 3) keypoints array with only ankle slots populated."""
    n = len(left_y)
    kp = np.zeros((n, SCHEMA.n_keypoints, 3), dtype=np.float32)
    if left_conf is None:
        left_conf = np.ones(n, dtype=np.float32)
    if right_conf is None:
        right_conf = np.ones(n, dtype=np.float32)
    kp[:, SCHEMA.left_ankle, 1] = left_y
    kp[:, SCHEMA.left_ankle, 2] = left_conf
    kp[:, SCHEMA.right_ankle, 1] = right_y
    kp[:, SCHEMA.right_ankle, 2] = right_conf
    return kp


def _sine_ankles(
    n: int, freq_hz: float, amp: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    """Alternating sinusoidal left/right ankle Y signals (right offset by π)."""
    t = np.arange(n, dtype=np.float32) / FPS
    center = 400.0
    left_y = (center + amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    right_y = (center + amp * np.sin(2 * np.pi * freq_hz * t + np.pi)).astype(np.float32)
    return left_y, right_y


def _world(n: int, x_end: float = 3.0) -> np.ndarray:
    """(N, 2) world positions with linearly increasing X in [0, x_end]."""
    wx = np.linspace(0.0, x_end, n, dtype=np.float32)
    return np.stack([wx, np.zeros(n, dtype=np.float32)], axis=1)


def _ts(n: int) -> np.ndarray:
    """(N,) timestamps at FPS."""
    return np.arange(n, dtype=np.float32) / FPS


def _detector(min_len: float = 0.0) -> FootStrikeDetector:
    """FootStrikeDetector with configurable min step length (0 = no length gate)."""
    return FootStrikeDetector(
        fps=FPS,
        min_step_interval_sec=0.2,
        max_step_interval_sec=2.0,
        min_step_length_m=min_len,
        max_step_length_m=10.0,
        peak_distance_sec=0.2,
    )


# ── FootStrikeDetector tests ────────────────────────────────────────────────

class TestFootStrikeDetector:

    def test_detects_peaks_in_sinusoidal_signal(self):
        """1.5 Hz × 3s → ~9 foot-strike events (±2).

        Left ankle: 4 troughs; right ankle (π-offset): 5 troughs → 9 total.
        The detector finds troughs (invert=True) as foot-contact proxies.
        """
        n = int(3.0 * FPS)  # 90 frames
        left_y, right_y = _sine_ankles(n, freq_hz=1.5, amp=30.0)
        events = _detector().detect(
            _make_kp(left_y, right_y), _world(n), _ts(n), SCHEMA
        )
        assert 7 <= len(events) <= 11, f"Expected ≈9 events, got {len(events)}"

    def test_returns_empty_for_flat_signal(self):
        """Constant ankle Y (shuffling) → signal range < threshold → no events."""
        n = 90
        flat = np.full(n, 400.0, dtype=np.float32)
        events = _detector().detect(
            _make_kp(flat, flat), _world(n), _ts(n), SCHEMA
        )
        assert events == []

    def test_events_sorted_by_timestamp(self):
        """Returned events must be in strict chronological order."""
        n = 90
        left_y, right_y = _sine_ankles(n, 1.5)
        events = _detector().detect(
            _make_kp(left_y, right_y), _world(n), _ts(n), SCHEMA
        )
        if len(events) > 1:
            ts = [e.timestamp for e in events]
            assert ts == sorted(ts), "Events not sorted by timestamp"

    def test_event_fields_populated(self):
        """All FootStrikeEvent fields must hold valid values after detection."""
        n = 90
        left_y, right_y = _sine_ankles(n, 1.5)
        events = _detector().detect(
            _make_kp(left_y, right_y), _world(n, x_end=6.0), _ts(n), SCHEMA
        )
        assert len(events) > 0, "Need at least one event for field validation"
        for e in events:
            assert isinstance(e.frame_idx, int) and e.frame_idx >= 0
            assert e.timestamp >= 0.0
            assert e.side in {"L", "R"}
            assert 0.0 <= e.confidence <= 1.0
            assert e.step_length is not None and e.step_length >= 0.0

    def test_low_confidence_frames_ignored(self):
        """All-zero confidence → no valid ankle data → empty event list."""
        n = 90
        left_y, right_y = _sine_ankles(n, 1.5, amp=30.0)
        zero_conf = np.zeros(n, dtype=np.float32)
        events = _detector().detect(
            _make_kp(left_y, right_y, left_conf=zero_conf, right_conf=zero_conf),
            _world(n),
            _ts(n),
            SCHEMA,
        )
        assert events == []


# ── StepValidator tests ─────────────────────────────────────────────────────

def _event(
    frame_idx: int,
    timestamp: float,
    world_x: float,
    step_length: float = 0.5,
    side: str = "L",
) -> FootStrikeEvent:
    """Minimal FootStrikeEvent for validator unit tests."""
    return FootStrikeEvent(
        frame_idx=frame_idx,
        timestamp=timestamp,
        side=side,
        world_x=world_x,
        world_y=0.0,
        confidence=1.0,
        step_length=step_length,
        step_time=timestamp,  # simplification; validator uses timestamp directly
    )


class TestStepValidator:

    def test_valid_events_pass_through(self):
        """Well-spaced events with valid step lengths should all pass.

        Event 0 has step_length=0 and is expected to be rejected (too_short).
        Events 1–3 are all valid and must appear in the output.
        """
        validator = StepValidator(
            min_step_interval_sec=0.3,
            max_step_interval_sec=2.0,
            min_step_length_m=0.1,
            max_step_length_m=2.5,
        )
        events = [
            _event(0, 0.0, 0.0, step_length=0.0),   # rejected: too_short (0 < 0.1m)
            _event(18, 0.6, 0.5, step_length=0.5),  # valid
            _event(36, 1.2, 1.0, step_length=0.5),  # valid
            _event(54, 1.8, 1.5, step_length=0.5),  # valid
        ]
        valid, rejected = validator.validate(events)
        assert len(valid) == 3
        assert len(rejected) == 1

    def test_too_close_events_rejected(self):
        """Events closer than min_interval_sec are labelled 'too_soon'."""
        validator = StepValidator(
            min_step_interval_sec=0.3,
            max_step_interval_sec=2.0,
            min_step_length_m=0.0,
            max_step_length_m=10.0,
        )
        events = [
            _event(0, 0.0, 0.0, step_length=0.5),   # valid
            _event(3, 0.1, 0.2, step_length=0.2),   # 0.1s < 0.3s → too_soon
            _event(30, 1.0, 0.8, step_length=0.5),  # valid
        ]
        valid, rejected = validator.validate(events)
        assert len(valid) == 2
        assert any("too_soon" in reason for _, reason in rejected)

    def test_too_short_step_rejected(self):
        """Steps shorter than min_step_length_m are labelled 'too_short'."""
        validator = StepValidator(
            min_step_interval_sec=0.1,
            max_step_interval_sec=2.0,
            min_step_length_m=0.2,
            max_step_length_m=10.0,
        )
        events = [
            _event(0, 0.00, 0.0, step_length=0.5),   # valid
            _event(20, 0.67, 0.5, step_length=0.05),  # 0.05m < 0.2m → too_short
            _event(40, 1.33, 1.0, step_length=0.5),   # valid
        ]
        valid, rejected = validator.validate(events)
        assert len(valid) == 2
        assert any("too_short" in reason for _, reason in rejected)
