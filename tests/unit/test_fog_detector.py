"""Unit tests for FOGDetector.

Tests cover:
- compute_freeze_index_signal: frequency-band discrimination, output length, non-negativity
- detect: end-to-end episode detection from keypoints
- _detect_episodes: episode boundary logic, min-duration filtering, multiple episodes
"""

import numpy as np
import pytest

from stride.gait_events.fog_detector import FOGDetector
from stride.core import RTMPoseWholebody133

FPS = 30.0
SCHEMA = RTMPoseWholebody133   # module-level singleton instance


# ── helpers ───────────────────────────────────────────────────────────────────

def _sine_velocity(freq_hz: float, n_frames: int, amplitude: float = 1.0) -> np.ndarray:
    """Pure-frequency sinusoidal velocity signal."""
    t = np.arange(n_frames) / FPS
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def _make_keypoints(ankle_y: np.ndarray):
    """Minimal keypoints array with given ankle y trajectory (full confidence)."""
    schema = SCHEMA
    N = len(ankle_y)
    kps = np.zeros((N, schema.n_keypoints, 3), dtype=np.float32)
    kps[:, schema.left_ankle, 1] = ankle_y.astype(np.float32)
    kps[:, schema.right_ankle, 1] = ankle_y.astype(np.float32)
    kps[:, schema.left_ankle, 2] = 1.0
    kps[:, schema.right_ankle, 2] = 1.0
    timestamps = np.arange(N, dtype=np.float32) / FPS
    return kps, timestamps, schema


# ── TestComputeFreezeIndexSignal ──────────────────────────────────────────────

class TestComputeFreezeIndexSignal:
    def test_freeze_band_signal_has_high_fi(self):
        """5 Hz velocity (inside freeze band [3,8] Hz) → FI > 1.0 in covered windows."""
        detector = FOGDetector(fps=FPS)
        fi = detector.compute_freeze_index_signal(_sine_velocity(5.0, 150))
        # Expect at least some windows with clearly elevated FI
        assert fi.max() > 1.0, f"Expected FI > 1.0, got max={fi.max():.4f}"

    def test_loco_band_signal_has_low_fi(self):
        """1 Hz velocity (inside loco band [0.5,3] Hz) → FI < threshold (2.5)."""
        detector = FOGDetector(fps=FPS)
        fi = detector.compute_freeze_index_signal(_sine_velocity(1.0, 150))
        assert fi.max() < 2.5, f"Expected FI < 2.5, got max={fi.max():.4f}"

    def test_output_length_equals_input_length(self):
        """FI array must always be the same length as the input velocity signal."""
        detector = FOGDetector(fps=FPS)
        for n in [30, 60, 90, 150]:
            fi = detector.compute_freeze_index_signal(_sine_velocity(1.0, n))
            assert len(fi) == n

    def test_fi_is_non_negative(self):
        """FI is a power ratio — must be >= 0 for any signal."""
        detector = FOGDetector(fps=FPS)
        rng = np.random.default_rng(42)
        fi = detector.compute_freeze_index_signal(rng.standard_normal(150))
        assert (fi >= 0).all()

    def test_empty_signal_returns_empty_array(self):
        """Empty velocity signal → empty FI array, no crash."""
        detector = FOGDetector(fps=FPS)
        fi = detector.compute_freeze_index_signal(np.array([]))
        assert len(fi) == 0

    def test_signal_shorter_than_window_returns_zeros(self):
        """Signal shorter than one window → no windows processed → all zeros."""
        detector = FOGDetector(fps=FPS, window_sec=2.0)
        # 2-sec window at 30 fps = 60 frames; use 30 frames (half window)
        fi = detector.compute_freeze_index_signal(_sine_velocity(5.0, 30))
        # No full window can be formed → fi_values stay at zero initialisation
        assert (fi == 0.0).all()


# ── TestFOGDetectorDetect ─────────────────────────────────────────────────────

class TestFOGDetectorDetect:
    def test_freeze_band_ankle_yields_episodes(self):
        """5 Hz ankle oscillation for 5 seconds → at least one FOG episode."""
        detector = FOGDetector(fps=FPS, fi_threshold=2.5, window_sec=2.0, min_duration_sec=0.5)
        ankle_y = np.sin(2 * np.pi * 5.0 * np.arange(150) / FPS)
        kps, ts, schema = _make_keypoints(ankle_y)
        episodes = detector.detect(kps, ts, schema)
        assert len(episodes) >= 1

    def test_loco_band_ankle_no_fog(self):
        """1 Hz ankle oscillation (normal cadence) → no FOG episodes."""
        detector = FOGDetector(fps=FPS, fi_threshold=2.5, window_sec=2.0, min_duration_sec=0.5)
        ankle_y = np.sin(2 * np.pi * 1.0 * np.arange(150) / FPS)
        kps, ts, schema = _make_keypoints(ankle_y)
        episodes = detector.detect(kps, ts, schema)
        assert len(episodes) == 0

    def test_empty_keypoints_returns_empty_list(self):
        """Empty keypoints array → empty episode list, no crash."""
        detector = FOGDetector(fps=FPS)
        schema = SCHEMA
        episodes = detector.detect(
            np.zeros((0, schema.n_keypoints, 3), dtype=np.float32),
            np.array([], dtype=np.float32),
            schema,
        )
        assert episodes == []

    def test_zero_confidence_ankles_no_fog(self):
        """All-zero confidence → flat zero velocity → FI = 0 → no episodes."""
        detector = FOGDetector(fps=FPS, fi_threshold=2.5)
        schema = SCHEMA
        N = 150
        kps = np.zeros((N, schema.n_keypoints, 3), dtype=np.float32)
        # Leave confidence at 0 (default); ankle y also 0 → flat velocity
        ts = np.arange(N, dtype=np.float32) / FPS
        episodes = detector.detect(kps, ts, schema)
        assert len(episodes) == 0

    def test_episode_fields_internally_consistent(self):
        """Each detected episode must have valid frame indices, duration, severity."""
        detector = FOGDetector(fps=FPS, fi_threshold=2.5, window_sec=2.0, min_duration_sec=0.5)
        ankle_y = np.sin(2 * np.pi * 5.0 * np.arange(150) / FPS)
        kps, ts, schema = _make_keypoints(ankle_y)
        episodes = detector.detect(kps, ts, schema)
        assert len(episodes) >= 1
        for ep in episodes:
            assert ep.start_frame >= 0
            assert ep.end_frame >= ep.start_frame
            assert ep.duration_sec >= 0.0
            assert ep.severity >= 0.0

    def test_min_duration_filters_short_bursts(self):
        """min_duration_sec longer than total signal → no episodes reported."""
        # With min_duration_sec=10.0, a 75-frame (2.5 s) signal can never qualify
        detector = FOGDetector(fps=FPS, fi_threshold=0.001, window_sec=2.0, min_duration_sec=10.0)
        ankle_y = np.sin(2 * np.pi * 5.0 * np.arange(75) / FPS)
        kps, ts, schema = _make_keypoints(ankle_y)
        episodes = detector.detect(kps, ts, schema)
        assert len(episodes) == 0


# ── TestDetectEpisodes (boundary logic via private method) ────────────────────

class TestDetectEpisodes:
    """Test _detect_episodes directly with hand-crafted FI arrays."""

    def _detector(self) -> FOGDetector:
        return FOGDetector(fps=FPS, fi_threshold=2.5, min_duration_sec=0.5)

    def test_single_long_run_produces_one_episode(self):
        """Frames 20–59 above threshold (40 frames = 1.33 s) → exactly one episode."""
        d = self._detector()
        N = 90
        fi = np.zeros(N, dtype=np.float64)
        fi[20:60] = 5.0
        ts = np.arange(N, dtype=np.float64) / FPS
        episodes = d._detect_episodes(fi, ts)
        assert len(episodes) == 1
        assert episodes[0].start_frame == 20
        assert episodes[0].end_frame == 59
        assert episodes[0].severity == pytest.approx(5.0)

    def test_two_non_contiguous_runs_produce_two_episodes(self):
        """Two separated high-FI runs → two separate episodes."""
        d = self._detector()
        N = 150
        fi = np.zeros(N, dtype=np.float64)
        fi[10:30] = 3.0    # 20 frames = 0.67 s ≥ 0.5 s min
        fi[80:110] = 4.5   # 30 frames = 1.00 s
        ts = np.arange(N, dtype=np.float64) / FPS
        episodes = d._detect_episodes(fi, ts)
        assert len(episodes) == 2
        assert episodes[0].start_frame == 10
        assert episodes[1].start_frame == 80

    def test_run_below_min_duration_not_reported(self):
        """3-frame (0.1 s) run is below min_duration_sec (0.5 s) → no episodes."""
        d = self._detector()
        N = 60
        fi = np.zeros(N, dtype=np.float64)
        fi[20:23] = 5.0   # 3 frames = 0.1 s
        ts = np.arange(N, dtype=np.float64) / FPS
        episodes = d._detect_episodes(fi, ts)
        assert len(episodes) == 0

    def test_all_zeros_no_episodes(self):
        """All-zero FI → no threshold crossings → no episodes."""
        d = self._detector()
        N = 90
        episodes = d._detect_episodes(np.zeros(N), np.arange(N, dtype=np.float64) / FPS)
        assert len(episodes) == 0

    def test_empty_fi_no_crash(self):
        """Empty FI array → empty list, no crash."""
        d = self._detector()
        episodes = d._detect_episodes(np.array([]), np.array([]))
        assert episodes == []

    def test_episode_at_signal_end_included(self):
        """FOG episode that runs all the way to the last frame is still reported."""
        d = self._detector()
        N = 60
        fi = np.zeros(N, dtype=np.float64)
        fi[40:60] = 3.0   # 20 frames = 0.67 s ≥ 0.5 s min
        ts = np.arange(N, dtype=np.float64) / FPS
        episodes = d._detect_episodes(fi, ts)
        assert len(episodes) == 1
        assert episodes[0].end_frame == 59
