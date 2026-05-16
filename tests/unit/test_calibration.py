"""Unit tests for SVDAutoCalibrator and OneEuroFilter correctness.

Regression tests covering:
- SVD world_x anchoring: output must be in [0, path_length] not centered at 0.
- SVD direction sign: TOWARD walk must produce increasing world_x.
- OneEuro alpha formula: large-fc → large alpha (responsive), small-fc → small alpha (smooth).
"""

import math

import numpy as np
import pytest

from stride.calibration.homography import SVDAutoCalibrator
from stride.calibration.spatial_mapper import SpatialMapper
from stride.pose.smoother import OneEuroFilter


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_toward_turn_away_trajectory(
    n_frames: int = 400,
    start_px: float = 100.0,
    peak_px: float = 400.0,
    lateral_px: float = 240.0,
    noise_std: float = 0.0,
) -> np.ndarray:
    """Simulate ankle pixel trajectory for a 6m walk-toward-turn-walk-away test.

    Returns (n_frames, 2) array where:
    - First half: ankle_y increases from start_px → peak_px (approaching camera)
    - Second half: ankle_y decreases from peak_px → start_px (walking away)
    - ankle_x is constant at lateral_px (straight-line walk)
    """
    half = n_frames // 2
    y_toward = np.linspace(start_px, peak_px, half)
    y_away = np.linspace(peak_px, start_px, n_frames - half)
    y = np.concatenate([y_toward, y_away])
    x = np.full(n_frames, lateral_px)
    pts = np.column_stack([x, y]).astype(np.float32)
    if noise_std > 0:
        pts += np.random.default_rng(42).normal(0, noise_std, pts.shape).astype(np.float32)
    return pts


# ── SVDAutoCalibrator: anchoring ──────────────────────────────────────────────

class TestSVDAnchoring:

    def test_world_x_starts_at_zero(self):
        """world_x minimum must be 0 (not negative) for a canonical trajectory."""
        pts = _make_toward_turn_away_trajectory()
        cal = SVDAutoCalibrator(path_length_m=6.0).calibrate(pts)
        wx = SpatialMapper(cal).image_to_world(pts)[:, 0]
        assert wx.min() == pytest.approx(0.0, abs=0.05), (
            f"world_x min={wx.min():.3f} — SVD should anchor start to 0, not centre at mean"
        )

    def test_world_x_max_equals_path_length(self):
        """world_x maximum must equal path_length_m (6.0 m by default)."""
        pts = _make_toward_turn_away_trajectory()
        cal = SVDAutoCalibrator(path_length_m=6.0).calibrate(pts)
        wx = SpatialMapper(cal).image_to_world(pts)[:, 0]
        assert wx.max() == pytest.approx(6.0, abs=0.05)

    def test_world_x_range_equals_path_length(self):
        """world_x span must equal path_length_m regardless of pixel scale."""
        pts = _make_toward_turn_away_trajectory(start_px=50, peak_px=800)
        cal = SVDAutoCalibrator(path_length_m=6.0).calibrate(pts)
        wx = SpatialMapper(cal).image_to_world(pts)[:, 0]
        assert (wx.max() - wx.min()) == pytest.approx(6.0, abs=0.05)

    def test_toward_phase_world_x_increases(self):
        """During the TOWARD phase (first half), world_x must increase monotonically."""
        pts = _make_toward_turn_away_trajectory(n_frames=400)
        cal = SVDAutoCalibrator(path_length_m=6.0).calibrate(pts)
        wx = SpatialMapper(cal).image_to_world(pts)[:, 0]
        toward = wx[:200]
        diffs = np.diff(toward)
        assert np.all(diffs >= -0.01), (
            "TOWARD world_x should be non-decreasing; direction sign may be flipped"
        )

    def test_away_phase_world_x_decreases(self):
        """During the AWAY phase (second half), world_x must decrease monotonically."""
        pts = _make_toward_turn_away_trajectory(n_frames=400)
        cal = SVDAutoCalibrator(path_length_m=6.0).calibrate(pts)
        wx = SpatialMapper(cal).image_to_world(pts)[:, 0]
        away = wx[200:]
        diffs = np.diff(away)
        assert np.all(diffs <= 0.01), (
            "AWAY world_x should be non-increasing; direction sign may be flipped"
        )

    def test_no_negative_world_x(self):
        """All world_x values must be non-negative after anchoring."""
        pts = _make_toward_turn_away_trajectory()
        cal = SVDAutoCalibrator(path_length_m=6.0).calibrate(pts)
        wx = SpatialMapper(cal).image_to_world(pts)[:, 0]
        assert np.all(wx >= -0.05), f"world_x has values below 0: min={wx.min():.3f}"

    def test_custom_path_length(self):
        """path_length_m parameter must set the world_x range correctly."""
        pts = _make_toward_turn_away_trajectory()
        cal = SVDAutoCalibrator(path_length_m=10.0).calibrate(pts)
        wx = SpatialMapper(cal).image_to_world(pts)[:, 0]
        assert wx.max() == pytest.approx(10.0, abs=0.1)
        assert wx.min() == pytest.approx(0.0, abs=0.1)

    def test_noisy_trajectory_still_anchored(self):
        """Anchoring must hold under realistic ankle-position noise."""
        pts = _make_toward_turn_away_trajectory(n_frames=500, noise_std=5.0)
        cal = SVDAutoCalibrator(path_length_m=6.0).calibrate(pts)
        wx = SpatialMapper(cal).image_to_world(pts)[:, 0]
        # Allow ±0.1 m tolerance under noise
        assert wx.min() == pytest.approx(0.0, abs=0.15)
        assert wx.max() == pytest.approx(6.0, abs=0.15)


# ── OneEuroFilter: alpha correctness ─────────────────────────────────────────

class TestOneEuroAlpha:

    def test_low_fc_gives_small_alpha(self):
        """At fc=1 Hz the EMA coefficient must be small (≈0.095), not large."""
        f = OneEuroFilter(fps=60.0, fcmin=1.0, beta=0.0)
        cutoff = 2 * math.pi * 1.0  # 1 Hz in rad/s
        alpha = f._alpha(cutoff)
        expected = (cutoff / 60) / (1 + cutoff / 60)
        assert alpha == pytest.approx(expected, rel=1e-4)
        assert alpha < 0.15, (
            f"At fc=1 Hz, alpha={alpha:.3f} should be small (smooth), not {alpha:.3f}. "
            "OneEuro alpha formula may be inverted."
        )

    def test_high_fc_gives_large_alpha(self):
        """At fc=50 Hz the EMA coefficient must be large (≈0.84), not small."""
        f = OneEuroFilter(fps=60.0, fcmin=1.0, beta=0.0)
        cutoff = 2 * math.pi * 50.0  # 50 Hz in rad/s
        alpha = f._alpha(cutoff)
        assert alpha > 0.8, (
            f"At fc=50 Hz, alpha={alpha:.3f} should be large (responsive). "
            "OneEuro alpha formula may be inverted."
        )

    def test_step_response_at_high_speed(self):
        """Filter must track a sudden step change in 1-2 frames when fc is high."""
        f = OneEuroFilter(fps=60.0, fcmin=1.0, beta=0.5)
        # Feed zeros, then a 100 px jump — large derivative → high fc → responsive
        out = []
        for i in range(20):
            v = np.array([0.0, 0.0]) if i < 5 else np.array([100.0, 0.0])
            out.append(float(f.filter(v)[0]))
        # After 2 frames from the jump (frames 6 and 7), should be above 90
        assert out[6] > 90.0, (
            f"Filter should track a jump quickly; got {out[6]:.1f} at frame 6. "
            "OneEuro is over-smoothing fast motion."
        )

    def test_stable_signal_stays_smooth(self):
        """A constant signal must not drift under the filter."""
        f = OneEuroFilter(fps=60.0, fcmin=1.0, beta=0.5)
        const = np.array([50.0, 50.0])
        for _ in range(100):
            out = f.filter(const)
        assert float(out[0]) == pytest.approx(50.0, abs=0.1)
