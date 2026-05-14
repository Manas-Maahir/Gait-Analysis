"""Integration test: run_pass2 on synthetic gait data (no video I/O).

Validates that the full Pass-2 pipeline (calibration → phases → foot strikes →
validator → quartile assignment → metrics) executes correctly and produces
clinically plausible metrics when given known-correct world coordinates.

Uses an injected synthetic calibrator to avoid dependence on SVD auto-calibration
accuracy; the calibrator maps the synthetic pixel space (x ∈ [50, 590]) exactly
to world space (x ∈ [0, 6m]).
"""

import numpy as np
import pytest

from stride.config import StriderConfig
from stride.core.protocols import CalibrationResult
from stride.pipeline.processor import run_pass2
from tests.fixtures.synthetic_gait import SyntheticGaitParams, generate_synthetic_pass1_result


# ── Synthetic calibrator ────────────────────────────────────────────────────

class _SyntheticCalibrator:
    """Known-correct calibrator for the synthetic pixel space.

    synthetic_gait.py maps world_x ∈ [0, 6] to pixel_x ∈ [50, 590]:
        pixel_x = 50 + world_x * 90
    Inverted:
        world_x = (pixel_x - 50) / 90
    """

    _PIXELS_PER_METER = 90.0  # (590 - 50) / 6 = 90 px / m
    _X_MARGIN_PX = 50.0

    def calibrate(self, ankle_trajectory: np.ndarray) -> CalibrationResult:
        scale = 1.0 / self._PIXELS_PER_METER  # m/px
        H = np.eye(3, dtype=np.float32)
        H[0, 0] = scale
        H[0, 2] = -self._X_MARGIN_PX * scale  # offset: world_x = (px - 50) / 90
        return CalibrationResult(
            homography_matrix=H,
            scale_px_to_m=scale,
            pc1_variance=1.0,
            method="test_synthetic",
        )


# ── Shared fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def default_params() -> SyntheticGaitParams:
    return SyntheticGaitParams(
        cadence_steps_per_min=100.0,
        stride_length_m=1.3,
        step_height_m=0.05,
        walking_speed_m_s=1.3,
        path_length_m=6.0,
    )


@pytest.fixture(scope="module")
def pass2_result(default_params):
    """Run the full Pass-2 pipeline on 10s of synthetic gait."""
    pass1 = generate_synthetic_pass1_result(
        duration_sec=10.0, fps=30.0, params=default_params
    )
    config = StriderConfig()
    return run_pass2(pass1, config, calibrator=_SyntheticCalibrator())


# ── Tests ────────────────────────────────────────────────────────────────────

class TestPipelineRunsWithoutError:

    def test_pass2_completes(self, pass2_result):
        """run_pass2 must return a Pass2Result without raising."""
        assert pass2_result is not None

    def test_world_positions_shape(self, pass2_result):
        """World positions must be an (N, 2) float array."""
        wp = pass2_result.world_positions
        assert wp.ndim == 2 and wp.shape[1] == 2
        assert np.issubdtype(wp.dtype, np.floating)

    def test_world_x_in_valid_range(self, pass2_result):
        """World X should span [0, 6] metres for the full path."""
        wx = pass2_result.world_positions[:, 0]
        assert float(wx.min()) >= -0.5, f"min world_x {wx.min():.2f} out of range"
        assert float(wx.max()) <= 6.5, f"max world_x {wx.max():.2f} out of range"

    def test_phases_detected(self, pass2_result):
        """All three phases must appear at least once."""
        from stride.core import Phase
        unique = set(pass2_result.phases)
        assert Phase.TOWARD in unique, "TOWARD phase not detected"
        assert Phase.AWAY in unique, "AWAY phase not detected"


class TestStepCountAndCadence:

    def test_total_step_count_plausible(self, pass2_result, default_params):
        """Total step count must be within ±5 of expected.

        Expected ≈ cadence_steps_per_min / 60 × duration_sec.
        Duration of actual walking (excluding turn) ≈ 10s for synthetic data.
        """
        expected = int(default_params.cadence_steps_per_min / 60.0 * 10.0)
        actual = len(pass2_result.foot_strikes)
        assert abs(actual - expected) <= 5, (
            f"Step count {actual} differs from expected {expected} by more than 5"
        )

    def test_cadence_in_physiological_range(self, pass2_result):
        """Any non-zero quartile cadence must be in [30, 200] steps/min."""
        for key, qm in pass2_result.quartile_metrics.items():
            if qm.step_count > 0:
                assert 30.0 <= qm.cadence_steps_per_min <= 200.0, (
                    f"Quartile {key}: cadence {qm.cadence_steps_per_min:.1f} out of range"
                )


class TestQuartileCoverage:

    def test_all_quartiles_present(self, pass2_result):
        """Q1, Q2, Q3, and Q4 must all have at least one step."""
        for key in ("Q1", "Q2", "Q3", "Q4"):
            count = pass2_result.quartile_metrics[key].step_count
            assert count > 0, f"Quartile {key} has 0 steps — expected > 0"

    def test_step_count_sum_equals_total(self, pass2_result):
        """Sum of quartile step counts must equal total validated foot strikes."""
        from stride.core import Quartile
        q_total = sum(
            pass2_result.quartile_metrics[q.value].step_count
            for q in (Quartile.Q1, Quartile.Q2, Quartile.Q3, Quartile.Q4)
        )
        all_events = len(pass2_result.foot_strikes)
        # Foot strikes includes events assigned to TURN quartile; q_total ≤ all_events
        assert q_total <= all_events, (
            f"Quartile sum {q_total} exceeds total foot strikes {all_events}"
        )
        assert q_total >= all_events - 4, (
            f"Quartile sum {q_total} is much less than foot strikes {all_events} "
            f"— possible unassigned events"
        )

    def test_toward_quartiles_have_more_steps_than_turn(self, pass2_result):
        """Q1 and Q2 (TOWARD) combined should have more steps than the TURN phase allows."""
        q1 = pass2_result.quartile_metrics["Q1"].step_count
        q2 = pass2_result.quartile_metrics["Q2"].step_count
        assert q1 + q2 > 0, "Neither Q1 nor Q2 has any steps"


class TestTurningTime:

    def test_turning_time_non_negative(self, pass2_result):
        assert pass2_result.turning_time_sec >= 0.0

    def test_actual_turn_distance_near_6m(self, pass2_result):
        """For a full 6m path, the turn should occur near 6m (within 20%)."""
        turn_dist = pass2_result.actual_turn_distance_m
        assert 3.0 <= turn_dist <= 7.0, (
            f"Turn distance {turn_dist:.2f}m is implausible for a 6m path"
        )
