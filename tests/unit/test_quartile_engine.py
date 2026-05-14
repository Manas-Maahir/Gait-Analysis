"""
Unit tests for the QuartileEngine.

Tests the core distance-based quartile assignment logic.
This is a CRITICAL module, so tests are comprehensive.
"""

import numpy as np
import pytest

from stride.segmentation.quartile_engine import QuartileEngine
from stride.core.types import Phase, Quartile


@pytest.fixture
def engine():
    """Provide a default QuartileEngine."""
    return QuartileEngine(path_length_m=6.0)


class TestQuartileAssignment:
    """Test distance-based quartile assignment."""

    def test_q1_toward(self, engine):
        """Test Q1 assignment in TOWARD phase."""
        assert engine.assign_quartile(0.0, Phase.TOWARD) == "Q1"
        assert engine.assign_quartile(1.5, Phase.TOWARD) == "Q1"
        assert engine.assign_quartile(2.99, Phase.TOWARD) == "Q1"

    def test_q2_toward(self, engine):
        """Test Q2 assignment in TOWARD phase."""
        assert engine.assign_quartile(3.0, Phase.TOWARD) == "Q2"
        assert engine.assign_quartile(4.5, Phase.TOWARD) == "Q2"
        assert engine.assign_quartile(6.0, Phase.TOWARD) == "Q2"

    def test_q3_away(self, engine):
        """Test Q3 assignment in AWAY phase (first 3m away from turn)."""
        # Away phase: world_x decreases from 6
        # Q3 when distance_away < 3m
        assert engine.assign_quartile(6.0, Phase.AWAY) == "Q3"  # away_dist = 0
        assert engine.assign_quartile(5.0, Phase.AWAY) == "Q3"  # away_dist = 1
        assert engine.assign_quartile(3.01, Phase.AWAY) == "Q3"  # away_dist = 2.99

    def test_q4_away(self, engine):
        """Test Q4 assignment in AWAY phase (final 3m away from turn)."""
        # Q4 when distance_away >= 3m
        assert engine.assign_quartile(3.0, Phase.AWAY) == "Q4"  # away_dist = 3.0
        assert engine.assign_quartile(1.5, Phase.AWAY) == "Q4"  # away_dist = 4.5
        assert engine.assign_quartile(0.0, Phase.AWAY) == "Q4"  # away_dist = 6.0

    def test_turn_phase(self, engine):
        """Test TURN phase assignment."""
        assert engine.assign_quartile(5.99, Phase.TURN) == "TURN"
        assert engine.assign_quartile(6.01, Phase.TURN) == "TURN"

    def test_boundary_exactly_at_3m_toward(self, engine):
        """Test boundary condition: exactly at 3m in TOWARD phase."""
        # 3m is the boundary between Q1 and Q2
        # By convention, Q2 starts at 3m (Q1 < 3, Q2 >= 3)
        assert engine.assign_quartile(3.0, Phase.TOWARD) == "Q2"

    def test_boundary_exactly_at_6m_toward(self, engine):
        """Test boundary condition: exactly at 6m in TOWARD phase."""
        assert engine.assign_quartile(6.0, Phase.TOWARD) == "Q2"

    def test_boundary_exactly_at_3m_away(self, engine):
        """Test boundary condition: exactly at 3m in AWAY phase."""
        # away_dist = 6.0 - 3.0 = 3.0
        # Boundary between Q3 and Q4 (Q3 < 3, Q4 >= 3)
        assert engine.assign_quartile(3.0, Phase.AWAY) == "Q4"

    def test_invalid_phase_raises_error(self, engine):
        """Test that invalid phase raises ValueError."""
        with pytest.raises(ValueError):
            engine.assign_quartile(3.0, "INVALID_PHASE")


class TestQuartileTimeWindows:
    """Test computing time ranges for each quartile."""

    def test_simple_sequence(self, engine):
        """Test time window computation on a simple synthetic sequence."""
        # Simulate a patient walking:
        # Frames 0-5: Q1 (world_x goes 0.0 → 2.5)
        # Frames 6-10: Q2 (world_x goes 3.0 → 6.0)
        # Frames 11-15: TURN (world_x = 6.0)
        # Frames 16-20: Q3 (world_x goes 5.5 → 3.5)
        # Frames 21-25: Q4 (world_x goes 3.0 → 0.0)

        world_x = np.array([
            0.0, 0.5, 1.0, 1.5, 2.0, 2.5,  # Q1
            3.0, 3.5, 4.5, 5.5, 6.0,  # Q2
            6.0, 6.0,  # TURN
            5.5, 5.0, 4.5, 4.0, 3.5,  # Q3
            3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0,  # Q4
        ], dtype=np.float32)

        phases = np.array([
            Phase.TOWARD, Phase.TOWARD, Phase.TOWARD, Phase.TOWARD, Phase.TOWARD, Phase.TOWARD,
            Phase.TOWARD, Phase.TOWARD, Phase.TOWARD, Phase.TOWARD, Phase.TOWARD,
            Phase.TURN, Phase.TURN,
            Phase.AWAY, Phase.AWAY, Phase.AWAY, Phase.AWAY, Phase.AWAY,
            Phase.AWAY, Phase.AWAY, Phase.AWAY, Phase.AWAY, Phase.AWAY, Phase.AWAY, Phase.AWAY,
        ])

        timestamps = np.arange(len(world_x), dtype=np.float32)

        windows = engine.compute_quartile_time_windows(world_x, phases, timestamps)

        # Check that all quartiles were visited
        assert windows["Q1"] is not None
        assert windows["Q2"] is not None
        assert windows["Q3"] is not None
        assert windows["Q4"] is not None
        assert windows["TURN"] is not None

        # Check time ranges are reasonable
        assert windows["Q1"][0] < windows["Q1"][1]
        assert windows["Q2"][0] < windows["Q2"][1]

    def test_missing_quartile(self, engine):
        """Test that missing quartiles return None."""
        # Only Q1 and Q2 (never turn around or go away)
        world_x = np.array([0.0, 1.5, 3.0, 4.5, 6.0], dtype=np.float32)
        phases = np.array([Phase.TOWARD] * 5)
        timestamps = np.arange(5, dtype=np.float32)

        windows = engine.compute_quartile_time_windows(world_x, phases, timestamps)

        assert windows["Q1"] is not None
        assert windows["Q2"] is not None
        assert windows["Q3"] is None
        assert windows["Q4"] is None
        assert windows["TURN"] is None


class TestQuartileBoundaries:
    """Test boundary definitions."""

    def test_get_quartile_boundaries(self, engine):
        """Test that boundaries are correctly reported."""
        boundaries = engine.get_quartile_boundaries()

        # Toward phase boundaries (in meters along path)
        assert boundaries["Q1"] == (0.0, 3.0)
        assert boundaries["Q2"] == (3.0, 6.0)

        # Away phase boundaries (in away-distance)
        assert boundaries["Q3"] == (0.0, 3.0)
        assert boundaries["Q4"] == (3.0, 6.0)


class TestInvariantValidation:
    """Test the step count invariant validation."""

    def test_valid_step_counts(self, engine):
        """Test that valid step counts pass validation."""
        counts = {
            "Q1": 10,
            "Q2": 12,
            "Q3": 11,
            "Q4": 13,
            "total_steps": 46,
        }
        assert engine.validate_step_assignments(counts) is True

    def test_invalid_step_counts(self, engine):
        """Test that invalid step counts fail validation."""
        counts = {
            "Q1": 10,
            "Q2": 12,
            "Q3": 11,
            "Q4": 13,
            "total_steps": 47,  # Should be 46
        }
        assert engine.validate_step_assignments(counts) is False

    def test_missing_quartile_in_counts(self, engine):
        """Test validation when a quartile is missing."""
        counts = {
            "Q1": 10,
            "Q2": 12,
            # Q3 and Q4 not visited
            "total_steps": 22,
        }
        assert engine.validate_step_assignments(counts) is True

    def test_all_zero_steps(self, engine):
        """Test validation when no steps detected."""
        counts = {
            "Q1": 0,
            "Q2": 0,
            "Q3": 0,
            "Q4": 0,
            "total_steps": 0,
        }
        assert engine.validate_step_assignments(counts) is True


class TestNonStandardPath:
    """Test handling of non-standard walking paths (turn before/after 6m)."""

    def test_early_turn_at_5m(self, engine):
        """Test that early turn is correctly handled."""
        # Patient turns at 5m instead of 6m
        # The assignment should still work correctly based on phase

        # At 5m in TOWARD phase = Q2 (since 5 >= 3)
        assert engine.assign_quartile(5.0, Phase.TOWARD) == "Q2"

        # When turning and walking away from 5m:
        # away_dist = 5.0 - 4.5 = 0.5 → Q3
        assert engine.assign_quartile(4.5, Phase.AWAY) == "Q3"

    def test_late_turn_at_7m(self, engine):
        """Test handling of late turn (beyond normal 6m)."""
        # This could happen if video extends beyond 6m
        assert engine.assign_quartile(7.0, Phase.TOWARD) == "Q2"


class TestIntegrationWithSyntheticData:
    """Test quartile engine with realistic synthetic gait data."""

    def test_full_trial_sequence(self, engine):
        """Test a complete trial: walk toward, turn, walk away."""
        # Simulate a full 6m walk test with 30 steps

        # Phase 1: Walk toward (0-6m), ~15 steps
        toward_x = np.linspace(0.0, 6.0, 15)
        toward_phase = np.array([Phase.TOWARD] * 15)

        # Phase 2: Turn (around 6m)
        turn_x = np.array([6.0, 6.0])
        turn_phase = np.array([Phase.TURN] * 2)

        # Phase 3: Walk away (6-0m), ~15 steps
        away_x = np.linspace(6.0, 0.0, 15)
        away_phase = np.array([Phase.AWAY] * 15)

        # Combine all
        world_x = np.concatenate([toward_x, turn_x, away_x]).astype(np.float32)
        phases = np.concatenate([toward_phase, turn_phase, away_phase])
        timestamps = np.arange(len(world_x), dtype=np.float32)

        # Assign each frame to a quartile
        assignments = np.array(
            [engine.assign_quartile(world_x[i], phases[i]) for i in range(len(world_x))]
        )

        # Count steps in each quartile (for demo, count frames)
        q1_count = np.sum(assignments == "Q1")
        q2_count = np.sum(assignments == "Q2")
        q3_count = np.sum(assignments == "Q3")
        q4_count = np.sum(assignments == "Q4")
        turn_count = np.sum(assignments == "TURN")

        # Validate invariant
        assert engine.validate_step_assignments({
            "Q1": q1_count,
            "Q2": q2_count,
            "Q3": q3_count,
            "Q4": q4_count,
            "total_steps": q1_count + q2_count + q3_count + q4_count,
        }) is True

        # Each quartile should have been visited
        assert q1_count > 0
        assert q2_count > 0
        assert q3_count > 0
        assert q4_count > 0
        assert turn_count > 0
