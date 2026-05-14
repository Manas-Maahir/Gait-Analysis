"""Consolidated domain types and enums for Strider gait analysis system."""

from enum import StrEnum


class Side(StrEnum):
    """Body side (left/right) for bilateral measurements."""

    L = "L"
    R = "R"


class Phase(StrEnum):
    """Walking phase classification based on direction along calibrated path."""

    TOWARD = "TOWARD"  # Walking toward the 6m endpoint
    TURN = "TURN"      # At turning point (reversing direction)
    AWAY = "AWAY"      # Walking away from the 6m endpoint


class Quartile(StrEnum):
    """Distance-based quartile assignment (world-space, not time-based)."""

    Q1 = "Q1"   # TOWARD phase: 0.0–3.0m
    Q2 = "Q2"   # TOWARD phase: 3.0–6.0m
    Q3 = "Q3"   # AWAY phase: distance 3.0m to turning point (backward)
    Q4 = "Q4"   # AWAY phase: 0.0m to 3.0m (backward, returning to start)
    TURN = "TURN"  # At the turning point (neutral/transition)


class ClinicalFlagType(StrEnum):
    """Clinically meaningful gait abnormality types."""

    FREEZING_OF_GAIT = "freezing_of_gait"
    ABNORMAL_ASYMMETRY = "abnormal_asymmetry"
    REDUCED_CADENCE = "reduced_cadence"
    EXCESSIVE_SWAY = "excessive_sway"
    HIGH_VARIABILITY = "high_variability"
    TURN_PROLONGATION = "turn_prolongation"
    ABNORMAL_STEP_LENGTH = "abnormal_step_length"
    GAIT_INSTABILITY = "gait_instability"


class ClinicalSeverity(StrEnum):
    """Severity level for clinical flags."""

    INFO = "INFO"           # Informational, within normal range
    WARNING = "WARNING"     # Mild-to-moderate abnormality
    CRITICAL = "CRITICAL"  # Severe abnormality, clinical concern
