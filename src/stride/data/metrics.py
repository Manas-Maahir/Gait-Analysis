"""Gait metrics data models (per-quartile, global, and composite)."""

from pydantic import BaseModel, Field

from ..core.types import Quartile


class QuartileMetrics(BaseModel):
    """Metrics computed for a single quartile (Q1, Q2, Q3, or Q4)."""

    quartile: Quartile = Field(..., description="Which quartile (Q1/Q2/Q3/Q4/TURN)")
    step_count: int = Field(..., description="Number of steps detected in this quartile")
    cadence_steps_per_min: float = Field(..., description="Steps per minute")
    asymmetry_score: float = Field(0.0, description="Robinson Asymmetry Index (0-100)")
    sway_rms_meters: float = Field(0.0, description="RMS mediolateral sway (meters)")
    duration_seconds: float = Field(..., description="Time spent in this quartile (seconds)")
    mean_step_length_m: float = Field(0.0, description="Average step length (meters)")
    step_length_cv_percent: float = Field(0.0, description="Coefficient of variation for step length (%)")
    mean_step_time_sec: float = Field(0.0, description="Average step interval (seconds)")
    step_time_cv_percent: float = Field(0.0, description="Coefficient of variation for step time (%)")
    left_step_count: int = Field(0, description="Number of left foot steps")
    right_step_count: int = Field(0, description="Number of right foot steps")
    fog_episodes_count: int = Field(0, description="Number of freezing episodes in this quartile")
    fog_total_duration_sec: float = Field(0.0, description="Total duration of FOG in this quartile")

    # Confidence scores (0-1, where 1 = very confident)
    step_count_confidence: float = Field(1.0, ge=0.0, le=1.0)
    cadence_confidence: float = Field(1.0, ge=0.0, le=1.0)
    asymmetry_confidence: float = Field(1.0, ge=0.0, le=1.0)
    sway_confidence: float = Field(1.0, ge=0.0, le=1.0)


class GlobalMetrics(BaseModel):
    """Trial-level metrics aggregated across all quartiles."""

    total_steps: int = Field(..., description="Total steps in entire trial")
    overall_cadence_steps_per_min: float = Field(..., description="Average cadence across entire trial")
    overall_asymmetry_score: float = Field(0.0, description="Mean asymmetry across quartiles")
    overall_sway_rms_meters: float = Field(0.0, description="Mean sway across quartiles")
    toward_6m_time_sec: float = Field(..., description="Time to walk toward 6 meters")
    away_6m_time_sec: float = Field(..., description="Time to walk away 6 meters")
    turning_time_sec: float = Field(..., description="Time to complete the turn")
    total_trial_time_sec: float = Field(..., description="Total trial time (toward + turn + away)")
    stride_length_cv_percent: float = Field(0.0, description="Stride length variability")
    step_time_cv_percent: float = Field(0.0, description="Step time variability")


class GaitMetrics(BaseModel):
    """Complete set of gait metrics for a trial.

    Uses dict[Quartile, QuartileMetrics] instead of hardcoded q1/q2/q3/q4 fields
    to support variable path lengths and custom segmentation in the future.
    """

    quartile_metrics: dict[Quartile, QuartileMetrics] = Field(
        ...,
        description="Per-quartile metrics indexed by Quartile enum"
    )
    global_metrics: GlobalMetrics = Field(..., description="Trial-level aggregated metrics")
    clinical_report: "ClinicalReport" = Field(None, description="Clinical interpretation and flags")


# Forward reference resolution (circular dependency with clinical.py)
from .clinical import ClinicalReport  # noqa: E402, F401
GaitMetrics.model_rebuild()
