"""Gait event data models (foot strikes, FOG episodes, etc.)."""

from typing import Optional
from pydantic import BaseModel, Field

from ..core.types import Phase, Quartile


class FootStrikeEvent(BaseModel):
    """A detected foot strike (heel-down) event."""

    frame_idx: int = Field(..., description="Frame index in video where event occurred")
    timestamp: float = Field(..., description="Time in seconds from video start")
    side: str = Field(..., description="Left ('L') or Right ('R') foot", pattern="^[LR]$")
    world_x: float = Field(..., description="Position along walking axis (meters)")
    world_y: float = Field(..., description="Lateral position (meters)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Keypoint confidence")
    step_length: Optional[float] = Field(None, description="Distance from previous opposite-foot strike (meters)")
    step_time: Optional[float] = Field(None, description="Time since previous opposite-foot strike (seconds)")
    detection_phase: Optional[Phase] = Field(None, description="Walking phase at detection time (TOWARD/TURN/AWAY)")
    quartile: Optional[Quartile] = Field(None, description="Which quartile this event belongs to (assigned post-detection)")


class FOGEpisode(BaseModel):
    """A Freezing of Gait episode detected in the signal."""

    start_frame: int = Field(..., description="Frame index where FOG begins")
    end_frame: int = Field(..., description="Frame index where FOG ends")
    duration_sec: float = Field(..., description="Episode duration (seconds)")
    severity: float = Field(..., description="Freeze Index value (higher = more severe)")
