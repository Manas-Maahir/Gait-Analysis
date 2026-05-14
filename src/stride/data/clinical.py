"""Clinical interpretation and flags."""

from typing import Optional
from pydantic import BaseModel, Field

from ..core.types import ClinicalFlagType, ClinicalSeverity, Quartile


class ClinicalFlag(BaseModel):
    """A clinical flag indicating an abnormal gait characteristic."""

    flag_type: ClinicalFlagType = Field(..., description="Type of flag")
    severity: ClinicalSeverity = Field(..., description="Severity level (INFO, WARNING, CRITICAL)")
    value: float = Field(..., description="Actual metric value that triggered the flag")
    threshold: float = Field(..., description="Threshold that was exceeded")
    quartile: Optional[Quartile] = Field(None, description="Quartile where flag was detected (None for global)")
    description: str = Field("", description="Human-readable description of the flag")
    reference: str = Field("", description="Scientific reference or evidence for the threshold")


class ClinicalReport(BaseModel):
    """Clinical analysis and interpretation of gait metrics."""

    flags: list[ClinicalFlag] = Field(default_factory=list, description="List of detected clinical abnormalities")
    summary: str = Field("", description="Human-readable clinical summary")
    recommendations: list[str] = Field(default_factory=list, description="Clinical recommendations based on findings")
