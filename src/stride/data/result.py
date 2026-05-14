"""Analysis result and metadata models."""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict

from .metrics import GaitMetrics
from .clinical import ClinicalReport


class ProcessingMetadata(BaseModel):
    """Metadata about the processing run."""

    video_path: str = Field(..., description="Path to input video")
    fps: float = Field(..., description="Frames per second of input video")
    total_frames: int = Field(..., description="Total number of frames processed")
    processing_time_sec: float = Field(..., description="Wall-clock time to process video (seconds)")
    timestamp: datetime = Field(default_factory=datetime.now, description="When processing was completed")
    config_hash: str = Field("", description="Hash of configuration used (for reproducibility)")
    errors: list[str] = Field(default_factory=list, description="Any warnings or errors encountered")


class AnalysisResult(BaseModel):
    """Complete analysis result for a single gait trial.

    Supports full JSON round-trip serialization via Pydantic's model_validate_json().
    """

    model_config = ConfigDict(
        json_schema_extra={"schema_version": "1.0"},
        json_encoders={Path: str},
    )

    schema_version: str = Field("1.0", description="Version of this result schema")
    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique experiment identifier")
    metadata: ProcessingMetadata = Field(..., description="Processing metadata")
    metrics: GaitMetrics = Field(..., description="All computed gait metrics")
    annotated_video_path: Optional[str] = Field(None, description="Path to annotated video output (if saved)")

    def to_json(self, path: str | Path) -> None:
        """Save results to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "AnalysisResult":
        """Load results from JSON file.

        Args:
            path: Input file path

        Returns:
            AnalysisResult reconstructed from JSON

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If JSON is invalid or schema version mismatch
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")

        json_str = path.read_text()
        return cls.model_validate_json(json_str)
