"""Strider: Clinical gait analysis system for neurological movement disorders.

This is the main public API for the Strider library.
"""

from . import core, config, data, pipeline

from .core import (
    Side,
    Phase,
    Quartile,
    ClinicalFlagType,
    ClinicalSeverity,
    KeypointSchema,
    get_keypoint_schema,
)

from .config import StriderConfig, get_default_config

from .data import (
    FootStrikeEvent,
    FOGEpisode,
    QuartileMetrics,
    GaitMetrics,
    ClinicalFlag,
    ClinicalReport,
    AnalysisResult,
)

__version__ = "0.1.0"
__author__ = "Manas Maahir"

__all__ = [
    # Core types and schemas
    "Side",
    "Phase",
    "Quartile",
    "ClinicalFlagType",
    "ClinicalSeverity",
    "KeypointSchema",
    "get_keypoint_schema",
    # Config
    "StriderConfig",
    "get_default_config",
    # Data models
    "FootStrikeEvent",
    "FOGEpisode",
    "QuartileMetrics",
    "GaitMetrics",
    "ClinicalFlag",
    "ClinicalReport",
    "AnalysisResult",
    # Submodules
    "core",
    "config",
    "data",
    "pipeline",
]
