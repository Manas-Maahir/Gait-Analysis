"""Strider configuration framework."""

from .schema import StriderConfig
from .presets import (
    get_default_config,
    get_pathological_gait_config,
    get_gpu_config,
)

__all__ = [
    "StriderConfig",
    "get_default_config",
    "get_pathological_gait_config",
    "get_gpu_config",
]
