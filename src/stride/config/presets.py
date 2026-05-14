"""Preset configurations for different gait analysis scenarios."""

from .schema import StriderConfig


def get_default_config() -> StriderConfig:
    """Get default configuration for normal gait analysis."""
    return StriderConfig()


def get_pathological_gait_config() -> StriderConfig:
    """Get configuration tuned for pathological gait (Parkinson's, shuffling, etc.)."""
    return StriderConfig(
        foot_strike_prominence_factor=0.15,  # More tolerant of low-amplitude steps
        foot_strike_min_interval_sec=0.1,    # Allow faster shuffling
        foot_strike_max_interval_sec=5.0,    # Allow longer pauses (FOG)
        normal_cadence_min=60.0,              # Lower normal cadence
        normal_cadence_max=120.0,
        abnormal_asymmetry_warning=15.0,     # Slightly more tolerant
    )


def get_gpu_config() -> StriderConfig:
    """Get configuration for GPU inference."""
    return StriderConfig(
        device="cuda",
        max_dim=1280,  # Can afford higher resolution on GPU
    )
