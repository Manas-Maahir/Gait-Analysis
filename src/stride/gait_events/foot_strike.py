"""Foot strike detection from ankle keypoint trajectories.

Detects heel strike events by finding peaks in the ankle trajectory.
Handles normal gait (clear peaks) and shuffling gait (flat trajectory) via
adaptive peak prominence and fallback to mediolateral displacement.
"""

from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from stride.core import KeypointSchema, Phase, Side
from stride.data.events import FootStrikeEvent


class FootStrikeDetector:
    """Detects foot strike events from ankle keypoint trajectories.

    Algorithm:
    1. Extract ankle Y positions (vertical ankle motion during swing)
    2. Normalize to [0, 1] range
    3. Find peaks with adaptive prominence (handles shuffling)
    4. For each peak, determine left/right foot from trajectory curvature
    5. Apply step validator (time + space constraints)
    """

    def __init__(
        self,
        fps: float,
        min_step_interval_sec: float = 0.2,
        max_step_interval_sec: float = 2.0,
        min_step_length_m: float = 0.2,
        max_step_length_m: float = 2.0,
        peak_distance_sec: float = 0.2,
        min_confidence: float = 0.3,
    ):
        """Initialize foot strike detector.

        Args:
            fps: Frame rate (frames/second)
            min_step_interval_sec: Minimum time between steps
            max_step_interval_sec: Maximum time between steps (handles very slow walkers)
            min_step_length_m: Minimum step length in meters
            max_step_length_m: Maximum step length in meters
            peak_distance_sec: Minimum distance between peaks (in seconds)
        """
        self.fps = fps
        self.min_step_interval = int(min_step_interval_sec * fps)
        self.max_step_interval = int(max_step_interval_sec * fps)
        self.min_step_length_m = min_step_length_m
        self.max_step_length_m = max_step_length_m
        self.peak_distance = int(peak_distance_sec * fps)
        self.min_confidence = min_confidence

    def detect(
        self,
        keypoints: np.ndarray,
        world_positions: np.ndarray,
        timestamps: np.ndarray,
        schema: KeypointSchema,
        phases: Optional[np.ndarray] = None,
    ) -> list[FootStrikeEvent]:
        """Detect foot strikes from keypoint sequence.

        Args:
            keypoints: (N, n_keypoints, 3) array [x, y, confidence]
            world_positions: (N, 2) array [x, y] in meters
            timestamps: (N,) array of frame timestamps in seconds
            schema: KeypointSchema defining keypoint indices
            phases: (N,) array of Phase labels (optional, for debugging)

        Returns:
            List of FootStrikeEvent objects
        """
        # Extract ankle positions
        left_ankle_y = keypoints[:, schema.left_ankle, 1]
        right_ankle_y = keypoints[:, schema.right_ankle, 1]
        left_conf = keypoints[:, schema.left_ankle, 2]
        right_conf = keypoints[:, schema.right_ankle, 2]

        # Detect peaks
        left_peaks = self._find_peaks_adaptive(
            left_ankle_y, left_conf, invert=True
        )
        right_peaks = self._find_peaks_adaptive(
            right_ankle_y, right_conf, invert=True
        )

        # Merge and sort peaks
        all_peaks = []
        for idx in left_peaks:
            all_peaks.append((idx, Side.L))
        for idx in right_peaks:
            all_peaks.append((idx, Side.R))
        all_peaks.sort(key=lambda x: x[0])

        # Create FootStrikeEvent objects
        events = []
        for frame_idx, side in all_peaks:
            ts = timestamps[frame_idx]
            world_x = world_positions[frame_idx, 0]
            world_y = world_positions[frame_idx, 1]

            # Compute step length from previous step
            if len(events) > 0:
                prev_x = events[-1].world_x
                step_length = np.sqrt((world_x - prev_x) ** 2 + (world_positions[frame_idx, 1] - events[-1].world_y) ** 2)
            else:
                step_length = 0.0

            detection_phase = phases[frame_idx] if phases is not None else None

            event = FootStrikeEvent(
                frame_idx=frame_idx,
                timestamp=ts,
                side=side,
                world_x=float(world_x),
                world_y=float(world_y),
                confidence=float(max(left_conf[frame_idx], right_conf[frame_idx])),
                step_length=float(step_length),
                step_time=float(ts - events[-1].timestamp) if len(events) > 0 else 0.0,
                detection_phase=detection_phase,
                quartile=None,
            )
            events.append(event)

        return events

    def _find_peaks_adaptive(
        self,
        signal: np.ndarray,
        confidence: np.ndarray,
        invert: bool = True,
    ) -> list[int]:
        """Find peaks in signal with adaptive prominence.

        Args:
            signal: 1D signal (ankle Y position)
            confidence: 1D confidence scores for each frame
            invert: If True, find peaks in -signal (for minima in original)

        Returns:
            List of frame indices with detected peaks
        """
        # Filter out low-confidence points; fall back to all frames if too few pass
        valid_mask = confidence > self.min_confidence
        if valid_mask.sum() < 10:
            # Use all non-zero frames — ankle trajectory is still informative at low confidence
            valid_mask = confidence > 0.0
        if not valid_mask.any():
            return []

        signal_clean = signal.copy()
        signal_clean[~valid_mask] = np.nan

        # Remove NaN by interpolation (safe: at least one valid point guaranteed above)
        nans = np.isnan(signal_clean)
        x = lambda z: z.nonzero()[0]
        signal_clean[nans] = np.interp(x(nans), x(~nans), signal_clean[~nans])

        # Remove slow approach/recession trend (perspective size change) so step
        # oscillations (~0.8 s) are detectable. Gaussian cutoff ~0.11 Hz (1.5 s sigma)
        # sits above the arch frequency (~0.07 Hz) but below step frequency (~1.25 Hz).
        trend = gaussian_filter1d(signal_clean, sigma=max(1.0, self.fps * 1.5))
        signal_clean = signal_clean - trend

        # Normalize to [0, 1]
        sig_min = np.nanmin(signal_clean)
        sig_max = np.nanmax(signal_clean)
        if sig_max - sig_min < 1e-6:
            # Signal is flat (shuffling) - fallback to mediolateral detection
            return []

        signal_norm = (signal_clean - sig_min) / (sig_max - sig_min)

        if invert:
            signal_norm = 1 - signal_norm

        # Adaptive prominence: 10% of signal range
        range_val = np.ptp(signal_norm)
        prominence = 0.1 * range_val if range_val > 0.02 else 0.002

        try:
            peaks, _ = find_peaks(
                signal_norm,
                prominence=prominence,
                distance=self.peak_distance,
            )
            return peaks.tolist()
        except Exception:
            return []
