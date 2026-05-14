"""Freezing of Gait (FOG) detection via spectral Freeze Index analysis.

Reference: Moore et al. 2008 — Ambulatory monitoring of freezing of gait in Parkinson's disease.
FI = P_freeze / P_loco (linear ratio, Moore 2008).

Window overlap aggregation uses np.maximum.at() so that each frame receives the highest
FI value from all spectral windows that cover it, avoiding the write-overwrite bias of
simple sliding-window assignments.
"""

import numpy as np
from scipy import signal as sp_signal

from ..core.keypoints import KeypointSchema
from ..data.events import FOGEpisode

_FREEZE_BAND_HZ = (3.0, 8.0)   # High-frequency tremor / freeze band
_LOCO_BAND_HZ = (0.5, 3.0)     # Normal locomotion band
_EPS = 1e-10                    # Guard against division by zero


class FOGDetector:
    """Detects Freezing of Gait episodes via overlapping spectral windows.

    Implements the Freeze Index (FI) from Moore et al. 2008:
        FI = P_freeze / P_loco
    where P_freeze is power in [3, 8] Hz and P_loco is power in [0.5, 3] Hz.

    A FOG episode is flagged when FI > fi_threshold for >= min_duration_sec.
    """

    def __init__(
        self,
        fps: float,
        fi_threshold: float = 2.5,
        window_sec: float = 2.0,
        min_duration_sec: float = 0.5,
    ) -> None:
        """
        Args:
            fps: Video frame rate (Hz)
            fi_threshold: FI value above which a frame is considered frozen
            window_sec: Spectral analysis window length (seconds)
            min_duration_sec: Minimum contiguous duration to report as FOG episode
        """
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self.fps = fps
        self.fi_threshold = fi_threshold
        self.window_sec = window_sec
        self.min_duration_sec = min_duration_sec

    def detect(
        self,
        keypoints: np.ndarray,
        timestamps: np.ndarray,
        schema: KeypointSchema,
    ) -> list[FOGEpisode]:
        """Detect FOG episodes from keypoint sequence.

        Args:
            keypoints: (N, n_keypoints, 3) array [x, y, confidence]
            timestamps: (N,) array of timestamps in seconds
            schema: KeypointSchema defining keypoint indices

        Returns:
            List of FOGEpisode objects sorted by start_frame
        """
        if len(keypoints) == 0:
            return []

        velocity = self._ankle_velocity_signal(keypoints, schema)
        fi_values = self.compute_freeze_index_signal(velocity)
        return self._detect_episodes(fi_values, timestamps)

    def compute_freeze_index_signal(self, ankle_velocity: np.ndarray) -> np.ndarray:
        """Compute per-frame Freeze Index using overlapping spectral windows.

        Each frame receives the maximum FI from all windows that cover it
        (np.maximum.at aggregation). This correctly handles the 50% overlap
        case where naive assignment would overwrite earlier window values.

        Args:
            ankle_velocity: (N,) vertical ankle velocity signal (any units)

        Returns:
            (N,) per-frame Freeze Index values (>= 0.0)
        """
        N = len(ankle_velocity)
        fi_values = np.zeros(N, dtype=np.float64)

        window_frames = max(4, int(self.window_sec * self.fps))
        stride_frames = max(1, window_frames // 2)   # 50% overlap

        for start in range(0, N - window_frames + 1, stride_frames):
            end = start + window_frames
            segment = ankle_velocity[start:end]

            # Welch PSD — nperseg = full window (already short)
            freqs, psd = sp_signal.welch(
                segment,
                fs=self.fps,
                nperseg=len(segment),
            )

            loco_mask = (freqs >= _LOCO_BAND_HZ[0]) & (freqs <= _LOCO_BAND_HZ[1])
            freeze_mask = (freqs >= _FREEZE_BAND_HZ[0]) & (freqs <= _FREEZE_BAND_HZ[1])

            P_loco = float(np.sum(psd[loco_mask]))
            P_freeze = float(np.sum(psd[freeze_mask]))

            fi = P_freeze / (P_loco + _EPS)

            # Max-aggregate: every frame in the window gets the higher of its
            # current value and this window's FI.
            np.maximum.at(fi_values, np.arange(start, end), fi)

        return fi_values

    def _ankle_velocity_signal(
        self,
        keypoints: np.ndarray,
        schema: KeypointSchema,
    ) -> np.ndarray:
        """Derive vertical ankle velocity signal from keypoints.

        Confidence-weighted average of left and right ankle y positions,
        with linear interpolation over frames where both ankles are occluded.
        """
        N = len(keypoints)

        left_y = keypoints[:, schema.left_ankle, 1]
        right_y = keypoints[:, schema.right_ankle, 1]
        left_conf = keypoints[:, schema.left_ankle, 2]
        right_conf = keypoints[:, schema.right_ankle, 2]

        total_conf = left_conf + right_conf
        safe_total = np.where(total_conf > 0, total_conf, 1.0)
        ankle_y = (left_y * left_conf + right_y * right_conf) / safe_total

        # Interpolate over frames where both ankles have zero confidence
        both_zero = total_conf == 0.0
        if both_zero.any() and not both_zero.all():
            indices = np.arange(N, dtype=np.float64)
            good = ~both_zero
            ankle_y = np.interp(indices, indices[good], ankle_y[good])

        # Central-difference velocity; np.gradient maintains signal length
        velocity: np.ndarray = np.gradient(ankle_y, 1.0 / self.fps)
        return velocity

    def _detect_episodes(
        self,
        fi_values: np.ndarray,
        timestamps: np.ndarray,
    ) -> list[FOGEpisode]:
        """Find contiguous runs where FI > threshold for >= min_duration_sec.

        Args:
            fi_values: (N,) per-frame Freeze Index
            timestamps: (N,) timestamps in seconds

        Returns:
            List of FOGEpisode, each carrying peak FI as severity
        """
        N = len(fi_values)
        if N == 0:
            return []

        min_frames = max(1, int(self.min_duration_sec * self.fps))
        above = fi_values > self.fi_threshold

        episodes: list[FOGEpisode] = []
        in_episode = False
        start_idx = 0

        for i in range(N + 1):
            currently_above = i < N and above[i]

            if currently_above and not in_episode:
                in_episode = True
                start_idx = i
            elif not currently_above and in_episode:
                in_episode = False
                duration_frames = i - start_idx
                if duration_frames >= min_frames:
                    end_idx = min(i - 1, N - 1)
                    t_start = float(timestamps[start_idx]) if start_idx < len(timestamps) else 0.0
                    t_end = float(timestamps[end_idx]) if end_idx < len(timestamps) else 0.0
                    peak_fi = float(np.max(fi_values[start_idx : i]))
                    episodes.append(FOGEpisode(
                        start_frame=start_idx,
                        end_frame=end_idx,
                        duration_sec=max(0.0, t_end - t_start),
                        severity=peak_fi,
                    ))

        return episodes
