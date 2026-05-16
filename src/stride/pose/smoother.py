"""OneEuro temporal filter for smoothing noisy pose keypoints.

Based on the OneEuro filter (Casiez et al. 2012), which uses exponential moving averages
and adds adaptive low-pass filtering to smooth while preserving responsiveness.
"""

import numpy as np


class OneEuroFilter:
    """OneEuro temporal filter for keypoint smoothing.

    Smooths pose keypoint sequences while preserving sharp movements. Parameters:
    - fcmin: Minimum cutoff frequency (Hz) — baseline smoothing
    - beta: Speed coefficient — increases cutoff frequency during fast motion
    - dcutoff: Derivative cutoff frequency (Hz) — smooths the velocity estimate
    """

    def __init__(
        self,
        fps: float,
        fcmin: float = 1.0,
        beta: float = 0.5,
        dcutoff: float = 1.0,
    ):
        """Initialize OneEuro filter.

        Args:
            fps: Frame rate (frames per second)
            fcmin: Minimum cutoff frequency (Hz), default 1.0
            beta: Speed coefficient (0-1), default 0.5
            dcutoff: Derivative cutoff frequency (Hz), default 1.0
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.fcmin = fcmin
        self.beta = beta
        self.dcutoff = dcutoff

        self.xfilt = None  # Filtered position
        self.dfilt = None  # Filtered velocity (derivative)

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Apply OneEuro filter to a single value or array.

        Args:
            x: Current measurement (scalar or ndarray of shape (...,))

        Returns:
            Filtered value (same shape as x)
        """
        if self.xfilt is None:
            # First frame: initialize with measurement
            self.xfilt = np.array(x, dtype=np.float32)
            self.dfilt = np.zeros_like(x, dtype=np.float32)
            return np.array(x, dtype=np.float32)

        x = np.asarray(x, dtype=np.float32)

        # Estimate velocity (derivative)
        d = (x - self.xfilt) / self.dt

        # Smooth velocity estimate with adaptive cutoff
        d_cutoff = 2 * np.pi * self.dcutoff
        a = self._alpha(d_cutoff)
        self.dfilt = a * d + (1 - a) * self.dfilt

        # Compute adaptive cutoff frequency based on velocity magnitude
        speed = np.linalg.norm(self.dfilt)
        fc = self.fcmin + self.beta * speed

        # Smooth position with adaptive cutoff
        fc_cutoff = 2 * np.pi * fc
        a_pos = self._alpha(fc_cutoff)
        self.xfilt = a_pos * x + (1 - a_pos) * self.xfilt

        return self.xfilt.copy()

    def _alpha(self, cutoff: float) -> float:
        """Compute exponential moving average coefficient.

        Args:
            cutoff: Cutoff frequency in rad/s

        Returns:
            Alpha coefficient for EMA update
        """
        # Standard 1st-order Euler approximation: alpha = tau_c*dt / (tau_c*dt + 1)
        # where tau_c = cutoff (in rad/s) * dt gives the normalised cutoff.
        # Large fc (fast motion) → alpha → 1 (responsive, follows signal).
        # Small fc (slow/still)  → alpha → 0 (smooth, holds previous value).
        t = cutoff * self.dt
        return t / (1.0 + t)

    def reset(self) -> None:
        """Reset filter state (for processing new sequences)."""
        self.xfilt = None
        self.dfilt = None


def smooth_keypoints(
    keypoints_seq: np.ndarray,
    fps: float,
    fcmin: float = 1.0,
    beta: float = 0.5,
) -> np.ndarray:
    """Smooth a sequence of keypoint frames using OneEuro filter.

    Args:
        keypoints_seq: Sequence of keypoints (N_frames, n_keypoints, 3) where
                      each keypoint is [x, y, confidence]
        fps: Frame rate (frames per second)
        fcmin: Minimum cutoff frequency (Hz)
        beta: Speed coefficient

    Returns:
        Smoothed keypoints (same shape as input)
    """
    n_frames, n_keypoints, _ = keypoints_seq.shape
    smoothed = np.zeros_like(keypoints_seq, dtype=np.float32)

    # Apply filter per keypoint
    for kpt_idx in range(n_keypoints):
        filt = OneEuroFilter(fps=fps, fcmin=fcmin, beta=beta)

        for frame_idx in range(n_frames):
            xy = keypoints_seq[frame_idx, kpt_idx, :2]
            conf = keypoints_seq[frame_idx, kpt_idx, 2]

            # Only smooth if confidence > 0
            if conf > 0:
                xy_smooth = filt.filter(xy)
                smoothed[frame_idx, kpt_idx, :2] = xy_smooth
                smoothed[frame_idx, kpt_idx, 2] = conf
            else:
                # Low confidence: pass through unchanged
                smoothed[frame_idx, kpt_idx, :] = keypoints_seq[frame_idx, kpt_idx, :]

    return smoothed
