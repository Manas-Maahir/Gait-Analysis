"""Homography-based spatial calibration: mapping image pixels to world meters.

Two calibration modes:
1. Manual: Clinician clicks 4 floor markers in video frame, assigns world coordinates
2. Auto: SVD fit of ankle trajectory to find walking axis, scale by 6m constraint
"""

from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np

from stride.core import CalibrationResult


@dataclass
class CalibrationData:
    """Raw calibration data (image coordinates)."""

    image_pts: np.ndarray  # (4, 2) pixel coordinates
    world_pts: np.ndarray  # (4, 2) world coordinates (meters)


class ManualHomographyCalibrator:
    """Manual 4-point homography calibration via interactive frame markup.

    User clicks 4 floor points in the frame, assigns world coordinates (e.g., path markers).
    Computes homography H such that: world_pt = H @ image_pt
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.image_pts: list[tuple[int, int]] = []
        self.world_pts: list[tuple[float, float]] = []
        self.frame = None

    def calibrate_interactive(
        self,
        frame: np.ndarray,
        window_name: str = "Calibration: Click 4 floor points",
    ) -> CalibrationResult:
        """Interactive 4-point calibration on a video frame.

        Args:
            frame: BGR video frame
            window_name: Window title for mouse callback

        Returns:
            CalibrationResult with homography matrix
        """
        self.image_pts = []
        self.world_pts = []
        self.frame = frame.copy()

        # Create window with mouse callback
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_click)

        if self.verbose:
            print("Click 4 floor points in order: Q1-corner, Q2-corner, Q3-corner, Q4-corner")

        # Wait for 4 clicks
        while len(self.image_pts) < 4:
            cv2.imshow(window_name, self._draw_points())
            key = cv2.waitKey(100)
            if key == 27:  # ESC
                raise KeyboardInterrupt("Calibration cancelled")

        cv2.destroyWindow(window_name)

        # Assign world coordinates: standard 6m path with 0.5m width
        self.world_pts = [
            (0.0, 0.0),    # Bottom-left (0m, Q1 start)
            (6.0, 0.0),    # Bottom-right (6m, Q2 end)
            (0.0, 0.5),    # Top-left
            (6.0, 0.5),    # Top-right
        ]

        return self._compute_homography()

    def _mouse_click(self, event, x, y, flags, param):
        """Mouse callback for manual point selection."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.image_pts) < 4:
            self.image_pts.append((x, y))
            if self.verbose:
                print(f"Point {len(self.image_pts)}: ({x}, {y})")

    def _draw_points(self) -> np.ndarray:
        """Draw selected points on frame."""
        out = self.frame.copy()
        for i, (x, y) in enumerate(self.image_pts):
            cv2.circle(out, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(
                out,
                str(i + 1),
                (int(x) + 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
            )
        return out

    def _compute_homography(self) -> CalibrationResult:
        """Compute homography from point pairs."""
        img_pts = np.array(self.image_pts, dtype=np.float32)
        world_pts = np.array(self.world_pts, dtype=np.float32)

        H, _ = cv2.findHomography(img_pts, world_pts, cv2.RANSAC, 4.0)

        if H is None:
            raise RuntimeError("Failed to compute homography")

        # Approximate scale from pixel distance between the two 6m endpoints
        px_dist = float(np.linalg.norm(img_pts[1] - img_pts[0]))
        scale_px_to_m = 6.0 / px_dist if px_dist > 1e-6 else 1.0 / 100.0

        return CalibrationResult(
            homography_matrix=H,
            scale_px_to_m=scale_px_to_m,
            pc1_variance=1.0,
            method="manual",
        )


class SVDAutoCalibrator:
    """Automatic homography calibration via SVD on ankle trajectory.

    Fits ankle keypoint trajectory to find walking axis direction, then scales
    by the 6m path length constraint to convert pixel distances to meters.

    Requires at least 10 frames of walking with good ankle visibility.
    """

    def __init__(self, path_length_m: float = 6.0, min_points: int = 10):
        self.path_length_m = path_length_m
        self.min_points = min_points

    def calibrate(self, ankle_trajectory: np.ndarray) -> CalibrationResult:
        """Auto-calibrate from ankle trajectory.

        Args:
            ankle_trajectory: (N, 2) array of ankle pixel coordinates

        Returns:
            CalibrationResult with fitted homography and PC1 variance
        """
        # Remove low-confidence points
        valid_mask = ~np.isnan(ankle_trajectory).any(axis=1)
        valid_pts = ankle_trajectory[valid_mask]

        if len(valid_pts) < self.min_points:
            raise ValueError(
                f"Need at least {self.min_points} valid ankle points, got {len(valid_pts)}"
            )

        # Fit line using SVD
        mean = np.mean(valid_pts, axis=0)
        centered = valid_pts - mean

        # SVD: centered = U @ S @ V^T
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        direction = Vt[0]  # Principal component (walking direction)

        # For the 6m walk-toward-turn-walk-away protocol the turn (nearest point to camera)
        # falls near the temporal midpoint of the recording. The correct direction has its
        # peak projection at the midpoint, not at the ends. Compare the central 20 % of
        # frames against the first 20 % and flip if the midpoint is lower — this is more
        # reliable than the quarter-based check for a symmetric walk whose first and last
        # quarters both sit at the far end of the path (equal projection ≈ no clear signal).
        proj_check = centered @ direction
        q = max(2, len(proj_check) // 5)
        mid = len(proj_check) // 2
        proj_mid = float(np.mean(proj_check[max(0, mid - q // 2) : mid + q // 2]))
        proj_start = float(np.mean(proj_check[:q]))
        if proj_mid < proj_start:
            direction = -direction

        # Compute PC1 variance
        pc1_variance = (S[0] ** 2) / np.sum(S**2)

        if pc1_variance < 0.85:
            print(
                f"Warning: PC1 variance {pc1_variance:.2f} < 0.85. "
                f"Auto-calibration may be inaccurate."
            )

        # Project all points onto walking axis
        proj = centered @ direction
        proj_range = np.max(proj) - np.min(proj)

        # Scale: pixel_distance / proj_range = meters / 6.0
        # Therefore: pixels_to_meters = 6.0 / proj_range
        if proj_range < 1.0:
            raise ValueError(
                f"Ankle trajectory spans only {proj_range:.1f} pixels. "
                f"Not enough motion for auto-calibration."
            )

        pixels_to_meters = self.path_length_m / proj_range

        # Construct transformation.
        # Baseline (unanchored): world_x = scale * dot(pixel - mean, direction)
        # This centres world_x at 0, giving ≈ [−3, +3] for a 6 m path.
        #
        # Anchoring: subtract proj_min so that world_x = 0 at the trajectory start
        # (person standing at the far end of the path) and world_x = 6 m at the turn.
        # Algebraically: world_x = scale * (dot(pixel - mean, direction) - proj_min)
        # which folds proj_min into the translation term of H[0,2].
        proj_min = float(np.min(proj))
        scale = pixels_to_meters
        H = np.eye(3, dtype=np.float32)
        H[0, 0] = direction[0] * scale
        H[0, 1] = direction[1] * scale
        H[0, 2] = -(mean[0] * direction[0] + mean[1] * direction[1]) * scale - proj_min * scale
        H[1, 0] = -direction[1] * scale
        H[1, 1] = direction[0] * scale
        H[1, 2] = -(mean[0] * (-direction[1]) + mean[1] * direction[0]) * scale

        return CalibrationResult(
            homography_matrix=H,
            scale_px_to_m=float(pixels_to_meters),
            pc1_variance=float(pc1_variance),
            method="svd_auto",
        )
