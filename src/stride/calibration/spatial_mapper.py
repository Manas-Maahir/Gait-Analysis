"""Vectorized image-to-world coordinate transformation using homography."""

import numpy as np

from stride.core import CalibrationResult


class SpatialMapper:
    """Transforms image pixel coordinates to world-space meters using homography.

    Once calibrated, this mapper converts:
    - Video frame keypoint positions (image pixels)
    - To world space positions (meters along 6m path)
    """

    def __init__(self, calibration: CalibrationResult):
        """Initialize mapper with calibration result.

        Args:
            calibration: CalibrationResult containing homography matrix
        """
        self.calibration = calibration
        self.H = calibration.homography_matrix  # (3, 3) matrix

    def image_to_world(self, image_pts: np.ndarray) -> np.ndarray:
        """Transform image coordinates to world coordinates.

        Args:
            image_pts: (..., 2) array of image coordinates [x, y]
                      Can be 1D, 2D, or any shape

        Returns:
            (..., 2) array of world coordinates [x_world, y_world] in meters
        """
        input_shape = image_pts.shape
        is_2d = len(input_shape) == 1

        if is_2d:
            # Single point
            img_pts = image_pts.reshape(1, 2)
        else:
            # Multiple points; flatten all but last dimension
            img_pts = image_pts.reshape(-1, 2)

        # Homogeneous coordinates
        ones = np.ones((img_pts.shape[0], 1), dtype=np.float32)
        img_pts_h = np.hstack([img_pts, ones])  # (N, 3)

        # Apply homography: world_pts = H @ img_pts^T
        world_pts_h = (self.H @ img_pts_h.T).T  # (N, 3)

        # Normalize by homogeneous coordinate
        world_pts = world_pts_h[:, :2] / (world_pts_h[:, 2:3] + 1e-8)

        if is_2d:
            return world_pts.reshape(2)
        else:
            return world_pts.reshape(*input_shape)

    def world_to_image(self, world_pts: np.ndarray) -> np.ndarray:
        """Transform world coordinates back to image coordinates.

        Args:
            world_pts: (..., 2) array of world coordinates [x_world, y_world]

        Returns:
            (..., 2) array of image coordinates [x, y]
        """
        input_shape = world_pts.shape
        is_2d = len(input_shape) == 1

        if is_2d:
            world_pts_arr = world_pts.reshape(1, 2)
        else:
            world_pts_arr = world_pts.reshape(-1, 2)

        # Homogeneous coordinates
        ones = np.ones((world_pts_arr.shape[0], 1), dtype=np.float32)
        world_pts_h = np.hstack([world_pts_arr, ones])  # (N, 3)

        # Apply inverse homography
        H_inv = np.linalg.inv(self.H)
        img_pts_h = (H_inv @ world_pts_h.T).T  # (N, 3)

        # Normalize
        img_pts = img_pts_h[:, :2] / (img_pts_h[:, 2:3] + 1e-8)

        if is_2d:
            return img_pts.reshape(2)
        else:
            return img_pts.reshape(*input_shape)

    def validate_roundtrip(self, tolerance_mm: float = 1.0) -> bool:
        """Validate homography by roundtrip test.

        Args:
            tolerance_mm: Acceptable roundtrip error in millimeters

        Returns:
            True if all test points roundtrip within tolerance
        """
        # Test points across the calibration area
        test_img_pts = np.array(
            [
                [0, 0],
                [100, 0],
                [200, 0],
                [0, 50],
                [100, 50],
                [200, 50],
            ],
            dtype=np.float32,
        )

        # Roundtrip
        world_pts = self.image_to_world(test_img_pts)
        img_pts_back = self.world_to_image(world_pts)

        # Compute error
        errors = np.linalg.norm(test_img_pts - img_pts_back, axis=1)
        max_error = np.max(errors)

        # Convert pixel error to mm (assuming ~100 pixels = 1 meter)
        pixels_per_meter = 100  # Approximate
        max_error_mm = max_error / pixels_per_meter * 1000

        return max_error_mm <= tolerance_mm

    def get_world_bounds(self, img_h: int, img_w: int) -> tuple[float, float, float, float]:
        """Get world coordinate bounds for a given image size.

        Returns (x_min, x_max, y_min, y_max) in meters.
        """
        corners_img = np.array(
            [
                [0, 0],
                [img_w, 0],
                [0, img_h],
                [img_w, img_h],
            ],
            dtype=np.float32,
        )

        corners_world = self.image_to_world(corners_img)

        x_min = np.min(corners_world[:, 0])
        x_max = np.max(corners_world[:, 0])
        y_min = np.min(corners_world[:, 1])
        y_max = np.max(corners_world[:, 1])

        return x_min, x_max, y_min, y_max
