"""ByteTrack: Simple Yet Effective Multi-Object Tracking (Zangwang et al. 2022).

ByteTrack uses detections from any detector and tracks objects through frames using
Kalman filtering and Hungarian assignment. Key innovation: treat low-confidence
detections as a secondary matching pool for lost tracks.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class KalmanState:
    """Kalman filter state for object tracking.

    State: [center_x, center_y, aspect_ratio, height, vx, vy, va, vh]
    where v* are velocities in each dimension.
    """

    x: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    P: np.ndarray = field(
        default_factory=lambda: np.eye(8, dtype=np.float32) * 10
    )  # Covariance

    # Standard deviation of process and measurement noise
    std_weight_position: float = 1.0 / 20
    std_weight_velocity: float = 1.0 / 160

    def predict(self) -> np.ndarray:
        """Predict next state using constant-velocity model."""
        std_pos = (
            self.std_weight_position * self.x[3]
        )  # Std grows with height (bigger objects less certain)
        std_vel = self.std_weight_velocity * self.x[3]

        Q = np.diag(
            [std_pos, std_pos, 0, std_pos, std_vel, std_vel, 0, std_vel]
        ) ** 2
        self.P = self.P + Q

        # x_new = x_old + v * dt (dt=1 frame)
        x = self.x.copy()
        x[0] += self.x[4]
        x[1] += self.x[5]
        x[2] += self.x[6]
        x[3] += self.x[7]

        return x

    def update(self, z: np.ndarray) -> None:
        """Update state with measurement z = [cx, cy, ar, h]."""
        # Measurement covariance
        std_pos = self.std_weight_position * z[3]
        std_height = 0.1 * z[3]
        R = np.diag(
            [std_pos, std_pos, 0.1, std_height]
        ) ** 2

        # Measurement function: observe position + aspect ratio + height
        H = np.eye(4, 8)

        # Innovation
        y = z - H @ self.x
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S + 1e-8)

        # Update state and covariance
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P


@dataclass
class Track:
    """Active track of a detected object."""

    track_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    frame_idx: int

    state: KalmanState = field(default_factory=KalmanState)
    age: int = 0  # Frames since track was created
    hits: int = 0  # Number of successful detections matched to this track
    time_since_update: int = 0  # Frames since last detection match


class ByteTrack:
    """ByteTrack multi-object tracker with Kalman filtering and Hungarian assignment.

    Tracks bounding boxes across frames. Uses detection confidence to handle
    high-confidence (primary matching) and low-confidence (secondary matching) pools.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 1,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
    ):
        """Initialize ByteTrack.

        Args:
            max_age: Maximum frames a track can survive without a detection match
            min_hits: Minimum matches before a track is considered confirmed
            iou_threshold: IoU threshold for matching
            confidence_threshold: Split detection pool at this confidence
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

        self.tracks: list[Track] = []
        self.next_id = 1
        self.frame_idx = 0

    def update(
        self,
        detections: list[tuple[float, float, float, float, float]],
    ) -> dict[int, tuple[float, float, float, float]]:
        """Update tracks with new detections.

        Args:
            detections: List of (x1, y1, x2, y2, confidence) tuples

        Returns:
            Dictionary mapping track_id -> bbox (x1, y1, x2, y2)
        """
        self.frame_idx += 1

        # Split detections by confidence
        detections = np.array(detections, dtype=np.float32)
        if len(detections) == 0:
            detections_high = np.empty((0, 5), dtype=np.float32)
            detections_low = np.empty((0, 5), dtype=np.float32)
        else:
            high_mask = detections[:, 4] >= self.confidence_threshold
            detections_high = detections[high_mask]
            detections_low = detections[~high_mask]

        # Predict track positions
        self._predict_tracks()

        # First matching stage: high-confidence detections vs all tracks
        if len(detections_high) > 0:
            matched_high, unmatched_det_high, unmatched_trk = self._match(
                self.tracks, detections_high
            )
        else:
            matched_high = []
            unmatched_det_high = np.arange(len(detections_high))
            unmatched_trk = np.arange(len(self.tracks))

        # Update matched tracks
        for trk_idx, det_idx in matched_high:
            bbox = tuple(detections_high[det_idx, :4])
            conf = float(detections_high[det_idx, 4])
            self._update_track(self.tracks[trk_idx], bbox, conf)

        # Second matching stage: low-confidence detections vs unmatched tracks
        if len(detections_low) > 0 and len(unmatched_trk) > 0:
            unmatched_tracks = [self.tracks[i] for i in unmatched_trk]
            matched_low, unmatched_det_low, unmatched_trk_2 = self._match(
                unmatched_tracks, detections_low
            )

            for trk_idx_local, det_idx in matched_low:
                trk_idx_global = unmatched_trk[trk_idx_local]
                bbox = tuple(detections_low[det_idx, :4])
                conf = float(detections_low[det_idx, 4])
                self._update_track(self.tracks[trk_idx_global], bbox, conf)

            unmatched_trk = unmatched_trk[unmatched_trk_2]
        else:
            unmatched_det_low = np.arange(len(detections_low))

        # Create new tracks from unmatched high-confidence detections
        for det_idx in unmatched_det_high:
            bbox = tuple(detections_high[det_idx, :4])
            conf = float(detections_high[det_idx, 4])
            self._create_track(bbox, conf)

        # Remove dead tracks (age-out only — tentative tracks stay until max_age)
        self.tracks = [
            trk
            for trk in self.tracks
            if trk.time_since_update < self.max_age
        ]

        # Return confirmed tracks only
        result = {}
        for trk in self.tracks:
            if trk.hits >= self.min_hits:
                result[trk.track_id] = trk.bbox

        return result

    def _predict_tracks(self) -> None:
        """Predict all tracks forward one frame."""
        for track in self.tracks:
            x_pred = track.state.predict()
            track.bbox = self._state_to_bbox(x_pred)
            track.age += 1
            track.time_since_update += 1

    def _match(
        self,
        tracks: list[Track],
        detections: np.ndarray,
    ) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
        """Match detections to tracks using Hungarian algorithm.

        Args:
            tracks: List of Track objects
            detections: (M, 5) array of detections [x1, y1, x2, y2, conf]

        Returns:
            (matched_pairs, unmatched_det_indices, unmatched_trk_indices)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], np.arange(len(detections)), np.arange(len(tracks))

        # Compute cost matrix (negative IoU)
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = self._iou(track.bbox, tuple(det[:4]))
                cost_matrix[i, j] = -iou  # Negative because linear_sum_assignment minimizes

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter by IoU threshold
        matched = []
        for i, j in zip(row_ind, col_ind):
            if -cost_matrix[i, j] >= self.iou_threshold:
                matched.append((i, j))

        matched = np.array(matched, dtype=np.int32) if matched else np.empty((0, 2), dtype=np.int32)

        unmatched_det = np.setdiff1d(np.arange(len(detections)), matched[:, 1] if len(matched) > 0 else [])
        unmatched_trk = np.setdiff1d(np.arange(len(tracks)), matched[:, 0] if len(matched) > 0 else [])

        return matched.tolist(), unmatched_det, unmatched_trk

    def _update_track(self, track: Track, bbox: tuple, conf: float) -> None:
        """Update a track with a new detection."""
        track.bbox = bbox
        track.confidence = conf
        track.frame_idx = self.frame_idx
        track.time_since_update = 0
        track.hits += 1

        # Update Kalman state
        z = self._bbox_to_state(bbox)
        track.state.update(z)

    def _create_track(self, bbox: tuple, conf: float) -> None:
        """Create a new track from a detection."""
        track = Track(
            track_id=self.next_id,
            bbox=bbox,
            confidence=conf,
            frame_idx=self.frame_idx,
        )
        track.state.x = self._bbox_to_state_full(bbox)
        track.hits = 1  # creation counts as first detection
        self.tracks.append(track)
        self.next_id += 1

    @staticmethod
    def _bbox_to_state(bbox: tuple[float, float, float, float]) -> np.ndarray:
        """Convert bbox to Kalman measurement [cx, cy, ar, h]."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        ar = w / (h + 1e-8)
        return np.array([cx, cy, ar, h], dtype=np.float32)

    @staticmethod
    def _bbox_to_state_full(bbox: tuple[float, float, float, float]) -> np.ndarray:
        """Convert bbox to full Kalman state [cx, cy, ar, h, vx, vy, va, vh]."""
        z = ByteTrack._bbox_to_state(bbox)
        v = np.zeros(4, dtype=np.float32)
        return np.concatenate([z, v])

    @staticmethod
    def _state_to_bbox(state: np.ndarray) -> tuple[float, float, float, float]:
        """Convert Kalman state to bbox."""
        cx, cy, ar, h = state[:4]
        w = ar * h
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return (x1, y1, x2, y2)

    @staticmethod
    def _iou(bbox1: tuple, bbox2: tuple) -> float:
        """Compute Intersection over Union."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / (union_area + 1e-8)
