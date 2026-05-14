"""Per-frame skeleton overlay renderer for RTMPose Pass 1 debug visualization.

Renders smoothed keypoints, skeleton connections, track ID, frame number, and
timestamp onto original video frames. Color encodes side (left/right/center)
and confidence level so pathological gait patterns are immediately visible.
"""

from typing import Optional

import cv2
import numpy as np

from ..core.keypoints import KeypointSchema


# ── COCO-17 skeleton: (from_idx, to_idx) ────────────────────────────────────
# RTMPoseWholebody133 stores COCO body as indices 0–16 in standard order:
#   0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
#   5=L_shoulder, 6=R_shoulder, 7=L_elbow, 8=R_elbow,
#   9=L_wrist, 10=R_wrist, 11=L_hip, 12=R_hip,
#   13=L_knee, 14=R_knee, 15=L_ankle, 16=R_ankle
COCO17_CONNECTIONS: list[tuple[int, int]] = [
    # left leg
    (15, 13), (13, 11),
    # right leg
    (16, 14), (14, 12),
    # hip bar + torso
    (11, 12), (5, 11), (6, 12),
    # shoulder bar + arms
    (5, 6),
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    # head connections
    (0, 5), (0, 6),    # nose → shoulders
    (0, 1), (0, 2),    # nose → eyes
    (1, 3), (2, 4),    # eyes → ears
    (3, 5), (4, 6),    # ears → shoulders
]

# Left / right / center body partition (COCO-17)
_LEFT_KPTS  = frozenset({1, 3, 5, 7,  9, 11, 13, 15})
_RIGHT_KPTS = frozenset({2, 4, 6, 8, 10, 12, 14, 16})

# BGR display colors
_C_LEFT_HI    = (255, 200,  50)   # cyan-blue  — left, high confidence
_C_RIGHT_HI   = (  0, 165, 255)   # orange     — right, high confidence
_C_CENTER_HI  = ( 50, 230,  50)   # lime green — center, high confidence
_C_MEDIUM     = (  0, 220, 220)   # yellow     — medium confidence
_C_LOW        = (  0,  60, 220)   # red        — low confidence
_C_VERY_LOW   = (110, 110, 110)   # grey       — below threshold
_C_BONE       = (160, 160, 160)   # bone connection default
_C_BONE_MED   = (  0, 200, 200)   # bone when one endpoint is medium conf
_C_BONE_LOW   = (  0,  40, 180)   # bone when one endpoint is low conf

# Confidence thresholds
_CONF_HIGH   = 0.60
_CONF_MEDIUM = 0.35
_CONF_LOW    = 0.10


def _kpt_color(idx: int, conf: float) -> tuple[int, int, int]:
    if conf < _CONF_LOW:
        return _C_VERY_LOW
    if conf < _CONF_MEDIUM:
        return _C_LOW
    if conf < _CONF_HIGH:
        return _C_MEDIUM
    if idx in _LEFT_KPTS:
        return _C_LEFT_HI
    if idx in _RIGHT_KPTS:
        return _C_RIGHT_HI
    return _C_CENTER_HI


def _bone_color(conf_a: float, conf_b: float) -> Optional[tuple[int, int, int]]:
    """Return bone color, or None to skip (both endpoints below floor)."""
    min_c = min(conf_a, conf_b)
    if min_c < _CONF_LOW:
        return None
    if min_c < _CONF_MEDIUM:
        return _C_BONE_LOW
    if min_c < _CONF_HIGH:
        return _C_BONE_MED
    return _C_BONE


class SkeletonRenderer:
    """Renders RTMPose skeleton + diagnostic HUD on a single BGR frame.

    Usage:
        renderer = SkeletonRenderer()
        annotated = renderer.render_frame(frame, keypoints, schema, idx, ts, tid)

    The renderer is stateless — every call is independent. Thread-safe.
    """

    def render_frame(
        self,
        frame: np.ndarray,
        keypoints: Optional[np.ndarray],
        schema: KeypointSchema,
        frame_idx: int,
        timestamp: float,
        track_id: int,
    ) -> np.ndarray:
        """Overlay skeleton and HUD onto one video frame.

        Args:
            frame: (H, W, 3) BGR array — not modified in-place.
            keypoints: (n_kpts, 3) array [x, y, conf], or None for missing frame.
            schema: KeypointSchema for the active pose model.
            frame_idx: Frame number within the video.
            timestamp: Seconds from video start.
            track_id: ByteTrack track ID locked to the patient.

        Returns:
            Annotated BGR frame of the same shape.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        self._draw_hud(out, frame_idx, timestamp, track_id, w, h)

        if keypoints is None or len(keypoints) == 0:
            _put_text_centered(out, "NO KEYPOINTS", w // 2, h // 2, (0, 0, 220), scale=0.9)
            return out

        n_available = len(keypoints)

        # ── Skeleton bones ───────────────────────────────────────────────────
        for idx_a, idx_b in COCO17_CONNECTIONS:
            if idx_a >= n_available or idx_b >= n_available:
                continue
            xa, ya, ca = keypoints[idx_a]
            xb, yb, cb = keypoints[idx_b]
            col = _bone_color(float(ca), float(cb))
            if col is None:
                continue
            cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), col, 2, cv2.LINE_AA)

        # ── Keypoint circles (COCO-17 body only) ────────────────────────────
        for kp_idx in range(min(17, n_available)):
            x, y, c = keypoints[kp_idx]
            conf = float(c)
            col = _kpt_color(kp_idx, conf)
            radius = 5 if conf >= _CONF_MEDIUM else 3
            cv2.circle(out, (int(x), int(y)), radius, col, -1, cv2.LINE_AA)
            if conf < _CONF_MEDIUM:
                # extra ring marks interpolated / low-quality joints
                cv2.circle(out, (int(x), int(y)), radius + 3, (0, 30, 200), 1, cv2.LINE_AA)

        # ── Highlight key gait joints (ankles + knees) with larger radius ────
        for kp_idx in [schema.left_ankle, schema.right_ankle,
                       schema.left_knee, schema.right_knee]:
            if kp_idx >= n_available:
                continue
            x, y, c = keypoints[kp_idx]
            if float(c) >= _CONF_MEDIUM:
                col = _kpt_color(kp_idx, float(c))
                cv2.circle(out, (int(x), int(y)), 7, col, 2, cv2.LINE_AA)

        # ── Confidence summary + warning banner ──────────────────────────────
        body_confs = keypoints[:min(17, n_available), 2].astype(np.float32)
        mean_conf = float(np.mean(body_confs))
        very_low_frac = float(np.mean(body_confs < _CONF_LOW))

        conf_color = (
            (0, 200, 0) if mean_conf >= _CONF_HIGH
            else ((0, 200, 200) if mean_conf >= _CONF_MEDIUM else (0, 0, 220))
        )
        cv2.putText(
            out, f"mean conf: {mean_conf:.2f}",
            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, conf_color, 1, cv2.LINE_AA,
        )

        if very_low_frac > 0.50:
            # Semi-transparent red banner — occlusion or tracking loss
            banner = out[h - 40:h, :].copy()
            red_rect = np.full_like(banner, (0, 0, 160))
            cv2.addWeighted(banner, 0.45, red_rect, 0.55, 0, out[h - 40:h, :])
            cv2.putText(
                out, "LOW CONFIDENCE / OCCLUDED FRAME",
                (w // 2 - 155, h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA,
            )

        return out

    # ── Legend rendering (optional, shown on first frame only) ───────────────

    def render_legend(self, frame: np.ndarray) -> np.ndarray:
        """Overlay a color legend in the bottom-right corner.

        Call on the first frame so reviewers can decode the color scheme.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        entries = [
            ("Left side (high conf)",   _C_LEFT_HI),
            ("Right side (high conf)",  _C_RIGHT_HI),
            ("Center (high conf)",      _C_CENTER_HI),
            ("Medium conf (0.35-0.60)", _C_MEDIUM),
            ("Low conf (0.10-0.35)",    _C_LOW),
            ("Very low / occluded",     _C_VERY_LOW),
        ]
        box_w, box_h = 230, len(entries) * 20 + 14
        x0 = w - box_w - 8
        y0 = h - box_h - 8

        # Dark semi-transparent background
        roi = out[y0:y0 + box_h, x0:x0 + box_w]
        black = np.zeros_like(roi)
        cv2.addWeighted(roi, 0.35, black, 0.65, 0, out[y0:y0 + box_h, x0:x0 + box_w])

        for i, (label, color) in enumerate(entries):
            cy = y0 + 12 + i * 20
            cv2.circle(out, (x0 + 10, cy), 6, color, -1, cv2.LINE_AA)
            cv2.putText(
                out, label, (x0 + 22, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA,
            )
        return out

    # ─────────────────────────────────────────────────────────────────────────

    def _draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        track_id: int,
        w: int,
        h: int,
    ) -> None:
        """Top-left HUD: frame index, timestamp, track ID."""
        lines = [
            f"Frame  {frame_idx:06d}",
            f"Time   {timestamp:7.3f}s",
            f"Track  {track_id}",
        ]
        x0, y0, bw, bh = 6, 6, 170, 64
        roi = frame[y0:y0 + bh, x0:x0 + bw]
        black = np.zeros_like(roi)
        cv2.addWeighted(roi, 0.35, black, 0.65, 0, frame[y0:y0 + bh, x0:x0 + bw])
        for i, line in enumerate(lines):
            cv2.putText(
                frame, line, (x0 + 6, y0 + 18 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA,
            )


def _put_text_centered(
    frame: np.ndarray,
    text: str,
    cx: int,
    cy: int,
    color: tuple[int, int, int],
    scale: float = 0.7,
) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
