"""RTMPose-l WholeBody pose estimation via ONNX Runtime.

RTMPose outputs SimCC (Spatial-Channel Separable Coordinate) logits, which must be decoded
to get final x,y coordinates. This module handles ONNX inference + SimCC decoding.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from stride.core import BoundingBox, KeypointFrame, KeypointSchema, PoseEstimator
from stride.core.keypoints import RTMPoseWholebody133


class RTMPoseEstimator:
    """RTMPose-l WholeBody 133-keypoint pose estimator using ONNX Runtime.

    Attributes:
        model_path: Path to ONNX model file
        input_size: Model input dimensions (default 384x384)
        simcc_split_ratio: SimCC decoding parameter (default 2)
        device: CPU or CUDA (default CPU)
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (256, 192),
        simcc_split_ratio: float = 2.0,
        device: str = "cpu",
    ):
        """Initialize RTMPose estimator.

        Args:
            model_path: Path to ONNX model
            input_size: Model input size (H, W)
            simcc_split_ratio: Decoder parameter for SimCC logits
            device: Execution device (cpu or cuda)

        Raises:
            FileNotFoundError: If model_path does not exist
            Exception: If ONNX Runtime cannot load the model
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio
        self.keypoint_schema = RTMPoseWholebody133

        # Setup ONNX session
        providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def estimate(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
    ) -> KeypointFrame:
        """Estimate pose from a video frame within a bounding box.

        Args:
            frame: Video frame (H, W, 3) in BGR format
            bbox: Bounding box for the person (x1, y1, x2, y2)

        Returns:
            KeypointFrame with 133 keypoints and confidence scores
        """
        # Crop and preprocess
        cropped, affine_matrix = self._crop_and_preprocess(frame, bbox)

        # ONNX inference
        input_dict = {self.input_name: cropped}
        outputs = self.session.run(self.output_names, input_dict)

        # Decode SimCC outputs
        if len(outputs) >= 2:
            simcc_x_logits = outputs[0]  # (1, n_keypoints, simcc_bins)
            simcc_y_logits = outputs[1]
        else:
            # Fallback: single output containing both
            combined = outputs[0]
            n_keypoints = self.keypoint_schema.n_keypoints
            simcc_x_logits = combined[:, :n_keypoints, :]
            simcc_y_logits = combined[:, n_keypoints:, :]

        keypoints = self._decode_simcc(simcc_x_logits, simcc_y_logits, affine_matrix)

        # Compute frame-level confidence as mean of all keypoint confidences
        confidence = float(np.mean(keypoints[:, 2]))

        return KeypointFrame(
            keypoints=keypoints,
            confidence=confidence,
        )

    def _crop_and_preprocess(
        self, frame: np.ndarray, bbox: BoundingBox
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop frame to bounding box and resize to model input size.

        Args:
            frame: Original frame (H, W, 3) BGR
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Tuple of (preprocessed_input, affine_matrix)
            - preprocessed_input: (1, 3, H_input, W_input) normalized float32
            - affine_matrix: (2, 3) matrix for inverse transform
        """
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        w_bbox = x2 - x1
        h_bbox = y2 - y1

        # Expand bbox slightly to include surrounding context
        scale = 1.25
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        w_scaled = w_bbox * scale
        h_scaled = h_bbox * scale

        x1_scaled = max(0, int(center_x - w_scaled / 2))
        y1_scaled = max(0, int(center_y - h_scaled / 2))
        x2_scaled = min(frame.shape[1], int(center_x + w_scaled / 2))
        y2_scaled = min(frame.shape[0], int(center_y + h_scaled / 2))

        cropped = frame[y1_scaled:y2_scaled, x1_scaled:x2_scaled, :]

        import cv2

        # Letterbox resize: maintain aspect ratio, pad with gray (128) to fill target
        # This avoids aspect-ratio distortion that degrades RTMPose accuracy on portrait videos
        crop_h, crop_w = cropped.shape[:2]
        target_w, target_h = self.input_size[1], self.input_size[0]
        scale_f = min(target_w / max(crop_w, 1), target_h / max(crop_h, 1))
        new_w = int(crop_w * scale_f)
        new_h = int(crop_h * scale_f)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        # Affine: crop coords → letterboxed model coords
        src_pts = np.array(
            [[0, 0], [crop_w, 0], [0, crop_h]],
            dtype=np.float32,
        )
        dst_pts = np.array(
            [[pad_x, pad_y], [pad_x + new_w, pad_y], [pad_x, pad_y + new_h]],
            dtype=np.float32,
        )
        affine = self._get_affine_matrix(src_pts, dst_pts)

        warped = cv2.warpAffine(
            cropped,
            affine[:2, :],
            (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(128, 128, 128),
        )

        # Normalize: BGR -> RGB, divide by 255, subtract mean, divide by std
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = rgb / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std

        # CHW format for ONNX
        chw = np.transpose(rgb, (2, 0, 1))
        batch = np.expand_dims(chw, 0).astype(np.float32)

        # Affine for inverse transform (to map from model output back to original frame)
        # We need: model_coords -> cropped_coords -> original_coords
        inv_affine = self._get_affine_matrix(dst_pts, src_pts)
        # Translate to original frame coords
        offset_affine = np.eye(3)
        offset_affine[0, 2] = x1_scaled
        offset_affine[1, 2] = y1_scaled
        full_inv_affine = offset_affine @ inv_affine

        return batch, full_inv_affine

    def _get_affine_matrix(
        self, src_pts: np.ndarray, dst_pts: np.ndarray
    ) -> np.ndarray:
        """Compute affine transformation matrix from source to destination points.

        Args:
            src_pts: Source points (3, 2)
            dst_pts: Destination points (3, 2)

        Returns:
            Affine matrix (3, 3)
        """
        # Use cv2 for robustness
        import cv2

        affine_2x3 = cv2.getAffineTransform(src_pts, dst_pts)
        affine_3x3 = np.vstack([affine_2x3, [0, 0, 1]])
        return affine_3x3

    def _decode_simcc(
        self,
        simcc_x_logits: np.ndarray,
        simcc_y_logits: np.ndarray,
        affine_matrix: np.ndarray,
    ) -> np.ndarray:
        """Decode SimCC logits to keypoint coordinates.

        Args:
            simcc_x_logits: X coordinate logits (1, n_keypoints, simcc_bins)
            simcc_y_logits: Y coordinate logits (1, n_keypoints, simcc_bins)
            affine_matrix: (3, 3) matrix to transform to original frame coords

        Returns:
            Keypoints array (n_keypoints, 3) with [x, y, confidence]
        """
        n_keypoints = self.keypoint_schema.n_keypoints

        # Squeeze batch dimension
        simcc_x = simcc_x_logits.squeeze(0)  # (n_keypoints, simcc_bins)
        simcc_y = simcc_y_logits.squeeze(0)

        # Decode by argmax over bins
        # Grid spacing in normalized coordinates
        grid_step = 1.0 / self.simcc_split_ratio

        x_indices = np.argmax(simcc_x, axis=1)  # (n_keypoints,)
        y_indices = np.argmax(simcc_y, axis=1)

        # Convert indices to normalized coordinates [0, input_size]
        x_norm = (x_indices.astype(np.float32) + 0.5) * grid_step
        y_norm = (y_indices.astype(np.float32) + 0.5) * grid_step

        # Compute confidence from logit softmax
        x_probs = np.exp(simcc_x - np.max(simcc_x, axis=1, keepdims=True))
        x_probs = x_probs / np.sum(x_probs, axis=1, keepdims=True)
        x_conf = np.max(x_probs, axis=1)

        y_probs = np.exp(simcc_y - np.max(simcc_y, axis=1, keepdims=True))
        y_probs = y_probs / np.sum(y_probs, axis=1, keepdims=True)
        y_conf = np.max(y_probs, axis=1)

        confidence = np.sqrt(x_conf * y_conf)  # Joint confidence

        # Transform to original frame coordinates
        pts_norm = np.column_stack([x_norm, y_norm, np.ones(n_keypoints)])  # (n_keypoints, 3)
        pts_orig = (affine_matrix @ pts_norm.T).T  # (n_keypoints, 3)

        # pts_orig are now in original frame coordinates — keep as-is
        keypoints = np.column_stack([pts_orig[:, 0], pts_orig[:, 1], confidence])
        return keypoints.astype(np.float32)
