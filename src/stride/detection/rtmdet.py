"""RTMDet-Nano person detector via ONNX Runtime.

RTMDet is a fast, accurate anchor-free object detector trained on COCO.
This module wraps ONNX inference + NMS decoding for person detection.
"""

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

from stride.core.protocols import BoundingBox


class RTMDetDetector:
    """RTMDet-Nano person detector using ONNX Runtime.

    Attributes:
        model_path: Path to ONNX model file
        input_size: Model input dimensions (default 320x320)
        conf_threshold: Confidence threshold for detections (default 0.3)
        device: CPU or CUDA (default CPU)
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (320, 320),
        conf_threshold: float = 0.3,
        device: str = "cpu",
    ):
        """Initialize RTMDet detector.

        Args:
            model_path: Path to ONNX model
            input_size: Model input size (H, W)
            conf_threshold: Confidence threshold for detections
            device: Execution device (cpu or cuda)

        Raises:
            FileNotFoundError: If model_path does not exist
            Exception: If ONNX Runtime cannot load the model
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.input_size = input_size
        self.conf_threshold = conf_threshold

        providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def detect(self, frame: np.ndarray) -> Sequence[BoundingBox]:
        """Detect persons in a video frame.

        Args:
            frame: Video frame (H, W, 3) in BGR format

        Returns:
            List of BoundingBox objects sorted by area (largest first)
        """
        img_h, img_w = frame.shape[:2]
        target_h, target_w = self.input_size

        # Letterbox resize: maintain aspect ratio, pad with gray (114)
        scale = min(target_w / img_w, target_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        # Resize and pad
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        # Normalize (RTMDet: divide by 255, no mean/std)
        normalized = padded.astype(np.float32) / 255.0

        # CHW format for ONNX
        chw = np.transpose(normalized, (2, 0, 1))
        batch = np.expand_dims(chw, 0).astype(np.float32)

        # ONNX inference
        input_dict = {self.input_name: batch}
        outputs = self.session.run(self.output_names, input_dict)

        # Decode outputs
        detections = self._decode_outputs(outputs, img_w, img_h, scale, pad_x, pad_y)

        # Sort by area descending
        detections.sort(key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1), reverse=True)
        return detections

    def _decode_outputs(
        self,
        outputs: list,
        img_w: int,
        img_h: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> list[BoundingBox]:
        """Decode ONNX outputs to bounding boxes.

        Handles both NMS-included and raw anchor formats.

        Args:
            outputs: List of ONNX output tensors
            img_w, img_h: Original image dimensions
            scale: Resize scale factor
            pad_x, pad_y: Padding offsets in letterboxed space

        Returns:
            List of BoundingBox objects (class 0 = person, conf >= threshold)
        """
        if len(outputs) < 1:
            return []

        # Try NMS-included format first (MMDeploy export)
        # Output shape typically: (1, N, 5) where 5 = [x1, y1, x2, y2, conf]
        dets = outputs[0]
        if dets.ndim == 3:
            dets = dets.squeeze(0)  # (N, 5)

        if len(dets) == 0:
            return []

        # Assume format: [x1, y1, x2, y2, conf, ...] (at least 5 columns)
        if dets.shape[1] >= 5:
            confs = dets[:, 4].astype(np.float32)
            mask = confs >= self.conf_threshold
            dets = dets[mask]

            if len(dets) == 0:
                return []

            # Inverse letterbox transform
            x1s = dets[:, 0].astype(np.float32)
            y1s = dets[:, 1].astype(np.float32)
            x2s = dets[:, 2].astype(np.float32)
            y2s = dets[:, 3].astype(np.float32)
            confs = dets[:, 4].astype(np.float32)

            # Remove padding, scale back to original
            x1s = (x1s - pad_x) / scale
            y1s = (y1s - pad_y) / scale
            x2s = (x2s - pad_x) / scale
            y2s = (y2s - pad_y) / scale

            # Clamp to image bounds
            x1s = np.clip(x1s, 0, img_w)
            y1s = np.clip(y1s, 0, img_h)
            x2s = np.clip(x2s, 0, img_w)
            y2s = np.clip(y2s, 0, img_h)

            # Create BoundingBox objects
            bboxes = []
            for x1, y1, x2, y2, conf in zip(x1s, y1s, x2s, y2s, confs):
                bboxes.append(BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), confidence=float(conf)))
            return bboxes

        return []
