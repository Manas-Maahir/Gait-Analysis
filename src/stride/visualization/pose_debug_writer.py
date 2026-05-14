"""RTMPose Pass 1 debug video exporter.

Re-reads the original video a second time (after Pass 1 completes), overlays
smoothed keypoints and diagnostic annotations on every frame, and writes a
self-contained MP4 to the requested output path.

This is purely a debug/visualization tool. It does NOT modify Pass1Result or
any downstream pipeline state.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..pipeline.context import Pass1Result
from .skeleton_renderer import SkeletonRenderer


class PoseDebugWriter:
    """Writes a skeleton-overlay debug video from a completed Pass1Result.

    The writer re-opens the original video file so that:
    - The main frame loop in run_pass1() is unmodified.
    - No raw frames are kept in memory during Pass 1.
    - Output is only produced when explicitly requested.

    Usage:
        writer = PoseDebugWriter()
        writer.write(video_path, pass1_result, output_path)
    """

    def __init__(self, renderer: Optional[SkeletonRenderer] = None) -> None:
        self._renderer = renderer or SkeletonRenderer()

    def write(
        self,
        video_path: str,
        pass1_result: Pass1Result,
        output_path: Path,
        verbose: bool = False,
    ) -> None:
        """Write debug MP4 with skeleton overlay to output_path.

        Args:
            video_path: Path to the original input video (re-opened read-only).
            pass1_result: Completed Pass1Result — provides smoothed keypoints,
                          timestamps, track IDs, fps, and schema.
            output_path: Destination .mp4 file path. Parent dir is created if needed.
            verbose: Print progress to stdout.

        Raises:
            IOError: If the original video cannot be re-opened.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(
                f"PoseDebugWriter: cannot re-open video for rendering: {video_path}"
            )

        fps     = pass1_result.fps
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = pass1_result.total_frames

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if verbose:
            print(f"[PoseDebugWriter] Rendering {n_frames} frames → {output_path}")

        try:
            for frame_idx in range(n_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Resolve per-frame data from Pass1Result
                if frame_idx < len(pass1_result.keypoints):
                    kps      = pass1_result.keypoints[frame_idx]   # (n_kpts, 3)
                    ts       = float(pass1_result.timestamps[frame_idx])
                    track_id = int(pass1_result.track_ids[frame_idx])
                else:
                    kps      = None
                    ts       = float(frame_idx) / fps
                    track_id = -1

                # First frame: overlay legend
                if frame_idx == 0:
                    frame = self._renderer.render_legend(frame)

                rendered = self._renderer.render_frame(
                    frame=frame,
                    keypoints=kps,
                    schema=pass1_result.schema,
                    frame_idx=frame_idx,
                    timestamp=ts,
                    track_id=track_id,
                )

                # Overlay bbox if available (debug: shows region fed to RTMPose)
                if pass1_result.bboxes is not None and frame_idx < len(pass1_result.bboxes):
                    x1, y1, x2, y2 = pass1_result.bboxes[frame_idx]
                    cv2.rectangle(rendered, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                writer.write(rendered)

                if verbose and frame_idx % max(1, n_frames // 20) == 0:
                    pct = frame_idx / n_frames * 100
                    print(f"  [{pct:5.1f}%] frame {frame_idx}/{n_frames}", flush=True)

        finally:
            cap.release()
            writer.release()

        if verbose:
            print(f"[PoseDebugWriter] Done → {output_path}")
