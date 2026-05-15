"""Main gait analysis pipeline orchestrator.

Two-pass design with dependency injection for all swappable components:
- Pass 1: Video I/O, pose estimation, tracking, keypoint smoothing → Pass1Result
- Pass 2: Spatial calibration, segmentation, gait event detection, metrics → Pass2Result

Uses immutable result dataclasses instead of mutable shared state.
"""

import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from ..config import StriderConfig
from ..core.protocols import (
    PoseEstimator,
    Tracker,
    PersonDetector,
    Calibrator,
    GaitEventDetector,
    MetricComputer,
    ClinicalAnalyzer,
    BoundingBox,
)
from ..core.types import Phase, Quartile
from ..data import AnalysisResult, ProcessingMetadata, GaitMetrics
from ..data.metrics import GlobalMetrics, QuartileMetrics
from ..data.clinical import ClinicalReport
from ..pose.smoother import smooth_keypoints
from ..tracking.bytetrack import ByteTrack
from ..calibration.homography import SVDAutoCalibrator
from ..calibration.spatial_mapper import SpatialMapper
from ..segmentation.phase_detector import PhaseDetector
from ..segmentation.turn_detector import TurnDetector
from ..segmentation.quartile_engine import QuartileEngine
from ..gait_events.foot_strike import FootStrikeDetector
from ..gait_events.step_validator import StepValidator
from ..gait_events.fog_detector import FOGDetector
from ..metrics.per_quartile import QuartileMetricsComputer
from .context import Pass1Result, Pass2Result


def run_pass1(
    video_path: str,
    config: StriderConfig,
    pose_estimator: Optional[PoseEstimator] = None,
    tracker: Optional[Tracker] = None,
    detector: Optional[PersonDetector] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    pose_debug_path: Optional[Path] = None,
) -> Pass1Result:
    """Execute Pass 1: video I/O, pose estimation, tracking, keypoint smoothing.

    Args:
        video_path: Path to input video file
        config: StriderConfig instance
        pose_estimator: PoseEstimator protocol implementation (optional)
        tracker: Tracker protocol implementation (optional)
        detector: PersonDetector protocol implementation (optional)
        progress_callback: Optional callback fn(progress_pct, stage_name)
        pose_debug_path: If set, write skeleton-overlay debug MP4 here after
                         the main frame loop completes (Pass 1 only).

    Returns:
        Pass1Result with raw keypoints, timestamps, track IDs
    """
    # Default implementations placeholder (will be filled later)
    if pose_estimator is None:
        raise NotImplementedError("pose_estimator must be provided (no default yet)")
    if tracker is None:
        raise NotImplementedError("tracker must be provided (no default yet)")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = None
    pass1_result: Optional[Pass1Result] = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if config.verbose:
            print(f"Video: {frame_width}x{frame_height} @ {fps} fps, {total_frames} frames")

        # Initialize ByteTrack directly (Phase 1: no protocol abstraction needed for concrete implementation)
        if tracker is None:
            bytetrack = ByteTrack(
                max_age=config.tracking_lost_threshold_frames,
                confidence_threshold=config.detection_score_threshold,
            )
        else:
            # If tracker is provided, we assume it's ByteTrack-compatible for Phase 1
            # Full protocol abstraction comes in Phase 2
            bytetrack = tracker

        schema = pose_estimator.keypoint_schema
        n_kpts = schema.n_keypoints

        # Output arrays for processed frames
        keypoints_list = []
        timestamps_list = []
        track_ids_list = []
        bboxes_list = []

        # Diagnostics (for debug CSV if debug_dir is set)
        confidence_debug_list = []

        # Phase 1: Patient selector (lock to largest bbox at frame 0)
        locked_track_id: Optional[int] = None
        frame_width_f = float(frame_width)
        frame_height_f = float(frame_height)

        # Full-frame fallback bbox (Phase 1: no RTMDet detector)
        fullframe_bbox = BoundingBox(
            x1=0.0, y1=0.0,
            x2=frame_width_f, y2=frame_height_f,
            confidence=1.0,
        )

        if progress_callback:
            progress_callback(0.05, "Loading video")

        # Frame loop: pose estimation + tracking
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= total_frames:
                break

            # Person detection: use RTMDet if available, else full-frame fallback
            if detector is not None:
                detected_bboxes = detector.detect(frame)
                detections = [(b.x1, b.y1, b.x2, b.y2, b.confidence) for b in detected_bboxes]
                if not detections:  # no person found → low-conf fallback
                    detections = [(0.0, 0.0, frame_width_f, frame_height_f, 0.5)]
                if config.verbose and frame_idx % 30 == 0:
                    print(f"  Frame {frame_idx}: {len(detected_bboxes)} RTMDet detection(s)")
            else:
                # Legacy full-frame fallback (reduced accuracy without detector)
                detections = [(0.0, 0.0, frame_width_f, frame_height_f, 1.0)]

            tracked = bytetrack.update(detections)

            # Patient selection: lock to largest area track at frame 0
            if frame_idx == 0 and tracked:
                best_id = max(
                    tracked.keys(),
                    key=lambda tid: (
                        (tracked[tid][2] - tracked[tid][0]) *
                        (tracked[tid][3] - tracked[tid][1])
                    ),
                )
                locked_track_id = best_id

            # Use locked track or fallback to full-frame
            if locked_track_id is not None and locked_track_id in tracked:
                x1, y1, x2, y2 = tracked[locked_track_id]
                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=1.0)
                track_ids_list.append(locked_track_id)
            else:
                bbox = fullframe_bbox
                track_ids_list.append(locked_track_id if locked_track_id else 0)

            # Store bbox for diagnostics
            bboxes_list.append([bbox.x1, bbox.y1, bbox.x2, bbox.y2])

            # Pose estimation
            try:
                kf = pose_estimator.estimate(frame, bbox)
                keypoints_list.append(kf.keypoints)
            except Exception as e:
                # Low confidence or error: append zeros (smoother handles this)
                if config.verbose:
                    print(f"Warning: Pose estimation failed at frame {frame_idx}: {e}")
                keypoints_list.append(np.zeros((n_kpts, 3), dtype=np.float32))

            timestamps_list.append(frame_idx / fps)

            # Confidence diagnostics
            if keypoints_list:
                kpts = keypoints_list[-1]
                body_confs = kpts[:17, 2]  # COCO-17 body only
                mean_conf = float(np.mean(body_confs)) if len(body_confs) > 0 else 0.0
                frac_below_low = float(np.mean(body_confs < 0.10))
                frac_below_med = float(np.mean(body_confs < 0.35))
                bbox_area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
                frame_area = float(frame_width * frame_height)
                bbox_area_pct = (bbox_area / frame_area) * 100 if frame_area > 0 else 0.0
                track_id = track_ids_list[-1] if track_ids_list else 0

                confidence_debug_list.append({
                    'frame_idx': frame_idx,
                    'timestamp': timestamps_list[-1],
                    'track_id': track_id,
                    'mean_conf': mean_conf,
                    'frac_below_low': frac_below_low,
                    'frac_below_med': frac_below_med,
                    'bbox_area_pct': bbox_area_pct,
                })

            # Report per-frame progress: 5%→60% range
            if progress_callback and frame_idx % max(1, total_frames // 100) == 0:
                pct = 0.05 + 0.55 * (frame_idx / total_frames)
                progress_callback(pct, f"Pose estimation frame {frame_idx}/{total_frames}")

            frame_idx += 1

        # Stack output arrays
        actual_frames = len(keypoints_list)
        keypoints = np.stack(keypoints_list, axis=0)  # (N, n_kpts, 3)
        timestamps = np.array(timestamps_list, dtype=np.float32)
        track_ids = np.array(track_ids_list, dtype=np.int32)
        bboxes = np.array(bboxes_list, dtype=np.float32) if bboxes_list else None

        # Smooth keypoints with OneEuro filter
        if progress_callback:
            progress_callback(0.58, "Smoothing keypoints")

        keypoints_smoothed = smooth_keypoints(
            keypoints, fps=fps,
            fcmin=1.0, beta=0.5,
        )

        pass1_result = Pass1Result(
            keypoints=keypoints_smoothed,
            timestamps=timestamps,
            track_ids=track_ids,
            fps=fps,
            total_frames=actual_frames,
            schema=schema,
            video_path=str(video_path),
            bboxes=bboxes,
        )

    finally:
        if cap is not None:
            cap.release()

    # Debug CSV export (confidence diagnostics) — if debug_dir is set
    if config.save_intermediate_data and pass1_result is not None and confidence_debug_list:
        try:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = config.output_dir / "confidence_per_frame.csv"
            import csv
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'frame_idx', 'timestamp', 'track_id', 'mean_conf',
                    'frac_below_low', 'frac_below_med', 'bbox_area_pct',
                ])
                writer.writeheader()
                writer.writerows(confidence_debug_list)
            if config.verbose:
                print(f"[Diagnostics] Confidence CSV → {csv_path}")
        except Exception as e:
            if config.verbose:
                print(f"Warning: confidence CSV export failed: {e}")

    # Debug video export (optional, Pass 1 only) — runs after cap is released
    if pose_debug_path is not None and pass1_result is not None:
        try:
            from ..visualization.pose_debug_writer import PoseDebugWriter
            PoseDebugWriter().write(
                video_path=str(video_path),
                pass1_result=pass1_result,
                output_path=pose_debug_path,
                verbose=config.verbose,
            )
        except Exception as _dbg_exc:
            if config.verbose:
                print(f"Warning: pose debug video export failed: {_dbg_exc}")

    return pass1_result


def run_pass2(
    pass1: Pass1Result,
    config: StriderConfig,
    calibrator: Optional[Calibrator] = None,
    event_detector: Optional[GaitEventDetector] = None,
    metric_computer: Optional[MetricComputer] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Pass2Result:
    """Execute Pass 2: spatial calibration, segmentation, gait event detection, metrics.

    Args:
        pass1: Pass1Result from run_pass1()
        config: StriderConfig instance
        calibrator: Calibrator protocol implementation (optional)
        event_detector: GaitEventDetector protocol implementation (optional)
        metric_computer: MetricComputer protocol implementation (optional)
        progress_callback: Optional callback fn(progress_pct, stage_name)

    Returns:
        Pass2Result with world positions, phases, events, metrics
    """
    schema = pass1.schema

    # ── STAGE 1: Calibration (60%→68%) ────────────────────────────────────
    if progress_callback:
        progress_callback(0.60, "Spatial calibration")

    # Extract ankle trajectory for SVD calibration
    left_ankle_xy = pass1.keypoints[:, schema.left_ankle, :2]   # (N, 2)
    right_ankle_xy = pass1.keypoints[:, schema.right_ankle, :2] # (N, 2)

    # Midpoint of both ankles (more stable than single ankle for straight-path fitting)
    ankle_midpoint = (left_ankle_xy + right_ankle_xy) / 2.0     # (N, 2)

    # Filter out frames with zero confidence (undetected)
    left_conf = pass1.keypoints[:, schema.left_ankle, 2]
    right_conf = pass1.keypoints[:, schema.right_ankle, 2]
    valid_mask = (left_conf > config.pose_confidence_threshold) | \
                 (right_conf > config.pose_confidence_threshold)

    valid_ankle_pts = ankle_midpoint[valid_mask]

    # Fallback: if too few high-confidence frames, use all non-zero ankle points
    # (argmax positions track the person even when SimCC confidence is uniformly low)
    if len(valid_ankle_pts) < 10:
        nonzero_mask = (ankle_midpoint[:, 0] != 0) | (ankle_midpoint[:, 1] != 0)
        valid_ankle_pts = ankle_midpoint[nonzero_mask] if nonzero_mask.any() else ankle_midpoint
        if config.verbose:
            print(f"Confidence-filtered ankle pts insufficient ({valid_mask.sum()}); "
                  f"using all {len(valid_ankle_pts)} non-zero frames for calibration.")

    # Calibration — use injected calibrator if provided, else SVD auto
    if calibrator is not None:
        calibration_result = calibrator.calibrate(valid_ankle_pts)
    else:
        auto_cal = SVDAutoCalibrator(
            path_length_m=config.path_length_meters,
            min_points=10,
        )
        try:
            calibration_result = auto_cal.calibrate(valid_ankle_pts)
            if config.verbose and calibration_result.pc1_variance < config.calibration_pc1_threshold:
                print(f"Warning: Auto-calibration low confidence (PC1={calibration_result.pc1_variance:.2f})")
        except Exception as e:
            if config.calibration_mode == "auto_with_fallback":
                # Fallback: identity-like homography (1 pixel = 1/100 meter)
                H_fallback = np.eye(3, dtype=np.float32)
                H_fallback[0, 0] = 1.0 / 100.0  # 100 px per meter
                H_fallback[1, 1] = 1.0 / 100.0
                from ..core.protocols import CalibrationResult
                calibration_result = CalibrationResult(
                    homography_matrix=H_fallback,
                    scale_px_to_m=0.01,
                    pc1_variance=0.0,
                    method="fallback_identity",
                )
                if config.verbose:
                    print(f"Calibration failed: {e}. Using fallback.")
            else:
                raise RuntimeError(f"Calibration failed: {e}")

    # ── STAGE 2: World coordinate transformation ───────────────────────────
    if progress_callback:
        progress_callback(0.63, "Mapping to world coordinates")

    mapper = SpatialMapper(calibration_result)

    # Map all ankle midpoints to world space: (N, 2) pixel → (N, 2) meters
    world_positions = mapper.image_to_world(ankle_midpoint)  # (N, 2)

    # ── STAGE 3: Phase detection (68%→72%) ─────────────────────────────────
    if progress_callback:
        progress_callback(0.68, "Detecting gait phases")

    phase_detector = PhaseDetector(
        fps=pass1.fps,
        turn_window_sec=0.5,
        velocity_zero_threshold=config.phase_velocity_threshold,
    )
    phases, (turn_start_frame, turn_end_frame) = phase_detector.detect(
        world_positions, pass1.timestamps
    )

    # Measure turning time with TurnDetector
    turn_detector = TurnDetector(
        fps=pass1.fps,
        velocity_threshold=config.phase_velocity_threshold,
    )
    turning_time_sec, (td_start, td_end) = turn_detector.detect_turn(
        world_positions, pass1.timestamps
    )

    # ── STAGE 4: Determine actual turn distance ────────────────────────────
    # Use the world_x at the turn midpoint as the actual turn distance
    turn_mid_frame = (turn_start_frame + turn_end_frame) // 2
    if turn_mid_frame < len(world_positions):
        actual_turn_distance_m = float(
            np.clip(world_positions[turn_mid_frame, 0], 0.01, config.path_length_meters * 1.5)
        )
    else:
        actual_turn_distance_m = config.path_length_meters

    # ── STAGE 5: Gait event detection (72%→75%) ───────────────────────────
    if progress_callback:
        progress_callback(0.72, "Detecting foot strikes")

    foot_strike_detector = FootStrikeDetector(
        fps=pass1.fps,
        min_step_interval_sec=config.foot_strike_min_interval_sec,
        max_step_interval_sec=config.foot_strike_max_interval_sec,
        min_step_length_m=0.2,
        max_step_length_m=2.0,
        peak_distance_sec=0.2,
        min_confidence=config.pose_confidence_threshold,
    )
    raw_events = foot_strike_detector.detect(
        keypoints=pass1.keypoints,
        world_positions=world_positions,
        timestamps=pass1.timestamps,
        schema=schema,
        phases=phases,
    )

    # Validate events
    validator = StepValidator(
        min_step_interval_sec=config.foot_strike_min_interval_sec,
        max_step_interval_sec=config.foot_strike_max_interval_sec,
        min_step_length_m=0.1,
        max_step_length_m=2.5,
    )
    valid_events, _ = validator.validate(raw_events)

    # ── STAGE 5b: FOG detection ────────────────────────────────────────────
    if progress_callback:
        progress_callback(0.74, "Detecting FOG episodes")

    fog_detector = FOGDetector(
        fps=pass1.fps,
        fi_threshold=config.fog_fi_threshold,
        window_sec=config.fog_window_sec,
        min_duration_sec=config.fog_min_duration_sec,
    )
    fog_episodes = fog_detector.detect(
        keypoints=pass1.keypoints,
        timestamps=pass1.timestamps,
        schema=schema,
    )

    # ── STAGE 6: Assign quartile to each event ─────────────────────────────
    # CRITICAL: FootStrikeEvent.phase must be set to Quartile BEFORE calling
    # QuartileMetricsComputer.compute(). The field is Optional[Quartile].
    quartile_engine = QuartileEngine(
        path_length_m=config.path_length_meters,
        turn_distance_m=actual_turn_distance_m,
    )

    for event in valid_events:
        event_phase = phases[event.frame_idx] if event.frame_idx < len(phases) else Phase.TOWARD
        assigned_quartile = quartile_engine.assign_quartile(event.world_x, event_phase)
        event.quartile = assigned_quartile

    # ── STAGE 7: Metrics computation (75%→85%) ────────────────────────────
    if progress_callback:
        progress_callback(0.75, "Computing quartile metrics")

    metrics_computer = QuartileMetricsComputer(
        path_length_m=config.path_length_meters,
        quartile_boundaries_m=(config.path_length_meters / 2.0, config.path_length_meters),
    )
    quartile_metrics_enum_keys = metrics_computer.compute(
        events=valid_events,
        timestamps=pass1.timestamps,
    )

    # Convert Quartile enum keys → string keys for Pass2Result storage
    quartile_metrics_str_keys = {
        q.value: qm for q, qm in quartile_metrics_enum_keys.items()
    }

    # Build phase boundary dicts
    phase_start_frames: dict[Phase, int] = {}
    phase_end_frames: dict[Phase, int] = {}

    for phase_val in [Phase.TOWARD, Phase.TURN, Phase.AWAY]:
        indices = np.where(phases == phase_val)[0]
        if len(indices) > 0:
            phase_start_frames[phase_val] = int(indices[0])
            phase_end_frames[phase_val] = int(indices[-1])
        else:
            # Phase not detected; use sensible defaults
            phase_start_frames[phase_val] = 0
            phase_end_frames[phase_val] = 0

    if progress_callback:
        progress_callback(0.85, "Pass 2 complete")

    return Pass2Result(
        world_positions=world_positions,
        phases=phases,
        foot_strikes=tuple(valid_events),
        fog_episodes=tuple(fog_episodes),
        calibration=calibration_result,
        actual_turn_distance_m=actual_turn_distance_m,
        phase_start_frames=phase_start_frames,
        phase_end_frames=phase_end_frames,
        quartile_metrics=quartile_metrics_str_keys,
        turning_time_sec=turning_time_sec,
    )


class GaitProcessor:
    """Orchestrates the two-pass gait analysis pipeline.

    Accepts pluggable protocol implementations for all components, enabling:
    - Runtime model swapping (RTMPose ↔ MediaPipe)
    - Testability (inject mocks for unit testing)
    - Reentrancy (immutable results between passes)
    """

    def __init__(
        self,
        config: Optional[StriderConfig] = None,
        pose_estimator: Optional[PoseEstimator] = None,
        tracker: Optional[Tracker] = None,
        detector: Optional[PersonDetector] = None,
        calibrator: Optional[Calibrator] = None,
        event_detector: Optional[GaitEventDetector] = None,
        metric_computer: Optional[MetricComputer] = None,
        clinical_analyzer: Optional[ClinicalAnalyzer] = None,
        debug_dir: Optional[Path] = None,
        pose_debug_path: Optional[Path] = None,
    ):
        """Initialize the gait processor with optional DI.

        Args:
            config: StriderConfig instance. If None, uses defaults.
            pose_estimator: PoseEstimator protocol. If None, will raise error in run_pass1.
            tracker: Tracker protocol. If None, will raise error in run_pass1.
            detector: PersonDetector protocol. If None, uses full-frame fallback.
            calibrator: Calibrator protocol. If None, will raise error in run_pass2.
            event_detector: GaitEventDetector protocol. If None, will raise error in run_pass2.
            metric_computer: MetricComputer protocol. If None, will raise error in run_pass2.
            clinical_analyzer: ClinicalAnalyzer protocol. Optional.
            debug_dir: If set, save keypoints.npy / world_positions.npy / summary.txt here.
            pose_debug_path: If set, write RTMPose skeleton-overlay debug MP4 here after Pass 1.
        """
        self.config = config or StriderConfig()
        self._pose_estimator = pose_estimator
        self._tracker = tracker
        self._detector = detector
        self._calibrator = calibrator
        self._event_detector = event_detector
        self._metric_computer = metric_computer
        self._clinical_analyzer = clinical_analyzer
        self._debug_dir = debug_dir
        self._pose_debug_path = pose_debug_path

    def process(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> AnalysisResult:
        """Process a video end-to-end and return gait analysis results.

        Args:
            video_path: Path to input video file
            progress_callback: Optional callback fn(progress_pct, stage_name)
                              for monitoring long-running analysis

        Returns:
            AnalysisResult containing all metrics, events, and metadata
        """
        start_time = time.time()

        def report_progress(pct: float, stage: str):
            """Report progress to callback if provided."""
            if progress_callback:
                progress_callback(pct, stage)
            if self.config.verbose:
                print(f"[{pct*100:.1f}%] {stage}")

        try:
            # Pass 1: Raw data extraction
            report_progress(0.05, "Initializing video reader")
            report_progress(0.20, "Pass 1: Pose estimation and tracking")

            pass1 = run_pass1(
                video_path,
                self.config,
                pose_estimator=self._pose_estimator,
                tracker=self._tracker,
                detector=self._detector,
                progress_callback=report_progress,
                pose_debug_path=self._pose_debug_path,
            )

            # Pass 2: Metric computation
            report_progress(0.65, "Computing spatial calibration")
            report_progress(0.70, "Pass 2: Phase detection and quartile assignment")

            pass2 = run_pass2(
                pass1,
                self.config,
                calibrator=self._calibrator,
                event_detector=self._event_detector,
                metric_computer=self._metric_computer,
                progress_callback=report_progress,
            )

            # Debug dump: save raw keypoints + world positions + phases
            if self._debug_dir is not None:
                self._debug_dir.mkdir(parents=True, exist_ok=True)
                np.save(str(self._debug_dir / "keypoints.npy"), pass1.keypoints)
                np.save(str(self._debug_dir / "world_positions.npy"), pass2.world_positions)
                np.save(str(self._debug_dir / "phases.npy"), pass2.phases)
                ankle_conf = pass1.keypoints[:, pass1.schema.left_ankle, 2]
                with open(self._debug_dir / "summary.txt", "w", encoding="utf-8") as f:
                    f.write(f"Frames: {pass1.total_frames}\n")
                    f.write(f"Left ankle conf  mean={ankle_conf.mean():.3f}  min={ankle_conf.min():.3f}  max={ankle_conf.max():.3f}\n")
                    phases_arr = pass2.phases
                    unique, counts = np.unique(phases_arr, return_counts=True)
                    for ph, ct in zip(unique, counts):
                        f.write(f"Phase {ph}: {ct} frames\n")
                    f.write(f"World X range: {pass2.world_positions[:, 0].min():.2f} to {pass2.world_positions[:, 0].max():.2f} m\n")
                    f.write(f"Foot strikes detected: {len(pass2.foot_strikes)}\n")
                if self.config.verbose:
                    print(f"Debug output saved to: {self._debug_dir}")

            # Assemble result
            report_progress(0.90, "Assembling results")

            elapsed_sec = time.time() - start_time
            metadata = ProcessingMetadata(
                video_path=str(video_path),
                fps=pass1.fps,
                total_frames=pass1.total_frames,
                processing_time_sec=elapsed_sec,
                config_hash=self.config.config_hash,
            )

            # Build GlobalMetrics from pass2 data
            # Total steps = sum across Q1+Q2+Q3+Q4 (excluding TURN)
            total_steps = len([
                e for e in pass2.foot_strikes
                if e.quartile in (Quartile.Q1, Quartile.Q2, Quartile.Q3, Quartile.Q4)
            ])

            # Compute phase durations from timestamps
            def _phase_duration(phase_val: Phase) -> float:
                start_f = pass2.phase_start_frames.get(phase_val, 0)
                end_f = pass2.phase_end_frames.get(phase_val, 0)
                if end_f > start_f and pass1.timestamps is not None and len(pass1.timestamps) > 0:
                    if end_f < len(pass1.timestamps):
                        return float(pass1.timestamps[end_f] - pass1.timestamps[start_f])
                return 0.0

            toward_time = _phase_duration(Phase.TOWARD)
            away_time = _phase_duration(Phase.AWAY)
            turning_time = pass2.turning_time_sec
            total_trial_time = float(pass1.timestamps[-1] - pass1.timestamps[0]) if len(pass1.timestamps) > 1 else 0.0

            # Overall cadence: steps per minute over total walk time (excluding turn)
            walk_time = toward_time + away_time
            overall_cadence = (total_steps / walk_time * 60.0) if walk_time > 0 else 0.0

            global_metrics = GlobalMetrics(
                total_steps=total_steps,
                overall_cadence_steps_per_min=overall_cadence,
                overall_asymmetry_score=0.0,    # Phase 2
                overall_sway_rms_meters=0.0,    # Phase 2
                toward_6m_time_sec=toward_time,
                away_6m_time_sec=away_time,
                turning_time_sec=turning_time,
                total_trial_time_sec=total_trial_time,
                stride_length_cv_percent=0.0,   # Phase 2
                step_time_cv_percent=0.0,       # Phase 2
            )

            # Convert string-keyed quartile_metrics (from Pass2Result) → Quartile enum keys
            quartile_metrics_enum: dict[Quartile, QuartileMetrics] = {
                Quartile(k): v for k, v in pass2.quartile_metrics.items()
            }

            # Phase 2 stub: empty ClinicalReport
            clinical_report = ClinicalReport(flags=[], summary="", recommendations=[])

            metrics = GaitMetrics(
                quartile_metrics=quartile_metrics_enum,
                global_metrics=global_metrics,
                clinical_report=clinical_report,
            )

            result = AnalysisResult(
                metadata=metadata,
                metrics=metrics,
            )

            report_progress(1.00, "Complete")

            if self.config.verbose:
                print(f"Processing complete in {elapsed_sec:.1f}s ({elapsed_sec/pass1.total_frames*pass1.fps:.1f}x real-time)")

            return result

        except Exception as e:
            if self.config.verbose:
                print(f"Error during processing: {e}")
            raise
