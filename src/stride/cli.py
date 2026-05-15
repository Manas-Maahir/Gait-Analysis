"""Command-line interface for STRIDE gait analysis.

Usage:
    python -m stride analyze --video input.mp4 --output results.json [--preset default]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="stride",
        description="STRIDE clinical gait analysis — 6-meter walk test",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── analyze subcommand ────────────────────────────────────────────────
    analyze = sub.add_parser("analyze", help="Analyze a gait video")
    analyze.add_argument(
        "--video", "-i", required=True, type=Path,
        help="Path to input video file",
    )
    analyze.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output JSON path (default: <video_stem>_result.json beside video)",
    )
    analyze.add_argument(
        "--preset", choices=["default", "pathological", "gpu"], default="default",
        help="Configuration preset to use",
    )
    analyze.add_argument(
        "--path-length", type=float, default=6.0,
        help="Walking path length in meters (default: 6.0)",
    )
    analyze.add_argument(
        "--calibration", choices=["auto", "manual", "auto_with_fallback"],
        default="auto_with_fallback",
        help="Calibration mode (default: auto_with_fallback)",
    )
    analyze.add_argument(
        "--model-dir", type=Path, default=Path("models"),
        help="Directory containing ONNX model weights",
    )
    analyze.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="Inference device (default: cpu)",
    )
    analyze.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging (also triggers --pose-debug auto output)",
    )
    analyze.add_argument(
        "--no-progress", action="store_true",
        help="Disable progress bar output",
    )
    analyze.add_argument(
        "--debug-dir", type=Path, default=None,
        help="Save RTMPose keypoints, world positions, and phase labels as .npy files here",
    )
    analyze.add_argument(
        "--pose-debug", type=Path, default=None, metavar="PATH",
        nargs="?", const=Path("__auto__"),
        help=(
            "Write RTMPose skeleton-overlay debug MP4 after Pass 1.  "
            "If PATH is omitted, saves to rtmpose_output/<video_stem>_pose_debug.mp4.  "
            "-v also enables this with the auto path."
        ),
    )

    return parser


def _make_progress_callback(verbose: bool, no_progress: bool):
    """Return a progress callback that prints to stderr."""
    if no_progress:
        return None

    last_pct = [-1.0]

    def callback(pct: float, stage: str) -> None:
        pct_int = int(pct * 100)
        if pct_int != last_pct[0] or verbose:
            bar_len = 40
            filled = int(bar_len * pct)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"\r[{bar}] {pct_int:3d}%  {stage}", end="", file=sys.stderr, flush=True)
            if pct >= 1.0:
                print(file=sys.stderr)
        last_pct[0] = pct_int

    return callback


def cmd_analyze(args: argparse.Namespace) -> int:
    """Execute the analyze command. Returns exit code."""
    from .config.presets import get_default_config, get_pathological_gait_config, get_gpu_config
    from .config.schema import StriderConfig
    from .pipeline.processor import GaitProcessor
    from .pose.rtmpose import RTMPoseEstimator
    from .tracking.bytetrack import ByteTrack

    # Build config from preset + overrides
    if args.preset == "pathological":
        config = get_pathological_gait_config()
    elif args.preset == "gpu":
        config = get_gpu_config()
    else:
        config = get_default_config()

    # Apply CLI overrides
    config = config.model_copy(update={
        "model_dir": args.model_dir,
        "device": args.device,
        "path_length_meters": args.path_length,
        "calibration_mode": args.calibration,
        "verbose": args.verbose,
    })

    # Determine output path
    video_path = args.video
    if args.output is None:
        output_path = video_path.parent / f"{video_path.stem}_result.json"
    else:
        output_path = args.output

    if not video_path.exists():
        print(f"Error: video file not found: {video_path}", file=sys.stderr)
        return 1

    # Ensure model directory exists (side effect explicit)
    config.ensure_directories()

    # Instantiate components
    rtmpose_model_path = config.model_dir / "rtmpose-l_simcc-body7_wholebody_coco-384x288.onnx"
    if not rtmpose_model_path.exists():
        print(
            f"Error: RTMPose model not found at {rtmpose_model_path}\n"
            f"Run: python scripts/download_models.py",
            file=sys.stderr,
        )
        return 1

    try:
        pose_estimator = RTMPoseEstimator(
            model_path=rtmpose_model_path,
            device=config.device,
            input_size=config.rtmpose_input_size,
        )
    except Exception as e:
        print(f"Error loading RTMPose model: {e}", file=sys.stderr)
        return 1

    tracker = ByteTrack(
        max_age=config.tracking_lost_threshold_frames,
        confidence_threshold=config.detection_score_threshold,
    )

    # Person detector (optional, improves accuracy significantly)
    detector = None
    rtmdet_model_path = getattr(args, "rtmdet_model", None)
    if rtmdet_model_path is None:
        _candidate = config.model_dir / "rtmdet-n-person.onnx"
        if _candidate.exists():
            rtmdet_model_path = _candidate

    if rtmdet_model_path is not None and Path(rtmdet_model_path).exists():
        try:
            from .detection.rtmdet import RTMDetDetector
            detector = RTMDetDetector(model_path=rtmdet_model_path, device=config.device, conf_threshold=0.15)
            if args.verbose:
                print(f"Person detector: {rtmdet_model_path} (conf_threshold=0.15)")
        except Exception as e:
            print(f"Warning: RTMDet load failed ({e}), using full-frame fallback", file=sys.stderr)
    else:
        if args.verbose:
            print("Warning: No RTMDet model found. Using full-frame fallback (reduced accuracy).")

    # Resolve pose debug output path:
    # --pose-debug PATH  → use PATH
    # --pose-debug       → auto path (const "__auto__")
    # -v with no flag    → also auto path
    _SENTINEL = Path("__auto__")
    pose_debug_path: Optional[Path] = None
    raw_pd = getattr(args, "pose_debug", None)
    if raw_pd is not None and raw_pd != _SENTINEL:
        pose_debug_path = raw_pd
    elif raw_pd == _SENTINEL or args.verbose:
        pose_debug_path = (
            Path("rtmpose_output") / f"{video_path.stem}_pose_debug.mp4"
        )

    processor = GaitProcessor(
        config=config,
        pose_estimator=pose_estimator,
        tracker=tracker,
        detector=detector,
        debug_dir=args.debug_dir,
        pose_debug_path=pose_debug_path,
    )

    progress_callback = _make_progress_callback(
        verbose=args.verbose, no_progress=args.no_progress
    )

    if args.verbose:
        print(f"Analyzing: {video_path}")
        print(f"Output:    {output_path}")
        print(f"Config:    {config.config_hash}")

    t0 = time.time()
    try:
        result = processor.process(str(video_path), progress_callback=progress_callback)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Processing failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    elapsed = time.time() - t0

    # Save result
    result.to_json(output_path)

    # Print summary
    gm = result.metrics.global_metrics
    print(f"\nAnalysis complete in {elapsed:.1f}s")
    print(f"  Total steps:  {gm.total_steps}")
    print(f"  Cadence:      {gm.overall_cadence_steps_per_min:.1f} steps/min")
    print(f"  Toward time:  {gm.toward_6m_time_sec:.2f}s")
    print(f"  Away time:    {gm.away_6m_time_sec:.2f}s")
    print(f"  Turning time: {gm.turning_time_sec:.2f}s")
    print(f"  Results:      {output_path}")
    if pose_debug_path is not None:
        print(f"  Pose debug:   {pose_debug_path}")

    return 0


def main(argv=None) -> int:
    """Main entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        return cmd_analyze(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
