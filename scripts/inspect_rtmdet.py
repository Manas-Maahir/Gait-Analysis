#!/usr/bin/env python
"""Inspect RTMDet-nano ONNX model output format.

Loads the model and runs inference on the first frame of tests/Healthy.mp4,
then prints raw output shapes and values to diagnose the format.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import onnxruntime as ort


def main():
    model_path = Path("models/rtmdet-n-person.onnx")
    if not model_path.exists():
        print(f"Error: {model_path} not found")
        return

    print(f"[*] Loading ONNX model: {model_path}")
    sess = ort.InferenceSession(str(model_path))

    print("\n[*] Model Inputs:")
    for i in sess.get_inputs():
        print(f"  {i.name}: shape={i.shape}, type={i.type}")

    print("\n[*] Model Outputs:")
    for o in sess.get_outputs():
        print(f"  {o.name}: shape={o.shape}, type={o.type}")

    # Load first frame of test video
    video_path = Path("tests/Healthy.mp4")
    if not video_path.exists():
        print(f"Error: {video_path} not found")
        return

    print(f"\n[*] Reading first frame from: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame")
        return

    print(f"  Frame shape: {frame.shape}")

    # Letterbox to 320x320 (same as RTMDetDetector.detect)
    h, w = frame.shape[:2]
    scale = min(320 / w, 320 / h)
    nw, nh = int(w * scale), int(h * scale)
    px, py = (320 - nw) // 2, (320 - nh) // 2

    print(f"  Letterbox: scale={scale:.4f}, new_size={nw}x{nh}, pad=({px},{py})")

    padded = np.full((320, 320, 3), 114, dtype=np.uint8)
    padded[py : py + nh, px : px + nw] = cv2.resize(frame, (nw, nh))
    inp = np.transpose(padded.astype(np.float32) / 255.0, (2, 0, 1))[None]

    print(f"  Input to ONNX: shape={inp.shape}, dtype={inp.dtype}, range=[{inp.min():.4f}, {inp.max():.4f}]")

    # Run inference
    print("\n[*] Running inference...")
    outputs = sess.run(None, {sess.get_inputs()[0].name: inp})

    print(f"\n[*] Raw ONNX Outputs: {len(outputs)} output(s)")
    for i, out in enumerate(outputs):
        print(f"\n  Output[{i}]:")
        print(f"    shape={out.shape}, dtype={out.dtype}")
        print(f"    range=[{out.min():.6f}, {out.max():.6f}]")

        if out.ndim == 3:
            sq = out.squeeze(0)
            print(f"    squeezed shape: {sq.shape}")
            print(f"    first 3 rows (transposed for readability):")
            for row_idx in range(min(3, len(sq))):
                print(f"      row {row_idx}: {sq[row_idx]}")
        elif out.ndim == 2:
            print(f"    first 3 rows:")
            for row_idx in range(min(3, len(out))):
                print(f"      row {row_idx}: {out[row_idx]}")
        elif out.ndim == 1:
            print(f"    first 10 values: {out[:10]}")
        else:
            print(f"    value: {out}")

    print("\n[*] Diagnosis:")
    if len(outputs) == 1:
        dets = outputs[0]
        if dets.ndim == 3:
            dets = dets.squeeze(0)
        if dets.ndim == 2:
            n_cols = dets.shape[1]
            print(f"  Single output with {n_cols} columns per detection")
            if n_cols >= 6:
                print(f"    → Likely [x1, y1, x2, y2, conf, class_id]")
                print(f"       Parsing: conf=col[4], class=col[5]")
            elif n_cols == 5:
                print(f"    → Likely [x1, y1, x2, y2, conf]")
                print(f"       Parsing: conf=col[4]")
            elif n_cols == 4:
                print(f"    → Likely [x1, y1, x2, y2] (scores separate)")
    elif len(outputs) == 2:
        print(f"  Two outputs detected")
        print(f"    Output[0]: {outputs[0].shape}")
        print(f"    Output[1]: {outputs[1].shape}")
        print(f"    → Likely [bboxes, scores] format")
    else:
        print(f"  {len(outputs)} outputs (unexpected format)")

    print("\n[+] Done.")


if __name__ == "__main__":
    main()
