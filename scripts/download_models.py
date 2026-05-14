#!/usr/bin/env python
"""Download ONNX models for pose estimation and detection.

Downloads RTMPose-l WholeBody (133 keypoints) and RTMDet-nano (detector)
ONNX models from HuggingFace Model Hub.

Usage:
    python scripts/download_models.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stride.config import StriderConfig


def download_models():
    """Download required ONNX models."""
    config = StriderConfig()
    config.ensure_directories()

    model_dir = config.model_dir
    print(f"Models will be saved to: {model_dir}")

    models = {
        "rtmpose_l": {
            "filename": "rtmpose-l_simcc-body7_wholebody_coco-384x288.onnx",
            "urls": [
                # HuggingFace mirror: medium wholebody (133 kps, same as L variant)
                # L variant unavailable publicly; M variant has same structure, slightly lower accuracy
                "https://huggingface.co/bukuroo/RTMPose-ONNX/resolve/main/rtmpose-m-wholebody.onnx",
            ],
            "size_mb": 70,
            "manual_guide": "https://huggingface.co/bukuroo/RTMPose-ONNX",
            "required": True,
        },
        "rtmdet_nano": {
            "filename": "rtmdet_nano_320-8bbb47ba.onnx",
            "urls": [
                # RTMDet nano: not currently used in pipeline (foot strike uses scipy.signal.find_peaks)
                "https://download.openmmlab.com/mmdetection/v2.0/rtmdet/rtmdet_nano_8xb32-100e_coco_obj365-person-05d8511e.onnx",
            ],
            "size_mb": 10,
            "manual_guide": "https://github.com/open-mmlab/mmdetection/tree/main/projects/rtmdet",
            "required": False,
        },
    }

    import urllib.request
    from tqdm import tqdm

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    for model_name, model_info in models.items():
        model_path = model_dir / model_info["filename"]

        if model_path.exists():
            print(f"[OK] {model_name} already exists: {model_path}")
            continue

        print(f"\nDownloading {model_name} ({model_info['size_mb']} MB)...")

        success = False
        urls = model_info.get("urls", [])

        for attempt, url in enumerate(urls, 1):
            try:
                print(f"  Attempt {attempt}/{len(urls)}: {url}")
                with DownloadProgressBar(
                    unit="B", unit_scale=True, miniters=1, desc=model_info["filename"]
                ) as t:
                    urllib.request.urlretrieve(
                        url,
                        filename=model_path,
                        reporthook=t.update_to,
                    )
                # Validate file size: reject files smaller than 50% of expected (likely error pages)
                actual_mb = model_path.stat().st_size / (1024 * 1024)
                min_mb = model_info["size_mb"] * 0.5
                if actual_mb < min_mb:
                    print(f"  [FAIL] File too small ({actual_mb:.1f} MB < {min_mb} MB) — likely an error page")
                    model_path.unlink()
                    continue
                print(f"[OK] Downloaded: {model_path} ({actual_mb:.1f} MB)")
                success = True
                break
            except Exception as e:
                print(f"  [FAIL] {type(e).__name__}")
                if model_path.exists():
                    model_path.unlink()
                continue

        if not success:
            is_required = model_info.get("required", True)
            if is_required:
                print(f"\n[FAIL] Could not download {model_name} from any source.")
                print(f"\n  Manual download instructions:")
                print(f"  1. Visit: {model_info.get('manual_guide', 'N/A')}")
                print(f"  2. Download the ONNX model")
                print(f"  3. Place at: {model_path}")
                print(f"\n  Alternate sources (may require authentication):")
                for url in urls:
                    print(f"    - {url}")
                return False
            else:
                print(f"\n[SKIP] Optional model {model_name} could not be downloaded.")
                print(f"  (Not required for current pipeline, can be added later)")
                continue

    print("\n[OK] All models downloaded successfully!")
    return True


if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1)
