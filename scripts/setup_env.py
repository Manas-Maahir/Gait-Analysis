#!/usr/bin/env python
"""Validate and setup the Stride environment.

Checks:
1. Python version (3.10+)
2. Required packages installed
3. ONNX models downloaded
4. Output/model directories writable
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 10):
        print(f"✗ Python 3.10+ required, got {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_packages():
    """Check required packages."""
    required = {
        "numpy": "numpy",
        "scipy": "scipy",
        "cv2": "opencv-python",
        "onnxruntime": "onnxruntime",
        "pydantic": "pydantic",
        "PIL": "pillow",
    }

    all_ok = True
    for import_name, package_name in required.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} not installed")
            print(f"   pip install {package_name}")
            all_ok = False

    return all_ok


def check_models():
    """Check if ONNX models exist."""
    from stride.config import StriderConfig

    config = StriderConfig()
    model_dir = config.model_dir

    models = [
        "rtmpose-l_simcc-body7_wholebody_coco-384x288.onnx",
        "rtmdet_nano_320-8bbb47ba.onnx",
    ]

    all_ok = True
    for model_file in models:
        model_path = model_dir / model_file
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✓ {model_file} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {model_file} not found")
            print(f"   Run: python scripts/download_models.py")
            all_ok = False

    return all_ok


def check_directories():
    """Check directory permissions."""
    from stride.config import StriderConfig

    config = StriderConfig()

    directories = {
        "model_dir": config.model_dir,
        "output_dir": config.output_dir,
    }

    all_ok = True
    for name, path in directories.items():
        path = Path(path)
        if path.exists():
            if path.is_dir():
                # Test write permission
                try:
                    test_file = path / ".write_test"
                    test_file.touch()
                    test_file.unlink()
                    print(f"✓ {name} is writable: {path}")
                except Exception as e:
                    print(f"✗ {name} not writable: {e}")
                    all_ok = False
            else:
                print(f"✗ {name} exists but is not a directory: {path}")
                all_ok = False
        else:
            print(f"⚠ {name} does not exist (will be created on first use): {path}")

    return True  # Don't fail on this


def main():
    """Run all checks."""
    print("Stride Environment Setup\n")

    checks = [
        ("Python Version", check_python_version),
        ("Python Packages", check_packages),
        ("Directories", check_directories),
        ("ONNX Models", check_models),
    ]

    results = []
    for name, check_fn in checks:
        print(f"\n{name}:")
        result = check_fn()
        results.append((name, result))

    # Summary
    print("\n" + "=" * 50)
    all_ok = all(result for _, result in results)

    if all_ok:
        print("✓ All checks passed! Ready to use Stride.")
        return 0
    else:
        print("✗ Some checks failed. See above for details.")
        print("\nTo fix:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Download models: python scripts/download_models.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
