"""
Healthy Beans - Phase 1
Environment Verification

This script verifies that the core libraries required for
the Healthy Beans machine-learning project are installed
and working correctly.

Phase 1 only:
- Does NOT modify the dataset
- Does NOT train a model
- Does NOT preprocess images
"""

import sys


def check_library(name, import_name=None):
    """Check whether a Python library can be imported."""
    if import_name is None:
        import_name = name

    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "version unavailable")
        print(f"[OK] {name:<15} {version}")
        return True
    except Exception as error:
        print(f"[FAIL] {name:<15} {error}")
        return False


def test_pytorch():
    """Test PyTorch and create a small tensor."""
    try:
        import torch

        print(f"[OK] PyTorch         {torch.__version__}")

        # Basic PyTorch test
        tensor = torch.tensor([[1, 2], [3, 4]])

        print("[OK] PyTorch tensor test passed")
        print(f"     Tensor: {tensor.tolist()}")

        # Check available hardware
        if torch.cuda.is_available():
            print("[OK] CUDA/GPU        Available")
            print(f"     GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[INFO] CUDA/GPU       Not available")
            print("       Training will use the CPU.")

        return True

    except Exception as error:
        print(f"[FAIL] PyTorch test   {error}")
        return False


def main():
    print("=" * 60)
    print("          HEALTHY BEANS ENVIRONMENT CHECK")
    print("=" * 60)

    print("\nPython")
    print("-" * 60)
    print(f"Version: {sys.version}")
    print(f"Executable: {sys.executable}")

    print("\nCore Libraries")
    print("-" * 60)

    results = []

    results.append(check_library("NumPy", "numpy"))
    results.append(check_library("Pandas", "pandas"))
    results.append(check_library("OpenCV", "cv2"))
    results.append(check_library("Pillow", "PIL"))
    results.append(check_library("Matplotlib", "matplotlib"))
    results.append(check_library("Scikit-learn", "sklearn"))

    print("\nDeep Learning Framework")
    print("-" * 60)

    results.append(test_pytorch())

    print("\nTorchvision")
    print("-" * 60)

    results.append(check_library("Torchvision", "torchvision"))

    print("\n" + "=" * 60)

    if all(results):
        print("ENVIRONMENT STATUS: READY")
        print("=" * 60)
        print("\nHealthy Beans Phase 1 completed successfully.")
        print("Environment ready for Phase 2: Dataset Audit.")

    else:
        print("ENVIRONMENT STATUS: ATTENTION REQUIRED")
        print("=" * 60)
        print("\nOne or more components failed.")
        print("Fix the failed components before moving to Phase 2.")


if __name__ == "__main__":
    main()