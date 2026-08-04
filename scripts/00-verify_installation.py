#!/usr/bin/env python3

"""
Installation verification script for Go1 Challenge.

This script verifies that all required dependencies and configurations are properly installed
for running the Go1 locomotion training and playing scripts.

Usage:
   python scripts/00-verify_installation.py
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path
import torch


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_status(item, status, details=""):
    """Print status with formatting."""
    status_symbol = "✓" if status else "✗"
    status_text = "PASS" if status else "FAIL"
    print(f"{status_symbol} {item:<40} [{status_text}] {details}")


def check_python_version():
    """Check if Python version is compatible."""
    print_header("PYTHON VERSION CHECK")

    version = sys.version_info
    required_major, required_minor = 3, 10

    compatible = version.major >= required_major and version.minor >= required_minor
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    print_status("Python Version", compatible, f"Current: {version_str}, Required: >={required_major}.{required_minor}")

    return compatible


def check_required_packages():
    """Check if all required Python packages are installed."""
    print_header("PYTHON PACKAGES CHECK")

    required_packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("pyapriltags", "PyAprilTags"),
        ("isaaclab", "Isaac Lab"),
        ("omni", "Omniverse"),
        ("isaaclab.app", "Isaac Lab App"),
        ("rsl_rl", "RSL-RL"),
        ("go1_challenge", "Go1 Challenge"),
    ]

    all_passed = True

    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print_status(name, True)
        except ImportError as e:
            print_status(name, False, f"Import error: {e}")
            all_passed = False

    return all_passed


def check_pytorch_cuda():
    """Check PyTorch CUDA availability."""
    print_header("PYTORCH CUDA CHECK")

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    print_status("CUDA Available", cuda_available)
    if cuda_available:
        print_status("CUDA Device Count", device_count > 0, f"Devices: {device_count}")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print_status(f"  GPU {i}", True, device_name)

    return cuda_available


def check_environment_variables():
    """Check required environment variables."""
    print_header("ENVIRONMENT VARIABLES CHECK")

    required_vars = [
        ("ISAACLAB_PATH", "Isaac Lab installation path"),
        ("ISAACSIM_PATH", "Isaac Sim installation path"),
    ]

    all_set = True

    for var_name, description in required_vars:
        value = os.environ.get(var_name)
        is_set = value is not None and value != ""
        print_status(description, is_set, f"{var_name}={value}" if is_set else f"{var_name} not set")
        if not is_set:
            all_set = False

    return all_set


def main():
    """Main verification function."""
    print("Go1 Challenge Installation Verification")
    print("This script checks if all required dependencies are properly installed.")

    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("PyTorch CUDA", check_pytorch_cuda),
    ]

    results = []

    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ Error running {check_name}: {e}")
            results.append((check_name, False))

    # Summary
    print_header("VERIFICATION SUMMARY")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        print_status(check_name, result)

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 ALL CHECKS PASSED!")
        print("You are set for the competition!")
        return True
    else:
        print(f"\n❌ {total - passed} checks failed!")
        print("Please fix the failing checks before proceeding.")
        print("\nTroubleshooting tips:")
        print("1. Make sure Isaac Lab is properly installed")
        print("2. Install missing Python packages: pip install <package_name>")
        print("3. Check that ISAACLAB_PATH and ISAACSIM_PATH are set correctly")
        return False


if __name__ == "__main__":
    success = main()
