#!/usr/bin/env python3
"""Stage 2 dependency and F-drive environment verification.

This script intentionally does not load a model, access Hugging Face, or run
any project inference. It checks imports, versions, CUDA tensor operations,
ONNX Runtime providers, and cache/environment paths only.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import pathlib
import sys
import traceback


CACHE_VARS = (
    "XDG_CACHE_HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    "PIP_CACHE_DIR",
    "CONDA_PKGS_DIRS",
    "CONDA_ENVS_PATH",
    "GRADIO_TEMP_DIR",
    "TMPDIR",
)

PACKAGE_NAMES = (
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "peft",
    "insightface",
    "onnxruntime-gpu",
    "numpy",
    "scipy",
    "opencv-python",
    "Pillow",
    "gradio",
    "tqdm",
)


def check_cache_paths(errors: list[str]) -> None:
    print("[cache paths]")
    for name in CACHE_VARS:
        value = os.environ.get(name, "")
        print(f"{name}={value}")
        if not value:
            errors.append(f"{name} is not set")
        elif not pathlib.Path(value).as_posix().startswith("/mnt/f/ICVIP"):
            errors.append(f"{name} is outside /mnt/f/ICVIP: {value}")
        elif not pathlib.Path(value).exists():
            errors.append(f"{name} path does not exist: {value}")

    executable = pathlib.Path(sys.executable)
    prefix = pathlib.Path(sys.prefix)
    virtual_env = os.environ.get("VIRTUAL_ENV", "")
    print(f"python_executable={executable}")
    print(f"python_prefix={prefix}")
    print(f"VIRTUAL_ENV={virtual_env}")
    # venv/bin/python is commonly a symlink to the system interpreter. The
    # environment location is therefore checked via sys.prefix/VIRTUAL_ENV,
    # not by resolving the interpreter symlink.
    if not prefix.as_posix().startswith("/mnt/f/ICVIP/"):
        errors.append(f"Python environment prefix is outside F drive: {prefix}")
    if virtual_env and not pathlib.Path(virtual_env).as_posix().startswith("/mnt/f/ICVIP/"):
        errors.append(f"VIRTUAL_ENV is outside F drive: {virtual_env}")


def print_versions(errors: list[str]) -> None:
    print("[package versions]")
    for name in PACKAGE_NAMES:
        try:
            print(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing package metadata: {name}")


def check_imports(errors: list[str]) -> None:
    print("[imports]")
    for module_name in (
        "diffusers",
        "transformers",
        "accelerate",
        "peft",
        "insightface",
        "numpy",
        "scipy",
        "cv2",
        "PIL",
        "gradio",
    ):
        try:
            module = importlib.import_module(module_name)
            print(f"{module_name}=PASS ({getattr(module, '__version__', 'imported')})")
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"import {module_name} failed: {exc}")
            print(f"{module_name}=FAIL: {exc}")


def check_torch(errors: list[str]) -> None:
    print("[pytorch and cuda]")
    try:
        import torch

        print(f"torch_version={torch.__version__}")
        print(f"torch_cuda_build={torch.version.cuda}")
        available = bool(torch.cuda.is_available())
        print(f"cuda_available={available}")
        if not available:
            errors.append("torch.cuda.is_available() is False")
            return
        print(f"gpu_name={torch.cuda.get_device_name(0)}")
        x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        result = float((x * x).sum().item())
        print(f"cuda_tensor_sum_squares={result}")
        if result != 14.0:
            errors.append(f"unexpected CUDA tensor result: {result}")
        block = torch.empty((1024, 1024), dtype=torch.float32, device="cuda")
        print(f"memory_allocated_before_release={torch.cuda.memory_allocated(0)}")
        print(f"memory_reserved_before_release={torch.cuda.memory_reserved(0)}")
        del block, x
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"memory_allocated_after_release={torch.cuda.memory_allocated(0)}")
        print(f"memory_reserved_after_release={torch.cuda.memory_reserved(0)}")
    except Exception as exc:  # pragma: no cover - diagnostic path
        errors.append(f"PyTorch/CUDA check failed: {exc}")
        traceback.print_exc()


def check_onnxruntime(errors: list[str]) -> None:
    print("[onnxruntime providers]")
    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
        print(f"providers={providers}")
        for required in ("CPUExecutionProvider", "CUDAExecutionProvider"):
            if required not in providers:
                errors.append(f"ONNX Runtime provider unavailable: {required}")
    except Exception as exc:  # pragma: no cover - diagnostic path
        errors.append(f"ONNX Runtime check failed: {exc}")
        traceback.print_exc()


def main() -> int:
    errors: list[str] = []
    print("STAGE2_ENVIRONMENT_CHECK=START")
    print(f"python_version={sys.version}")
    check_cache_paths(errors)
    print_versions(errors)
    check_imports(errors)
    check_torch(errors)
    check_onnxruntime(errors)
    print("[result]")
    if errors:
        print("STAGE2_ENVIRONMENT_CHECK=FAIL")
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("STAGE2_ENVIRONMENT_CHECK=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
