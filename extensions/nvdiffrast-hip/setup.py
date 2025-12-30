# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# HIP/ROCm port additions for AMD GPU support.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import setuptools
import os
import re
import subprocess
import platform
import sys

import torch

# Fix PyTorch bug with HIP paths containing spaces on Windows
# See: torch/utils/cpp_extension.py line 1032
# The bug: spaces in -I paths are replaced with backslashes (wrong!)
# E.g., "C:\Users\Olivier Chafik" becomes "C:\Users\Olivier\Chafik"
# The fix: patch _nt_quote_args to detect and undo the mangling for -I paths
if platform.system() == 'Windows' and hasattr(torch.version, 'hip') and torch.version.hip:
    import ctypes
    from ctypes import wintypes
    import torch.utils.cpp_extension as cpp_ext

    def _get_short_path_win(path):
        """Convert path to Windows 8.3 short format."""
        if not path or not os.path.exists(path):
            return path
        try:
            GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
            GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
            GetShortPathNameW.restype = wintypes.DWORD
            buf_size = GetShortPathNameW(path, None, 0)
            if buf_size == 0:
                return path
            buf = ctypes.create_unicode_buffer(buf_size)
            GetShortPathNameW(path, buf, buf_size)
            return buf.value
        except Exception:
            return path

    def _try_unmangle_path(mangled_path):
        """Try to recover a path that had spaces replaced with backslashes.

        E.g., 'C:\\Users\\Olivier\\Chafik\\AppData' -> 'C:\\Users\\Olivier Chafik\\AppData'

        Strategy: find longest existing prefix, then try joining remaining parts with spaces.
        """
        if os.path.exists(mangled_path):
            return mangled_path

        parts = mangled_path.split('\\')

        # Find the longest existing prefix path
        for i in range(len(parts), 0, -1):
            prefix = '\\'.join(parts[:i])
            if os.path.exists(prefix):
                remaining = parts[i:]
                if not remaining:
                    return prefix

                # Try combining remaining parts with spaces (recursive)
                def try_combine(rem_parts, current_path):
                    if not rem_parts:
                        return current_path if os.path.exists(current_path) else None

                    # Try with backslash first (normal path separator)
                    with_slash = current_path + '\\' + rem_parts[0]
                    result = try_combine(rem_parts[1:], with_slash)
                    if result:
                        return result

                    # Try with space (undo the mangle)
                    with_space = current_path + ' ' + rem_parts[0]
                    result = try_combine(rem_parts[1:], with_space)
                    if result:
                        return result

                    return None

                result = try_combine(remaining, prefix)
                if result and os.path.exists(result):
                    return result

        return mangled_path  # Give up, return original

    def _fix_include_arg(arg):
        """Fix a -I argument that may have been mangled by PyTorch line 1032."""
        if not arg.startswith('-I'):
            return arg

        path = arg[2:]

        # Path exists as-is - just convert to short path if needed
        if os.path.exists(path):
            if ' ' in path:
                return '-I' + _get_short_path_win(path)
            return arg

        # Path doesn't exist - try to unmangle it
        fixed_path = _try_unmangle_path(path)
        if fixed_path != path and os.path.exists(fixed_path):
            short_path = _get_short_path_win(fixed_path)
            print(f"[nvdiffrast-hip] Fixed: -I{path} -> -I{short_path}")
            return '-I' + short_path

        return arg  # Couldn't fix, return as-is

    # Patch _nt_quote_args to fix mangled -I paths before quoting
    _orig_nt_quote_args = cpp_ext._nt_quote_args

    def _patched_nt_quote_args(args):
        if not args:
            return []
        fixed_args = [_fix_include_arg(arg) if arg.startswith('-I') else arg for arg in args]
        return _orig_nt_quote_args(fixed_args)

    cpp_ext._nt_quote_args = _patched_nt_quote_args
    print("[nvdiffrast-hip] Patched _nt_quote_args to fix HIP include paths")

    # Fix PyTorch looking for hipcc in 'bin' instead of 'Scripts' on Windows
    _orig_get_hipcc_path = cpp_ext._get_hipcc_path
    def _patched_get_hipcc_path():
        path = _orig_get_hipcc_path()
        # Replace \bin\ with \Scripts\ on Windows
        if '\\bin\\' in path:
            fixed = path.replace('\\bin\\', '\\Scripts\\')
            if os.path.exists(fixed):
                print(f"[nvdiffrast-hip] Fixed hipcc path: {path} -> {fixed}")
                return fixed
        return path
    cpp_ext._get_hipcc_path = _patched_get_hipcc_path

# Detect if we're building for HIP/ROCm or CUDA
IS_HIP = hasattr(torch.version, 'hip') and torch.version.hip is not None
IS_WINDOWS = platform.system() == 'Windows'

def detect_gpu_arch():
    """Detect AMD GPU architecture from PyTorch, rocminfo, or environment."""
    # Check environment variable first
    gpu_arch = os.environ.get('GPU_ARCH', os.environ.get('PYTORCH_ROCM_ARCH', ''))
    if gpu_arch:
        return gpu_arch

    # Try to detect from PyTorch (most reliable on Windows)
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if hasattr(props, 'gcnArchName') and props.gcnArchName:
                print(f"[nvdiffrast-hip] Detected GPU arch from PyTorch: {props.gcnArchName}")
                return props.gcnArchName
    except Exception as e:
        print(f"[nvdiffrast-hip] PyTorch GPU detection failed: {e}")

    # Try to detect from rocminfo
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'gfx' in line.lower():
                    match = re.search(r'gfx\d+[a-z0-9]*', line.lower())
                    if match:
                        return match.group(0)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Default fallback
    print("[nvdiffrast-hip] Could not detect GPU arch, defaulting to gfx1100")
    return "gfx1100"

# Print build configuration
if IS_HIP:
    gpu_arch = detect_gpu_arch()
    print("\n" + "=" * 70)
    print("Building nvdiffrast for AMD ROCm/HIP")
    print(f"PyTorch: {torch.__version__}, HIP: {torch.version.hip}")
    print(f"GPU Architecture: {gpu_arch}")
    print(f"Platform: {'Windows' if IS_WINDOWS else 'Linux'}")
    print("=" * 70 + "\n")
else:
    print("\n" + "=" * 70)
    print("Building nvdiffrast for NVIDIA CUDA")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print("=" * 70 + "\n")

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    print("\n\n" + "*" * 70)
    print("ERROR! Cannot compile nvdiffrast extension. Please ensure that:\n")
    print("1. You have PyTorch installed")
    print("2. You run 'pip install' with --no-build-isolation flag")
    print("*" * 70 + "\n\n")
    exit(1)

# Source files for HIP
if IS_HIP:
    # Rename _hip.cpp files to .hip so they're compiled with hipcc instead of MSVC
    # This must be done after PyTorch's hipify runs
    import glob as glob_module
    import shutil

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Files that need HIP compilation (use HIP APIs, device guards, streams)
    hip_cpp_files = [
        "csrc/torch/torch_antialias_hip.cpp",
        "csrc/torch/torch_interpolate_hip.cpp",
        "csrc/torch/torch_rasterize_hip.cpp",
        "csrc/torch/torch_texture_hip.cpp",
        "csrc/common/hipraster/impl/Buffer.cpp",
        "csrc/common/hipraster/impl/CudaRaster.cpp",
        "csrc/common/hipraster/impl/RasterImpl.cpp",
    ]

    for cpp_file in hip_cpp_files:
        cpp_path = os.path.join(base_dir, cpp_file)
        if os.path.exists(cpp_path):
            hip_path = cpp_path.replace('.cpp', '.hip')
            if not os.path.exists(hip_path) or os.path.getmtime(cpp_path) > os.path.getmtime(hip_path):
                shutil.copy2(cpp_path, hip_path)
                print(f"[nvdiffrast-hip] Copied {cpp_file} -> {os.path.basename(hip_path)}")

    sources = [
        # HIP kernels (compiled with hipcc)
        "csrc/common/antialias.cu",
        "csrc/common/interpolate.cu",
        "csrc/common/rasterize.cu",
        "csrc/common/texture_kernel.cu",
        "csrc/common/hipraster/impl/RasterImpl_kernel.hip",
        # HIP impl files (need hipcc for HIP API calls)
        "csrc/common/hipraster/impl/Buffer.hip",
        "csrc/common/hipraster/impl/CudaRaster.hip",
        "csrc/common/hipraster/impl/RasterImpl.hip",
        # HIP torch bindings (need hipcc for device guards/streams)
        "csrc/torch/torch_antialias_hip.hip",
        "csrc/torch/torch_interpolate_hip.hip",
        "csrc/torch/torch_rasterize_hip.hip",
        "csrc/torch/torch_texture_hip.hip",
        # Pure C++ files (compiled with MSVC, no HIP includes)
        "csrc/common/common.cpp",
        "csrc/common/texture.cpp",
        "csrc/torch/torch_bindings.cpp",
    ]
else:
    sources = [
        "csrc/common/antialias.cu",
        "csrc/common/common.cpp",
        "csrc/common/cudaraster/impl/Buffer.cpp",
        "csrc/common/cudaraster/impl/CudaRaster.cpp",
        "csrc/common/cudaraster/impl/RasterImpl.cpp",
        "csrc/common/cudaraster/impl/RasterImpl_kernel.cu",
        "csrc/common/interpolate.cu",
        "csrc/common/rasterize.cu",
        "csrc/common/texture.cpp",
        "csrc/common/texture_kernel.cu",
        "csrc/torch/torch_antialias.cpp",
        "csrc/torch/torch_bindings.cpp",
        "csrc/torch/torch_interpolate.cpp",
        "csrc/torch/torch_rasterize.cpp",
        "csrc/torch/torch_texture.cpp",
    ]

# Compiler flags, include dirs, and library dirs
include_dirs = []
library_dirs = []

if IS_HIP:
    # HIP/ROCm build flags
    cxx_flags = ["-DNVDR_TORCH", "-D__HIP_PLATFORM_AMD__"]
    nvcc_flags = ["-DNVDR_TORCH", "-D__HIP_PLATFORM_AMD__"]

    # Add GPU architecture flag
    if 'gpu_arch' in dir():
        nvcc_flags.append(f"--offload-arch={gpu_arch}")

    # Note: RDNA GPUs (gfx10xx, gfx11xx) use Wave32 by default, which is correct
    # for nvdiffrast kernels that assume 32-thread warps

    # Windows-specific flags
    if IS_WINDOWS:
        cxx_flags.extend(["/wd4067", "/wd4624", "/wd4996"])
        # STRIP_ERROR_MESSAGES makes TORCH_CHECK use std::runtime_error
        # instead of c10::ValueError, avoiding ABI issues between hipcc/clang and MSVC
        cxx_flags.append("-DSTRIP_ERROR_MESSAGES")
        nvcc_flags.append("-DSTRIP_ERROR_MESSAGES")

    # Add csrc directory first for thrust stub headers
    csrc_dir = os.path.join(base_dir, "csrc")
    if os.path.exists(csrc_dir):
        include_dirs.append(csrc_dir)
        print(f"[nvdiffrast-hip] Added csrc include (for thrust stub): {csrc_dir}")

    # Add ROCm SDK include and library paths
    # TheRock installs in _rocm_sdk_core/
    rocm_sdk_base = os.path.join(sys.prefix, "Lib", "site-packages", "_rocm_sdk_core")
    if not os.path.exists(rocm_sdk_base):
        rocm_sdk_base = os.path.join(sys.prefix, "lib", "python3.12", "site-packages", "_rocm_sdk_core")

    if os.path.exists(rocm_sdk_base):
        rocm_sdk_include = os.path.join(rocm_sdk_base, "include")
        rocm_sdk_lib = os.path.join(rocm_sdk_base, "lib")

        if os.path.exists(rocm_sdk_include):
            include_dirs.append(rocm_sdk_include)
            print(f"[nvdiffrast-hip] Added ROCm SDK include: {rocm_sdk_include}")

        if os.path.exists(rocm_sdk_lib):
            library_dirs.append(rocm_sdk_lib)
            print(f"[nvdiffrast-hip] Added ROCm SDK library: {rocm_sdk_lib}")

    extra_compile_args = {
        "cxx": cxx_flags,
        "nvcc": nvcc_flags,
    }
else:
    # CUDA build flags
    extra_compile_args = {
        "cxx": ["-DNVDR_TORCH"]
        + (["/wd4067", "/wd4624", "/wd4996"] if os.name == "nt" else []),
        "nvcc": ["-DNVDR_TORCH", "-lineinfo"],
    }

setuptools.setup(
    ext_modules=[
        CUDAExtension(
            "_nvdiffrast_c",
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
