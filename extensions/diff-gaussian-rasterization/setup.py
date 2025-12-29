#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import re
import sys
import glob
import torch
import subprocess
import platform

# Check if we're running on ROCm/HIP
is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
is_windows = platform.system() == 'Windows'
base_dir = os.path.dirname(os.path.abspath(__file__))

# Fix PyTorch bug with HIP paths containing spaces on Windows
# See: torch/utils/cpp_extension.py line 1032
# The bug: spaces in -I paths are replaced with backslashes (wrong!)
# The fix: patch _nt_quote_args to detect and undo the mangling for -I paths
if is_windows and is_rocm:
    import ctypes
    from ctypes import wintypes
    from torch.utils import cpp_extension as cpp_ext

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
        """Try to recover a path that had spaces replaced with backslashes."""
        if os.path.exists(mangled_path):
            return mangled_path
        parts = mangled_path.split('\\')
        for i in range(len(parts), 0, -1):
            prefix = '\\'.join(parts[:i])
            if os.path.exists(prefix):
                remaining = parts[i:]
                if not remaining:
                    return prefix
                def try_combine(rem_parts, current_path):
                    if not rem_parts:
                        return current_path if os.path.exists(current_path) else None
                    with_slash = current_path + '\\' + rem_parts[0]
                    result = try_combine(rem_parts[1:], with_slash)
                    if result:
                        return result
                    with_space = current_path + ' ' + rem_parts[0]
                    result = try_combine(rem_parts[1:], with_space)
                    if result:
                        return result
                    return None
                result = try_combine(remaining, prefix)
                if result and os.path.exists(result):
                    return result
        return mangled_path

    def _fix_include_arg(arg):
        """Fix a -I argument that may have been mangled."""
        if not arg.startswith('-I'):
            return arg
        path = arg[2:]
        if os.path.exists(path):
            return '-I' + _get_short_path_win(path) if ' ' in path else arg
        fixed_path = _try_unmangle_path(path)
        if fixed_path != path and os.path.exists(fixed_path):
            short_path = _get_short_path_win(fixed_path)
            print(f"[diff-gaussian] Fixed: -I{path} -> -I{short_path}")
            return '-I' + short_path
        return arg

    _orig_nt_quote_args = cpp_ext._nt_quote_args
    def _patched_nt_quote_args(args):
        if not args:
            return []
        fixed_args = [_fix_include_arg(arg) if arg.startswith('-I') else arg for arg in args]
        return _orig_nt_quote_args(fixed_args)
    cpp_ext._nt_quote_args = _patched_nt_quote_args
    print("[diff-gaussian] Patched _nt_quote_args to fix HIP include paths")

    # Fix PyTorch looking for hipcc in 'bin' instead of 'Scripts' on Windows
    _orig_get_hipcc_path = cpp_ext._get_hipcc_path
    def _patched_get_hipcc_path():
        path = _orig_get_hipcc_path()
        if '\\bin\\' in path:
            fixed = path.replace('\\bin\\', '\\Scripts\\')
            if os.path.exists(fixed):
                print(f"[diff-gaussian] Fixed hipcc path: {path} -> {fixed}")
                return fixed
        return path
    cpp_ext._get_hipcc_path = _patched_get_hipcc_path

def detect_gpu_arch():
    """Detect AMD GPU architecture from rocminfo or environment."""
    # Check environment variable first
    gpu_arch = os.environ.get('GPU_ARCH', os.environ.get('PYTORCH_ROCM_ARCH', ''))
    if gpu_arch:
        return gpu_arch

    # Try to detect from rocminfo
    try:
        if is_windows:
            # On Windows with TheRock, try rocm-smi or rocminfo
            result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=10)
        else:
            result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'gfx' in line.lower():
                    match = re.search(r'gfx\d+[a-z0-9]*', line.lower())
                    if match:
                        return match.group(0)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Default fallback - gfx1100 for RDNA3
    print("[HIP] Could not detect GPU arch, defaulting to gfx1100")
    print("[HIP] Set GPU_ARCH or PYTORCH_ROCM_ARCH environment variable to override")
    return "gfx1100"

def fix_hipify_issues(content):
    """Fix issues in hipified files for HIP/ROCm compatibility."""
    
    # Fix malformed kernel launch syntax: << <grid, block >> > -> hipLaunchKernelGGL
    # Actual format from hipify: '<< <grid, block >> > ('
    pattern = r'(\w+(?:<[^>]+>)?)\s*<<\s*<\s*([^,]+),\s*([^>]+)\s*>>\s*>\s*\('
    
    def convert_kernel_launch(match):
        kernel = match.group(1)
        grid = match.group(2).strip()
        block = match.group(3).strip()
        return f'hipLaunchKernelGGL({kernel}, {grid}, {block}, 0, 0, '
    
    # Apply the fix
    fixed = re.sub(pattern, convert_kernel_launch, content)
    
    # Also handle standard <<< >>> syntax
    pattern2 = r'(\w+(?:<[^>]+>)?)\s*<<<\s*([^,]+),\s*([^>]+)>>>\s*\('
    fixed = re.sub(pattern2, convert_kernel_launch, fixed)
    
    # Fix extra 0, 0 args
    fixed = re.sub(r',\s*0,\s*0,\s*0,\s*0,', ', 0, 0,', fixed)
    
    return fixed


# Include and library directories for ROCm
include_dirs = []
library_dirs = []

if is_rocm:
    # Add ROCm SDK paths for Windows TheRock builds
    if is_windows:
        # Add base_dir first for thrust stub headers
        include_dirs.append(base_dir)
        print(f"[diff-gaussian] Added base_dir include (for thrust stub): {base_dir}")

        # Use ROCm 7.10.0 TheRock installation for rocprim, hipcub, thrust
        # These are only added to hipcc, not MSVC (they can conflict with MSVC's standard headers)
        therock_include = r"C:\TheRock\build\include"
        if os.path.exists(therock_include):
            print(f"[diff-gaussian] Using TheRock ROCm 7.10.0 includes: {therock_include}")
        rocprim_include = therock_include
        hipcub_include = therock_include
        rocthrust_include = therock_include

        rocm_sdk_base = os.path.join(sys.prefix, "Lib", "site-packages", "_rocm_sdk_core")
        if not os.path.exists(rocm_sdk_base):
            rocm_sdk_base = os.path.join(sys.prefix, "lib", "python3.12", "site-packages", "_rocm_sdk_core")

        if os.path.exists(rocm_sdk_base):
            rocm_sdk_include = os.path.join(rocm_sdk_base, "include")
            rocm_sdk_lib = os.path.join(rocm_sdk_base, "lib")

            if os.path.exists(rocm_sdk_include):
                include_dirs.append(rocm_sdk_include)
                print(f"[diff-gaussian] Added ROCm SDK include: {rocm_sdk_include}")

            if os.path.exists(rocm_sdk_lib):
                library_dirs.append(rocm_sdk_lib)
                print(f"[diff-gaussian] Added ROCm SDK library: {rocm_sdk_lib}")

    # For ROCm/HIP, we need to use our pre-fixed hip files as sources
    # This prevents PyTorch's CUDAExtension from re-running hipify
    hip_rasterizer_dir = os.path.join(base_dir, "hip_rasterizer")

    # Check if we have already hipified and fixed files
    hip_forward = os.path.join(hip_rasterizer_dir, "forward.hip")
    has_fixed_files = os.path.exists(hip_forward)

    if has_fixed_files:
        # Check if the files have already been fixed
        with open(hip_forward, 'r') as f:
            content = f.read()
        needs_fix = '<< <' in content  # malformed syntax indicator

        if needs_fix:
            print("[HIP FIX] Fixing kernel launch syntax in hipified files...")
            hip_files = glob.glob(os.path.join(hip_rasterizer_dir, "*.hip"))
            for hip_file in hip_files:
                with open(hip_file, 'r') as f:
                    content = f.read()
                fixed_content = fix_hipify_issues(content)
                if fixed_content != content:
                    print(f"[HIP FIX] Fixed: {os.path.basename(hip_file)}")
                    with open(hip_file, 'w') as f:
                        f.write(fixed_content)

        # Use the hipified .hip files directly as sources
        sources = [
            os.path.join(hip_rasterizer_dir, "rasterizer_impl.hip"),
            os.path.join(hip_rasterizer_dir, "forward.hip"),
            os.path.join(hip_rasterizer_dir, "backward.hip"),
            os.path.join(base_dir, "rasterize_points.hip") if os.path.exists(os.path.join(base_dir, "rasterize_points.hip")) else os.path.join(base_dir, "rasterize_points.cu"),
            os.path.join(base_dir, "ext.cpp"),
        ]

        # Detect GPU architecture
        gpu_arch = detect_gpu_arch()
        print(f"[HIP] Detected GPU architecture: {gpu_arch}")

        # Build compiler flags - Windows vs Linux have different requirements
        glm_path = os.path.join(base_dir, "third_party", "glm")

        nvcc_flags = [
            f"-I{glm_path}",
            f"-I{hip_rasterizer_dir}",
            f"--offload-arch={gpu_arch}",
        ]

        # Add TheRock include to hipcc only (not MSVC - can cause header conflicts)
        if os.path.exists(therock_include):
            nvcc_flags.append(f"-I{therock_include}")
            print(f"[diff-gaussian] Added TheRock include for hipcc: {therock_include}")

        cxx_flags = []

        # Windows-specific flags
        if is_windows:
            cxx_flags.extend(["/wd4067", "/wd4624", "/wd4996"])
            # STRIP_ERROR_MESSAGES avoids ABI issues between hipcc/clang and MSVC
            cxx_flags.append("-DSTRIP_ERROR_MESSAGES")
            nvcc_flags.append("-DSTRIP_ERROR_MESSAGES")
        else:
            nvcc_flags.append("-fgpu-rdc")  # GPU relocatable device code

        extra_compile_args = {"cxx": cxx_flags, "nvcc": nvcc_flags}
        print(f"[HIP] Using pre-fixed hipified sources from {hip_rasterizer_dir}")
    else:
        # First run - use original .cu files, hipify will generate them
        sources = [
            os.path.join(base_dir, "cuda_rasterizer", "rasterizer_impl.cu"),
            os.path.join(base_dir, "cuda_rasterizer", "forward.cu"),
            os.path.join(base_dir, "cuda_rasterizer", "backward.cu"),
            os.path.join(base_dir, "rasterize_points.cu"),
            os.path.join(base_dir, "ext.cpp"),
        ]
        extra_compile_args = {
            "nvcc": [
                f"-I{os.path.join(base_dir, 'third_party', 'glm')}",
                f"-I{os.path.join(base_dir, 'cuda_rasterizer')}",
            ]
        }
        print("[HIP] First build - will use original .cu files. Run again after hipify to use fixed files.")
else:
    # CUDA path - use original files
    sources = [
        os.path.join(base_dir, "cuda_rasterizer", "rasterizer_impl.cu"),
        os.path.join(base_dir, "cuda_rasterizer", "forward.cu"),
        os.path.join(base_dir, "cuda_rasterizer", "backward.cu"),
        os.path.join(base_dir, "rasterize_points.cu"),
        os.path.join(base_dir, "ext.cpp"),
    ]
    extra_compile_args = {
        "nvcc": [f"-I{os.path.join(base_dir, 'third_party', 'glm')}"]
    }


setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_compile_args=extra_compile_args)
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
