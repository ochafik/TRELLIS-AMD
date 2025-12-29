import glob
import os
import re
import sys
import subprocess
import platform

import torch
import torch.cuda
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

# from torchsparse import __version__

version_file = open("./torchsparse/version.py")
version = version_file.read().split("'")[1]
print("torchsparse version:", version)

# Check if we're building for ROCm/HIP
is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
is_windows = platform.system() == 'Windows'
print(f"ROCm/HIP detected: {is_rocm}")
print(f"Platform: {'Windows' if is_windows else 'Linux'}")

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
            print(f"[torchsparse] Fixed: -I{path} -> -I{short_path}")
            return '-I' + short_path
        return arg

    _orig_nt_quote_args = cpp_ext._nt_quote_args
    def _patched_nt_quote_args(args):
        if not args:
            return []
        fixed_args = [_fix_include_arg(arg) if arg.startswith('-I') else arg for arg in args]
        return _orig_nt_quote_args(fixed_args)
    cpp_ext._nt_quote_args = _patched_nt_quote_args
    print("[torchsparse] Patched _nt_quote_args to fix HIP include paths")

    # Fix PyTorch looking for hipcc in 'bin' instead of 'Scripts' on Windows
    _orig_get_hipcc_path = cpp_ext._get_hipcc_path
    def _patched_get_hipcc_path():
        path = _orig_get_hipcc_path()
        if '\\bin\\' in path:
            fixed = path.replace('\\bin\\', '\\Scripts\\')
            if os.path.exists(fixed):
                print(f"[torchsparse] Fixed hipcc path: {path} -> {fixed}")
                return fixed
        return path
    cpp_ext._get_hipcc_path = _patched_get_hipcc_path

def detect_gpu_arch():
    """Detect AMD GPU architecture from environment or rocminfo."""
    gpu_arch = os.environ.get('GPU_ARCH', os.environ.get('PYTORCH_ROCM_ARCH', ''))
    if gpu_arch:
        return gpu_arch

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

    return "gfx1100"  # Default

if is_rocm:
    gpu_arch = detect_gpu_arch()
    print(f"GPU architecture: {gpu_arch}")

if (torch.cuda.is_available() and (CUDA_HOME is not None or is_rocm)) or (
    os.getenv("FORCE_CUDA", "0") == "1"
):
    device = "cuda"
    # Always use pybind_cuda.cu - PyTorch will hipify it for ROCm
    pybind_fn = "pybind_cuda.cu"
else:
    device = "cpu"
    pybind_fn = f"pybind_{device}.cpp"

# Files with CUDA-specific PTX inline assembly that don't work with HIP
# These use tensor core MMA ops and PTX cvta.to.shared instructions
HIP_EXCLUDED_FILES = [
    'implicit_gemm',  # All implicit GEMM files have PTX assembly
    'fetch_on_demand',  # May also have issues
]

sources = [os.path.join("torchsparse", "backend", pybind_fn)]
for fpath in glob.glob(os.path.join("torchsparse", "backend", "**", "*")):
    if fpath.endswith("_cpu.cpp") and device in ["cpu", "cuda"]:
        sources.append(fpath)
    elif device == "cuda":
        # Always use _cuda.cu files - PyTorch will hipify them on ROCm
        # Our custom HipFixBuildExtension patches the hipified output
        if fpath.endswith("_cuda.cu"):
            # Exclude files with PTX assembly issues for ROCm
            if is_rocm and any(excluded in fpath for excluded in HIP_EXCLUDED_FILES):
                print(f"  Excluding (PTX assembly): {fpath}")
                continue
            sources.append(fpath)


print(f"Building with device: {device}, sources: {len(sources)} files")
if is_rocm:
    print("Excluded for HIP:", HIP_EXCLUDED_FILES)

extension_type = CUDAExtension if device == "cuda" else CppExtension

# Include and library directories
include_dirs = []
library_dirs = []
base_dir = os.path.dirname(os.path.abspath(__file__))

# Add ROCm SDK paths for Windows TheRock builds
if is_rocm and is_windows:
    # Use TheRock 7.10.0 installation
    therock_include = r"C:\TheRock\build\include"
    therock_lib = r"C:\TheRock\build\lib"

    if os.path.exists(therock_include):
        include_dirs.append(therock_include)
        print(f"[torchsparse] Added TheRock include: {therock_include}")

    # Add sparsehash for google/dense_hash_map
    # We need both paths:
    # - third_party/sparsehash/sparsehash for google/ compatibility headers
    # - third_party/sparsehash for sparsehash/ headers (used by compat headers)
    sparsehash_base = os.path.join(base_dir, "third_party", "sparsehash")
    sparsehash_include = os.path.join(sparsehash_base, "sparsehash")
    if os.path.exists(sparsehash_include):
        include_dirs.append(sparsehash_base)  # For sparsehash/dense_hash_map
        include_dirs.append(sparsehash_include)  # For google/dense_hash_map
        print(f"[torchsparse] Added sparsehash includes: {sparsehash_base}, {sparsehash_include}")

    if os.path.exists(therock_lib):
        library_dirs.append(therock_lib)
        print(f"[torchsparse] Added TheRock library: {therock_lib}")

    # Also add pip ROCm SDK paths as fallback
    rocm_sdk_base = os.path.join(sys.prefix, "Lib", "site-packages", "_rocm_sdk_core")
    if os.path.exists(rocm_sdk_base):
        rocm_sdk_lib = os.path.join(rocm_sdk_base, "lib")
        if os.path.exists(rocm_sdk_lib):
            library_dirs.append(rocm_sdk_lib)
            print(f"[torchsparse] Added ROCm SDK library: {rocm_sdk_lib}")

# Platform-specific compile flags
if is_windows:
    extra_compile_args = {
        "cxx": ["/O2", "/openmp"],
        "nvcc": ["-O3", "-std=c++17"],
    }
    # Add STRIP_ERROR_MESSAGES for ROCm to avoid ABI issues
    if is_rocm:
        extra_compile_args["cxx"].append("-DSTRIP_ERROR_MESSAGES")
        extra_compile_args["nvcc"].append("-DSTRIP_ERROR_MESSAGES")
        extra_compile_args["cxx"].extend(["/wd4067", "/wd4624", "/wd4996"])
else:
    extra_compile_args = {
        "cxx": ["-g", "-O3", "-fopenmp", "-lgomp"],
        "nvcc": ["-O3", "-std=c++17"],
    }

# Add GPU arch for ROCm builds
if is_rocm and 'gpu_arch' in dir():
    extra_compile_args["nvcc"].append(f"--offload-arch={gpu_arch}")

# Custom BuildExtension that fixes hipified files before compilation
import re

# HIP-compatible half2 atomicAdd implementation to inject into hipified files
HIP_HALF2_ATOMICADD = '''
// HIP-compatible atomicAdd for half2 - injected by torchsparse setup.py
#if defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ half2 atomicAdd(half2* address, half2 val) {
    // Fallback implementation using float atomics
    // Convert half2 to two separate half values, do atomic add via unsigned int CAS
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    do {
        assumed = old;
        half2 old_val = *reinterpret_cast<half2*>(&assumed);
        half2 new_val = __hadd2(old_val, val);
        old = atomicCAS(address_as_uint, assumed, *reinterpret_cast<unsigned int*>(&new_val));
    } while (assumed != old);
    return *reinterpret_cast<half2*>(&old);
}
#endif
'''

def fix_hipify_kernel_launches(content):
    """Fix malformed hipLaunchKernelGGL calls and add HIP compatibility shims.
    
    PyTorch's hipify converts CUDA <<<grid, block>>> to:
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, 0, 0, args...)
    
    The correct format is:
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, args...)
    
    This removes the extra "0, 0, " arguments.
    
    Also adds HIP-compatible atomicAdd for half2 types since HIP doesn't natively
    support atomicAdd for half2.
    """
    # Pattern: ", 0, 0, 0, 0," should be ", 0, 0,"
    fixed = re.sub(r',\s*0,\s*0,\s*0,\s*0,', ', 0, 0,', content)
    
    # Inject half2 atomicAdd if file uses atomicAdd with half2 and doesn't already have fix
    if 'atomicAdd' in fixed and 'half2' in fixed and 'HIP-compatible atomicAdd for half2' not in fixed:
        # Find the first include statement and add our shim after includes
        # Look for the last #include line before the first function/kernel
        lines = fixed.split('\n')
        last_include_idx = 0
        for idx, line in enumerate(lines):
            if line.strip().startswith('#include'):
                last_include_idx = idx
        # Insert our shim after the last include
        lines.insert(last_include_idx + 1, HIP_HALF2_ATOMICADD)
        fixed = '\n'.join(lines)
    
    return fixed


class HipFixBuildExtension(BuildExtension):
    """Custom build extension that fixes hipified files before compilation."""
    
    def build_extensions(self):
        if is_rocm:
            # Find and fix all hipified files before compilation
            import glob as glob_module
            hip_files = glob_module.glob("torchsparse/backend/**/*_hip.hip", recursive=True)
            for hip_file in hip_files:
                try:
                    with open(hip_file, 'r') as f:
                        content = f.read()
                    fixed = fix_hipify_kernel_launches(content)
                    if fixed != content:
                        with open(hip_file, 'w') as f:
                            f.write(fixed)
                        print(f"  Fixed hipLaunchKernelGGL in: {hip_file}")
                except Exception as e:
                    print(f"  Warning: Could not fix {hip_file}: {e}")
        
        super().build_extensions()

setup(
    name="torchsparse",
    version=version,
    packages=find_packages(),
    ext_modules=[
        extension_type(
            "torchsparse.backend", sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_compile_args=extra_compile_args
        )
    ],
    url="https://github.com/mit-han-lab/torchsparse",
    install_requires=[
        "numpy",
        "backports.cached_property",
        "tqdm",
        "typing-extensions",
        "wheel",
        "rootpath",
        "torch",
        "torchvision"
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cu118'
    ],
    cmdclass={"build_ext": HipFixBuildExtension},
    zip_safe=False,
)

