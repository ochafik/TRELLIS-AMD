---
name: cuda-to-rocm-migration
description: Use this skill when porting CUDA/NVIDIA GPU code to AMD ROCm/HIP. Covers warp intrinsics, cooperative groups, setup.py patterns, kernel modifications, and common pitfalls.
---

# CUDA to ROCm/HIP Migration Skill

A comprehensive guide for porting CUDA code to AMD ROCm/HIP, derived from real-world migration of TRELLIS (3D asset generation) to AMD GPUs.

---

## Quick Reference

### Header Mappings

| CUDA | HIP |
|------|-----|
| `#include <cuda.h>` | `#include <hip/hip_runtime.h>` |
| `#include <cuda_runtime.h>` | `#include <hip/hip_runtime.h>` |
| `#include <cuda_fp16.h>` | `#include <hip/hip_fp16.h>` |
| `#include <cublas_v2.h>` | `#include <hipblas/hipblas.h>` |
| `#include <cooperative_groups.h>` | Custom implementation (see below) |

### API Mappings

| CUDA | HIP |
|------|-----|
| `cudaMalloc` | `hipMalloc` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `cudaGetDevice` | `hipGetDevice` |
| `cudaFuncSetCacheConfig` | `hipFuncSetCacheConfig` |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | `hipOccupancyMaxActiveBlocksPerMultiprocessor` |

### Preprocessor Detection

```cpp
#ifdef __HIP_PLATFORM_AMD__
  // AMD/ROCm specific code
#else
  // NVIDIA CUDA code
#endif

// In Python
import torch
IS_ROCM = hasattr(torch.version, 'hip') and torch.version.hip is not None
```

---

## 1. Warp Intrinsic Compatibility Layer

HIP doesn't have `_sync` versions of warp intrinsics. Create compatibility macros:

```cpp
// Add at the top of .hip or .cu files
#ifdef __HIP_PLATFORM_AMD__

#ifndef __ballot_sync
#define __ballot_sync(mask, predicate) __ballot(predicate)
#endif

#ifndef __all_sync
#define __all_sync(mask, predicate) __all(predicate)
#endif

#ifndef __any_sync
#define __any_sync(mask, predicate) __any(predicate)
#endif

#ifndef __match_any_sync
#define __match_any_sync(mask, val) __ballot(1)  // Simplified fallback
#endif

#ifndef __syncwarp
#define __syncwarp(...) __threadfence_block()
#endif

#endif // __HIP_PLATFORM_AMD__
```

### Fast Math Intrinsics

```cpp
#ifdef __HIP_PLATFORM_AMD__
#ifndef __frcp_rz
#define __frcp_rz(x) (1.0f / (x))
#endif
#endif
```

### Float/Int Bit Casting (Host-Compatible)

```cpp
#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
inline int __float_as_int(float x) {
    union { float f; int i; } u;
    u.f = x;
    return u.i;
}

inline float __int_as_float(int x) {
    union { float f; int i; } u;
    u.i = x;
    return u.f;
}
#endif
```

---

## 2. Cooperative Groups Implementation

HIP doesn't provide `cooperative_groups`. Create a minimal implementation:

```cpp
// hip_cooperative_groups.h
#ifndef HIP_COOPERATIVE_GROUPS_H
#define HIP_COOPERATIVE_GROUPS_H

#include <hip/hip_runtime.h>

#ifdef __HIP_PLATFORM_AMD__

namespace cooperative_groups {

class thread_block {
public:
    __device__ __forceinline__ dim3 group_index() const { return blockIdx; }
    __device__ __forceinline__ dim3 thread_index() const { return threadIdx; }

    __device__ __forceinline__ unsigned int thread_rank() const {
        return threadIdx.x + threadIdx.y * blockDim.x +
               threadIdx.z * blockDim.x * blockDim.y;
    }

    __device__ __forceinline__ unsigned int size() const {
        return blockDim.x * blockDim.y * blockDim.z;
    }

    __device__ __forceinline__ void sync() const { __syncthreads(); }
};

class grid_group {
public:
    __device__ __forceinline__ unsigned long long thread_rank() const {
        unsigned long long block_idx = blockIdx.x +
            (unsigned long long)blockIdx.y * gridDim.x +
            (unsigned long long)blockIdx.z * gridDim.x * gridDim.y;
        unsigned long long threads_per_block =
            (unsigned long long)blockDim.x * blockDim.y * blockDim.z;
        unsigned long long thread_idx = threadIdx.x +
            threadIdx.y * blockDim.x +
            threadIdx.z * blockDim.x * blockDim.y;
        return block_idx * threads_per_block + thread_idx;
    }

    __device__ __forceinline__ unsigned long long size() const {
        return (unsigned long long)gridDim.x * gridDim.y * gridDim.z *
               blockDim.x * blockDim.y * blockDim.z;
    }
};

__device__ __forceinline__ thread_block this_thread_block() {
    return thread_block();
}

__device__ __forceinline__ grid_group this_grid() {
    return grid_group();
}

} // namespace cooperative_groups

#endif // __HIP_PLATFORM_AMD__
#endif // HIP_COOPERATIVE_GROUPS_H
```

Usage in code:

```cpp
#ifdef __HIP_PLATFORM_AMD__
#include "hip_cooperative_groups.h"
#else
#include <cooperative_groups.h>
#endif
namespace cg = cooperative_groups;
```

---

## 3. Half2 AtomicAdd Implementation

HIP doesn't natively support `atomicAdd` for `half2`. Inject this implementation:

```cpp
#if defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ half2 atomicAdd(half2* address, half2 val) {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    do {
        assumed = old;
        half2 old_val = *reinterpret_cast<half2*>(&assumed);
        half2 new_val = __hadd2(old_val, val);
        old = atomicCAS(address_as_uint, assumed,
                        *reinterpret_cast<unsigned int*>(&new_val));
    } while (assumed != old);
    return *reinterpret_cast<half2*>(&old);
}
#endif
```

---

## 4. Setup.py Patterns for ROCm

### Detecting ROCm Build

```python
import torch

IS_ROCM = hasattr(torch.version, 'hip') and torch.version.hip is not None

if IS_ROCM:
    print(f"Building for AMD ROCm/HIP: {torch.version.hip}")
else:
    print(f"Building for NVIDIA CUDA: {torch.version.cuda}")
```

### Conditional Source Files

```python
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

if IS_ROCM:
    sources = [
        "csrc/common/kernel.cu",        # hipcc handles .cu
        "csrc/hip_impl/raster.hip",     # native .hip file
        "csrc/bindings.cpp",
    ]
    extra_compile_args = {
        "cxx": ["-DNVDR_TORCH", "-D__HIP_PLATFORM_AMD__"],
        "nvcc": ["-DNVDR_TORCH", "-D__HIP_PLATFORM_AMD__"],
    }
else:
    sources = [
        "csrc/common/kernel.cu",
        "csrc/cuda_impl/raster.cu",
        "csrc/bindings.cpp",
    ]
    extra_compile_args = {
        "cxx": ["-DNVDR_TORCH"],
        "nvcc": ["-DNVDR_TORCH", "-lineinfo"],
    }

setup(
    ext_modules=[
        CUDAExtension("my_extension", sources, extra_compile_args=extra_compile_args)
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

### Excluding PTX Assembly Files

Some CUDA files use inline PTX assembly that won't work on HIP:

```python
HIP_EXCLUDED_FILES = [
    'implicit_gemm',    # Tensor core MMA ops
    'fetch_on_demand',  # PTX cvta.to.shared
]

sources = []
for fpath in glob.glob("backend/**/*.cu"):
    if IS_ROCM and any(excluded in fpath for excluded in HIP_EXCLUDED_FILES):
        print(f"  Excluding (PTX assembly): {fpath}")
        continue
    sources.append(fpath)
```

### Fixing Hipified Kernel Launches

PyTorch's hipify can produce malformed `hipLaunchKernelGGL` calls:

```python
import re

def fix_hipify_kernel_launches(content):
    """Fix malformed hipLaunchKernelGGL calls.

    PyTorch's hipify converts CUDA <<<grid, block>>> to:
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, 0, 0, args...)

    Correct format:
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, args...)
    """
    return re.sub(r',\s*0,\s*0,\s*0,\s*0,', ', 0, 0,', content)


class HipFixBuildExtension(BuildExtension):
    def build_extensions(self):
        if IS_ROCM:
            import glob
            for hip_file in glob.glob("**/*_hip.hip", recursive=True):
                with open(hip_file, 'r') as f:
                    content = f.read()
                fixed = fix_hipify_kernel_launches(content)
                if fixed != content:
                    with open(hip_file, 'w') as f:
                        f.write(fixed)
                    print(f"  Fixed hipLaunchKernelGGL in: {hip_file}")
        super().build_extensions()
```

---

## 5. Manual HIP Build Script

For complex extensions that PyTorch's build system struggles with:

```bash
#!/bin/bash
# build_hip.sh - Manual HIP build with proper path quoting

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$VIRTUAL_ENV"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ROCM_DIR="/opt/rocm"
GPU_ARCH=$(rocminfo | grep -o 'gfx[0-9a-z]*' | head -1)

# Use arrays to handle paths with spaces
COMMON_FLAGS=(
    -fPIC -O3 -std=c++17
    "--offload-arch=${GPU_ARCH}"
    -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1
    -DTORCH_API_INCLUDE_EXTENSION_H
)

TORCH_INCLUDE="${VENV_DIR}/lib/python${PYTHON_VERSION}/site-packages/torch/include"

INCLUDES=(
    "-I${TORCH_INCLUDE}"
    "-I${TORCH_INCLUDE}/torch/csrc/api/include"
    "-I${ROCM_DIR}/include"
    "-I${SCRIPT_DIR}/src"
)

# Compile with proper quoting
"${ROCM_DIR}/bin/hipcc" "${COMMON_FLAGS[@]}" "${INCLUDES[@]}" \
    -c "${SCRIPT_DIR}/src/kernel.hip" -o "${BUILD_DIR}/kernel.o"
```

---

## 6. Python Backend Selection

### Environment Variables

```bash
# Disable xformers (CUDA-only attention)
export XFORMERS_DISABLED=1

# Use PyTorch's SDPA attention
export ATTN_BACKEND=sdpa

# Use torchsparse for sparse convolutions (works with HIP)
export SPARSE_BACKEND=torchsparse
```

### Backend Selection in Code

```python
import os

# Attention backend
env_attn = os.environ.get('ATTN_BACKEND')
if env_attn in ['xformers', 'flash_attn', 'sdpa', 'naive']:
    BACKEND = env_attn
elif torch.cuda.is_available():
    try:
        import xformers
        BACKEND = 'xformers'
    except ImportError:
        BACKEND = 'sdpa'
else:
    BACKEND = 'naive'

# Use backend
if BACKEND == 'xformers':
    import xformers.ops as xops
    out = xops.memory_efficient_attention(q, k, v)
elif BACKEND == 'sdpa':
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
elif BACKEND == 'flash_attn':
    from flash_attn import flash_attn_func
    out = flash_attn_func(q, k, v)
else:
    # Naive implementation
    attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d), dim=-1)
    out = attn @ v
```

---

## 7. Rasterizer Backend Selection

nvdiffrast CUDA rasterizer may have issues on HIP. Switch to OpenGL:

```python
import nvdiffrast.torch as dr

# CUDA context (may not work on HIP)
# glctx = dr.RasterizeCudaContext(device=device)

# OpenGL context (works on both)
glctx = dr.RasterizeGLContext(device=device)  # AMD HIP FIX

# For utils3d
import utils3d.torch
# rastctx = utils3d.torch.RastContext(backend='cuda')  # May fail on HIP
rastctx = utils3d.torch.RastContext(backend='gl')  # AMD HIP FIX
```

---

## 8. Warp-Level Synchronization Issues

### Problem

AMD GPUs have different warp (wavefront) semantics:
- NVIDIA: 32 threads per warp, lock-step execution
- AMD RDNA: 32 or 64 threads per wavefront, less strict synchronization

Code using conditional `__syncthreads()` or complex warp voting can deadlock.

### Solution: Simplified Kernels

Replace complex warp-synchronized kernels with simpler implementations:

```cpp
// ORIGINAL (may deadlock on AMD):
__device__ void complexKernel() {
    if (threadIdx.x < 16) {
        __syncwarp();  // Only half the warp syncs!
        // ...
    }
    __ballot_sync(0xffffffff, condition);  // Assumes all threads participate
}

// FIXED (AMD-safe):
__device__ void simpleKernel() {
    __syncthreads();  // Full block sync is safe

    // Process work without assuming warp-level coordination
    for (int i = threadIdx.x; i < work_size; i += blockDim.x) {
        // Each thread works independently
    }

    __syncthreads();  // Sync before shared memory access
}
```

---

## 9. Coalesced Atomics Compatibility

Disable advanced atomic coalescing on HIP (uses `__match_any_sync`):

```cpp
// Only enable on NVIDIA SM 7.0+, not on HIP
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700) && !defined(__HIP_PLATFORM_AMD__)

#define CA_SET_GROUP(group) \
    int tmask = __match_any_sync(0xffffffff, (group)); \
    // ... complex coalescing logic

#else  // Fallback for HIP and older CUDA

#define CA_SET_GROUP(group)
#define caAtomicAdd(ptr, value) atomicAdd((ptr), (value))

#endif
```

---

## 10. File Organization

### Directory Structure

```
extension/
├── cuda_impl/           # NVIDIA-only CUDA files
│   ├── kernel.cu
│   └── raster.cu
├── hip_impl/            # AMD-only HIP files
│   ├── kernel.hip
│   ├── raster.hip
│   └── hip_cooperative_groups.h
├── common/              # Shared code (hipcc handles .cu)
│   ├── utils.cu         # Will be hipified automatically
│   └── common.h         # Platform detection macros
├── setup.py             # Conditional source selection
└── build_hip.sh         # Manual build script
```

### File Extensions

| Extension | Purpose |
|-----------|---------|
| `.cu` | CUDA source (hipcc can compile, PyTorch will hipify) |
| `.hip` | Native HIP source (use for HIP-specific code) |
| `.cuh` | CUDA header |
| `.h` / `.hpp` | Shared headers with `#ifdef __HIP_PLATFORM_AMD__` |

---

## 11. Common Issues and Fixes

### Issue: Undefined `__shfl_sync`

```cpp
// HIP uses __shfl without _sync
#ifdef __HIP_PLATFORM_AMD__
#define __shfl_sync(mask, var, srcLane) __shfl(var, srcLane)
#define __shfl_up_sync(mask, var, delta) __shfl_up(var, delta)
#define __shfl_down_sync(mask, var, delta) __shfl_down(var, delta)
#define __shfl_xor_sync(mask, var, laneMask) __shfl_xor(var, laneMask)
#endif
```

### Issue: `__ldg` Not Available

```cpp
#ifdef __HIP_PLATFORM_AMD__
#define __ldg(ptr) (*(ptr))  // HIP has no __ldg, just dereference
#endif
```

### Issue: C++ ABI Mismatch

```bash
# Match PyTorch's C++ ABI (usually CXX11 on modern systems)
-D_GLIBCXX_USE_CXX11_ABI=1   # NOT 0
```

### Issue: Buffer Initialization

HIP may not zero-initialize buffers like CUDA. Explicitly initialize:

```cpp
__global__ void kernel(int* output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // AMD HIP FIX: Initialize output buffer
    if (idx < size) {
        output[idx] = 0;
    }
    __syncthreads();

    // ... rest of kernel
}
```

---

## 12. Testing ROCm Builds

### Verification Script

```python
#!/usr/bin/env python3
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if hasattr(torch.version, 'hip'):
    print(f"HIP version: {torch.version.hip}")
    print(f"ROCm build: YES")
else:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"ROCm build: NO")

if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Test basic operations
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = x @ y
    print(f"Matrix multiply: OK ({z.shape})")
```

### Extension Test

```python
def test_extension():
    try:
        import my_extension

        # Test with small input
        x = torch.randn(100, 3, device='cuda')
        result = my_extension.forward(x)

        print(f"Extension output shape: {result.shape}")
        print("Extension test: PASSED")
    except Exception as e:
        print(f"Extension test: FAILED - {e}")
```

---

## Summary Checklist

When porting CUDA to ROCm:

1. [ ] Add `#ifdef __HIP_PLATFORM_AMD__` detection
2. [ ] Add warp intrinsic compatibility macros (`__ballot_sync`, etc.)
3. [ ] Implement `cooperative_groups` if needed
4. [ ] Implement `half2 atomicAdd` if needed
5. [ ] Update `setup.py` with ROCm detection and conditional sources
6. [ ] Exclude files with PTX inline assembly
7. [ ] Fix hipified kernel launch syntax
8. [ ] Replace complex warp-synchronized kernels with simpler versions
9. [ ] Switch to OpenGL rasterizer if CUDA rasterizer fails
10. [ ] Set environment variables (`ATTN_BACKEND=sdpa`, etc.)
11. [ ] Explicitly initialize output buffers
12. [ ] Use correct C++ ABI (`_GLIBCXX_USE_CXX11_ABI=1`)
13. [ ] Quote paths with spaces in build scripts (use bash arrays)
14. [ ] Test with `rocminfo` and simple PyTorch operations first
