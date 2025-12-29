# TRELLIS AMD Windows Support - Overview

This documentation covers the work done to port TRELLIS to AMD GPUs on Windows using ROCm/HIP.

## Target Environment

- **OS**: Windows 11 (WSL2 compatible)
- **GPU**: AMD Radeon (RDNA3 architecture, e.g., gfx1100/gfx1151)
- **ROCm Version**: 7.11.0a (nightly) via TheRock
- **PyTorch**: 2.11.0a0+rocm7.11.0a20251218

## Components

TRELLIS requires several GPU-accelerated extensions:

1. **nvdiffrast-hip** - Differentiable rasterization (ported from CUDA)
2. **diff-gaussian-rasterization** - 3D Gaussian splatting rasterizer
3. **torchsparse** - Sparse tensor operations for 3D data
4. **spconv** - Sparse convolutions (work in progress)

## ROCm Setup

The project uses TheRock (AMD's new open build system) for ROCm on Windows:
- Headers: `C:\TheRock\build\include` (rocprim, hipcub, thrust)
- Libraries: `C:\TheRock\build\lib`

PyTorch ROCm nightlies provide:
- Core ROCm SDK via pip: `_rocm_sdk_core`
- hipcc compiler in `Scripts/` directory

## Key Challenges

1. **Path handling** - Windows paths with spaces break PyTorch's hipify
2. **Header conflicts** - ROCm headers can conflict with MSVC headers
3. **Type differences** - `long` is 32-bit on Windows vs 64-bit on Linux
4. **ABI compatibility** - hipcc (clang) vs MSVC object file compatibility
5. **Missing flash_attn** - No AMD support, use PyTorch's SDPA instead
