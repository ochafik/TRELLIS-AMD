# nvdiffrast HIP Crash Analysis

**Date:** 2024-12-30
**Status:** Under investigation

## Crash Symptoms

### Error Details
```
Exception Code: 0xC0000005 (Access Violation)
Faulting modules:
  - amdhip64_7.dll (HIP runtime)
  - _nvdiffrast_c.cp312-win_amd64.pyd (compiled extension)
```

### When Crashes Occur
1. Creating `RasterizeCudaContext` / `RasterizeGLContext`
2. Calling `dr.rasterize()` for mesh rendering
3. Calling `dr.texture()` in 'opt' mode for texture baking
4. Mesh normal preview in `render_utils.render_video()`

## Root Cause Analysis

### Finding 1: No Real OpenGL Path

The `RasterizeGLContext` is **deprecated** and internally uses `RasterizeCudaContext`:

```python
# nvdiffrast/torch/ops.py:550
class RasterizeGLContext(RasterizeCudaContext):
    def __init__(self, ...):
        warnings.warn("RasterizeGLContext has been deprecated and uses RasterizeCudaContext internally")
```

**Impact:** All rasterization goes through HIP kernels, no OpenGL fallback exists.

### Finding 2: Warp Intrinsic Incompatibility

The HIP port uses compatibility macros that are **semantically incorrect**:

```cpp
// RasterImpl_kernel.hip - Current implementation (WRONG)
#define __ballot_sync(mask, predicate) __ballot(predicate)
#define __all_sync(mask, predicate) __all(predicate)
#define __syncwarp(...) __threadfence_block()  // WRONG!
```

**Problems:**
1. `__syncwarp()` synchronizes threads within a warp (32 threads on NVIDIA)
2. `__threadfence_block()` is a memory fence, NOT a synchronization barrier
3. AMD uses wave64 by default (64 threads), different from CUDA's warp32

**Correct AMD equivalents:**
```cpp
// Proper HIP intrinsics for warp sync
#define __syncwarp(...) __builtin_amdgcn_wave_barrier()

// For ballot, need to handle wave64 vs wave32
// wave64: __ballot64(predicate)
// wave32: __ballot(predicate)
```

### Finding 3: Kernel Pipeline Dependencies

The rasterization pipeline has 4 kernels with data dependencies:

```
triangleSetupKernel → binRasterKernel → coarseRasterKernel → fineRasterKernel
```

Each kernel relies on warp-level communication that may fail on AMD:
- Triangle setup: writes to shared memory
- Bin raster: atomic operations on bin segments
- Coarse raster: warp-level voting for tile coverage (uses __ballot)
- Fine raster: pixel-level output with warp communication

### Finding 4: CoarseRaster Simplification

A simplified coarse raster was added to avoid warp sync:

```cpp
// RasterImpl_kernel.hip:54
// AMD HIP FIX: Use simplified coarse raster that doesn't use warp-level sync
__global__ void coarseRasterKernel(const CRParams p) {
    CR::coarseRasterImplSimple(p);  // Uses CoarseRasterSimple.inl
}
```

This helps but doesn't fix all issues - fine raster still has problems.

## nvdiffrast Usage in TRELLIS

### MeshRenderer (trellis/renderers/mesh_renderer.py)

```python
class MeshRenderer:
    def __init__(self):
        self.glctx = dr.RasterizeGLContext(device=device)  # Crashes here

    def render(self, mesh, ...):
        rast, _ = dr.rasterize(self.glctx, vertices, faces, resolution)  # Or here
        img = dr.interpolate(attrs, rast, faces)
        img = dr.antialias(img, rast, vertices, faces)
```

### postprocessing_utils.py - Texture Baking

```python
def bake_texture(...):
    # In 'opt' mode:
    render = dr.texture(texture, uv, uv_dr)[0]  # Crashes here

    # In 'fast' mode:
    # Uses simple projection, no dr.texture() - works
```

### postprocessing_utils.py - fill_holes

```python
def fill_holes(...):
    # Uses dr.rasterize() for visibility check
    # Returns incorrect data on AMD - marks all faces as invisible
    # Result: all faces removed
```

## Memory Access Pattern Analysis

The crash (0xC0000005) suggests illegal memory access. Likely causes:

1. **Out-of-bounds shared memory access** due to wave size mismatch
2. **Race conditions** from missing synchronization
3. **Uninitialized memory** read due to wrong ballot mask

### Shared Memory Layout Issue

CUDA code assumes warp size of 32:
```cpp
__shared__ int sharedData[WARPS_PER_BLOCK * 32];  // Assumes 32 threads per warp
int idx = threadIdx.x + warpId * 32;  // Wrong on AMD wave64
```

AMD wave64 would need:
```cpp
__shared__ int sharedData[WARPS_PER_BLOCK * 64];  // 64 threads per wave
int idx = threadIdx.x + warpId * 64;
```

## Affected Files

| File | Issue |
|------|-------|
| `extensions/nvdiffrast-hip/csrc/common/hipraster/impl/RasterImpl_kernel.hip` | Warp intrinsic macros |
| `extensions/nvdiffrast-hip/csrc/common/hipraster/impl/CoarseRaster.inl` | Original coarse raster (disabled) |
| `extensions/nvdiffrast-hip/csrc/common/hipraster/impl/FineRaster.inl` | Fine raster with warp ops |
| `extensions/nvdiffrast-hip/csrc/torch/torch_rasterize_hip.hip` | Torch bindings |
| `extensions/nvdiffrast-hip/csrc/torch/torch_texture_hip.hip` | Texture sampling kernel |

## Testing Strategy

To debug, need to:

1. **Isolate kernels** - Test each kernel independently
2. **Add bounds checking** - Verify all memory accesses
3. **Check wave size** - Query actual wave size at runtime
4. **Add synchronization** - Extra barriers at critical points
5. **Use ROCm debugger** - `rocgdb` for crash analysis

## References

- [HIP Porting Guide - Warp Functions](https://rocm.docs.amd.com/projects/HIP/en/latest/user_guide/hip_porting_guide.html)
- [AMD RDNA Architecture - Wave32 vs Wave64](https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/)
- [nvdiffrast GitHub Issues](https://github.com/NVlabs/nvdiffrast/issues)
