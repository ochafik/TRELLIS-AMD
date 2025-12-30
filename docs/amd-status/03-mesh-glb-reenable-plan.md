# Plan: Re-enable Mesh/GLB Export on AMD

**Date:** 2024-12-30
**Status:** Planning
**Priority:** High

## Current State

Mesh/GLB export is disabled on AMD due to nvdiffrast HIP crashes. Users can only export Gaussian PLY files.

## Goals

1. Enable mesh generation on AMD
2. Enable mesh preview rendering
3. Enable GLB export with textures
4. Maintain stability (no crashes)

## Options Analysis

### Option 1: Fix HIP Rasterizer

**Effort:** 2-4 weeks
**Success Probability:** 60-70%
**Outcome:** Full native performance

#### Steps

1. **Fix warp intrinsic macros** in `RasterImpl_kernel.hip`:
   ```cpp
   // Replace:
   #define __syncwarp(...) __threadfence_block()

   // With:
   #define __syncwarp(...) __builtin_amdgcn_wave_barrier()
   ```

2. **Handle wave64 vs wave32**:
   ```cpp
   // Query wave size at runtime
   int waveSize = __builtin_amdgcn_wavefrontsize();

   // Use appropriate ballot intrinsic
   #if __AMDGCN_WAVEFRONT_SIZE == 64
   #define __ballot_sync(mask, pred) __ballot64(pred)
   #else
   #define __ballot_sync(mask, pred) __ballot(pred)
   #endif
   ```

3. **Fix shared memory sizing**:
   - Audit all `__shared__` declarations
   - Scale by wave size where needed

4. **Add debug instrumentation**:
   ```cpp
   #ifdef NVDIFFRAST_DEBUG_BOUNDS
   #define BOUNDS_CHECK(ptr, size, idx) \
       if ((idx) >= (size)) printf("OOB: %s[%d] >= %d\n", #ptr, idx, size)
   #else
   #define BOUNDS_CHECK(ptr, size, idx)
   #endif
   ```

5. **Test each kernel**:
   - Create standalone test for triangleSetupKernel
   - Create standalone test for binRasterKernel
   - Create standalone test for coarseRasterKernel
   - Create standalone test for fineRasterKernel

#### Pros
- Native GPU performance
- Full feature parity with NVIDIA

#### Cons
- Complex GPU debugging required
- May require deep HIP/RDNA expertise
- Time-consuming

---

### Option 2: CPU Rasterization Fallback

**Effort:** 1 week
**Success Probability:** 95%
**Outcome:** Slower but working

#### Steps

1. **Add CPU rasterizer using trimesh/PyTorch3D**:
   ```python
   # trellis/renderers/cpu_mesh_renderer.py
   import trimesh
   import numpy as np

   class CPUMeshRenderer:
       def render(self, mesh, camera, resolution):
           scene = trimesh.Scene(mesh)
           # Use pyrender or trimesh's built-in renderer
           image = scene.save_image(resolution=resolution)
           return image
   ```

2. **Modify MeshRenderer to use CPU on AMD**:
   ```python
   # trellis/renderers/mesh_renderer.py
   class MeshRenderer:
       def __init__(self):
           if _is_amd():
               self.renderer = CPUMeshRenderer()
           else:
               self.glctx = dr.RasterizeCudaContext()
   ```

3. **Add CPU texture baking**:
   ```python
   def bake_texture_cpu(vertices, faces, uvs, observations, ...):
       """Project observations to UV space using numpy"""
       texture = np.zeros((texture_size, texture_size, 3))
       for face_idx, face in enumerate(faces):
           # Get UV coordinates for face
           # Project observations onto UV space
           # Blend overlapping regions
       return texture
   ```

4. **Test integration**:
   - Verify mesh preview renders correctly
   - Verify GLB export produces valid files
   - Benchmark performance

#### Pros
- Guaranteed to work
- Simple implementation
- No HIP debugging needed

#### Cons
- Much slower (10-100x for rendering)
- May not support all nvdiffrast features
- Texture quality may differ

---

### Option 3: OpenGL via EGL (Headless)

**Effort:** 1-2 weeks
**Success Probability:** 75%
**Outcome:** Good performance, no HIP dependency

#### Steps

1. **Create EGL-based rasterizer**:
   ```python
   # nvdiffrast_egl/rasterize.py
   import OpenGL.EGL as egl
   import OpenGL.GL as gl

   class RasterizeEGLContext:
       def __init__(self, device=None):
           self.display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
           # Initialize EGL, create context
           # Compile GLSL shaders
   ```

2. **Implement core operations**:
   - `rasterize()` - GLSL fragment shader
   - `interpolate()` - Barycentric interpolation in shader
   - `antialias()` - MSAA or post-process AA
   - `texture()` - Standard GL texture sampling

3. **Integrate with TRELLIS**:
   ```python
   # Detect AMD and use EGL path
   if _is_amd():
       import nvdiffrast_egl as dr
   else:
       import nvdiffrast.torch as dr
   ```

#### Pros
- Works on any GPU with OpenGL 4.5
- No CUDA/HIP dependency
- Good performance

#### Cons
- Significant development effort
- May have edge cases
- OpenGL debugging can be tricky

---

### Option 4: Hybrid Approach (Recommended)

**Effort:** 1-2 weeks
**Success Probability:** 90%
**Outcome:** Working solution with path to optimization

#### Phase 1: Immediate (2-3 days)

1. **Add CPU mesh preview fallback**:
   ```python
   def render_video_cpu(mesh, num_frames, resolution):
       """CPU-based mesh rendering for AMD"""
       import trimesh
       # Create turntable animation
       # Render each frame
   ```

2. **Re-enable mesh generation**:
   - Mesh extraction doesn't use nvdiffrast
   - Only preview/export uses nvdiffrast
   - Can generate mesh, just can't preview it

#### Phase 2: Short-term (1 week)

3. **Implement CPU texture baking**:
   ```python
   def bake_texture_cpu(vertices, faces, uvs, observations):
       # Simple but functional texture projection
   ```

4. **Re-enable GLB export with CPU baking**:
   - Use CPU texture baking on AMD
   - Keep GPU baking on NVIDIA

#### Phase 3: Medium-term (2-4 weeks)

5. **Debug HIP rasterizer**:
   - Focus on fineRasterKernel first (most critical)
   - Add proper wave synchronization
   - Test incrementally

6. **Optimize CPU fallbacks**:
   - Use numba/cython for hot paths
   - Parallelize texture baking

---

## Recommended Implementation Order

```
Week 1:
├── Day 1-2: CPU mesh preview (trimesh)
├── Day 3-4: CPU texture baking
└── Day 5: Re-enable GLB export with CPU path

Week 2:
├── Day 1-2: Test and fix edge cases
├── Day 3-5: Start HIP rasterizer debugging
└── Document findings

Week 3-4:
├── Continue HIP debugging
├── Fix warp intrinsics
└── Incremental testing
```

## Success Criteria

- [ ] Mesh generation works on AMD (no nvdiffrast needed)
- [ ] Mesh preview renders (CPU fallback)
- [ ] GLB export produces valid files (CPU texture baking)
- [ ] No crashes or hangs
- [ ] Reasonable performance (<5 min total generation)

## Files to Modify

| File | Changes |
|------|---------|
| `app.py` | Re-enable mesh generation, add CPU fallback switches |
| `trellis/utils/render_utils.py` | Add CPU rendering path |
| `trellis/utils/postprocessing_utils.py` | Add CPU texture baking |
| `trellis/renderers/mesh_renderer.py` | Add CPU renderer option |
| `extensions/nvdiffrast-hip/csrc/common/hipraster/impl/RasterImpl_kernel.hip` | Fix warp intrinsics (later) |

## Testing Plan

1. **Unit tests** for CPU fallbacks
2. **Integration test** full pipeline on AMD
3. **Quality test** compare output to NVIDIA reference
4. **Performance test** benchmark all stages

## Progress Tracking

| Task | Status | Date | Notes |
|------|--------|------|-------|
| Create planning docs | Done | 2024-12-30 | This document |
| CPU mesh preview | TODO | | |
| CPU texture baking | TODO | | |
| Re-enable GLB export | TODO | | |
| HIP rasterizer debug | TODO | | |
