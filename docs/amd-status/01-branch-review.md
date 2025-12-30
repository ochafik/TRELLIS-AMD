# AMD Branch Review (amd-wsl2-fixes)

**Date:** 2024-12-30
**Status:** Working with limitations
**Branch:** `amd-wsl2-fixes`

## Summary

The `amd-wsl2-fixes` branch enables TRELLIS to run on AMD GPUs using ROCm/HIP. The port is functional for Gaussian splatting generation but mesh/GLB export is currently disabled due to nvdiffrast HIP crashes.

## Commit History (11 commits)

| Commit | Description | Status |
|--------|-------------|--------|
| `800c840` | Add huggingface_hub as explicit dependency | Good |
| `7b10529` | Fix paths with spaces and improve install robustness | Good |
| `db75cca` | diff-gaussian-rasterization: AMD/HIP fixes for Windows | Working |
| `be18754` | nvdiffrast-hip: AMD/HIP rasterizer fixes for Windows | Partial |
| `d75f310` | torchsparse: AMD/HIP fixes for Windows | Working |
| `fab938a` | TRELLIS pipeline: AMD GPU fixes for sparse, attention, dtype | Working |
| `37a39b0` | Add AMD Windows build scripts, documentation, and tests | Good |
| `3eea94b` | fix temp dir | Trivial |
| `c2a9b12` | AMD ROCm: Run DINOv2 on CPU to fix Tensile kernel bugs | Workaround |
| `609e857` | AMD HIP: Disable mesh/GLB features due to nvdiffrast crashes | Workaround |
| `8fa6c03` | AMD HIP: Lazy import nvdiffrast to prevent crashes | Workaround |

## Files Changed

- **163 files changed**
- **+68,720 lines added**
- **-290 lines removed**

### Key Modified Files

| File | Changes |
|------|---------|
| `app.py` | AMD detection, lazy imports, skip mesh features, gradio patches |
| `trellis/pipelines/trellis_image_to_3d.py` | DINOv2 CPU workaround, dtype fixes |
| `trellis/modules/sparse/__init__.py` | Torchsparse AMD configuration |
| `trellis/modules/attention/__init__.py` | SDPA backend selection |
| `trellis/utils/postprocessing_utils.py` | AMD workarounds for texture baking |
| `trellis/utils/render_utils.py` | Lazy MeshRenderer import |
| `extensions/nvdiffrast-hip/` | Full HIP port of nvdiffrast |
| `extensions/diff-gaussian-rasterization/` | HIP port of Gaussian rasterizer |
| `extensions/torchsparse/` | AMD/HIP fixes + sparsehash headers |

## What Works on AMD

- Image preprocessing (rembg background removal)
- DINOv2 image encoding (on CPU - slower but correct)
- Sparse structure sampling (12 steps)
- SLAT sampling (12 steps)
- Gaussian representation generation
- Gaussian splatting preview video (120 frames)
- Gaussian PLY export

## What Doesn't Work on AMD

| Feature | Issue | Workaround |
|---------|-------|------------|
| Mesh generation | nvdiffrast crashes | Skipped |
| Mesh normal preview | nvdiffrast segfault | Skipped |
| GLB export | nvdiffrast crash in texture baking | Disabled with error |
| fill_holes | Visibility check returns wrong data | Disabled |
| Optimized texture baking | dr.texture() crashes | Using 'fast' mode |

## Performance (AMD Radeon 8060S / gfx1151)

| Stage | Time | Notes |
|-------|------|-------|
| DINOv2 encoding | ~60s | On CPU (GPU has Tensile bugs) |
| SS Sampling (12 steps) | ~90s | ~7.5s/step |
| SLAT Sampling (12 steps) | ~85s | ~7s/step |
| Gaussian preview render | ~1.5s | 120 frames @ 80 fps |
| **Total generation** | ~4 min | |

## Known Issues

1. **DINOv2 on GPU produces wrong output** - Tensile GEMM kernel bug on gfx1151
2. **nvdiffrast HIP crashes** - Warp intrinsic compatibility issues
3. **fill_holes removes all faces** - OpenGL rasterizer visibility bug
4. **Verbose nvdiffrast logs** - Silenced with NVDIFFRAST_DEBUG=0

## Environment Variables

```bash
# Auto-set by app.py when ROCm detected:
SPARSE_BACKEND=torchsparse
ATTN_BACKEND=sdpa
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Optional:
TRELLIS_DETERMINISTIC=1  # Enable deterministic mode for debugging
```
