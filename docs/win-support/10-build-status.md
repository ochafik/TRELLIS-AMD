# Build Status Summary

Last Updated: 2025-12-29 (v3 - Full pipeline including rendering working!)

## Extension Build Status

| Extension | Status | Notes |
|-----------|--------|-------|
| nvdiffrast-hip | Built | Differentiable rasterization |
| diff-gaussian-rasterization | Built | 3D Gaussian splatting |
| torchsparse | Built | Sparse tensor operations |
| spconv | Not Built | Using torchsparse as alternative |

## Runtime Status

| Component | Status | Notes |
|-----------|--------|-------|
| PyTorch ROCm | Working | 2.11.0a0+rocm7.11.0a20251218 |
| Attention Backend | Working | Using SDPA (auto-detected) |
| Sparse Backend | **Working** | torchsparse with hashmap mode + warmup |
| Gradio | Working | With utils.py patch |

## TRELLIS Pipeline Status

**FULLY WORKING** as of 2025-12-29!

The full image-to-3D pipeline now runs successfully on AMD GPUs:
- sparse_structure_flow sampling: Working
- slat_flow sampling: Working
- Gaussian output generation: Working

## Fixes Applied

### 1. GatherScatter Dataflow (conv_config.py)
ImplicitGEMM uses PTX assembly which doesn't work on HIP. Modified to auto-detect ROCm and use GatherScatter dataflow.

### 2. GPU Architecture Targeting
Set `GPU_ARCH=gfx1151` and `PYTORCH_ROCM_ARCH=gfx1151` before building for AMD Radeon 8060S.

### 3. Hashmap Mode (TorchSparse Issue #347)
The default `hashmap_on_the_fly` mode causes kernel crashes on AMD GPUs. Applied workaround to use `hashmap` mode instead.

```python
import torchsparse.nn.functional as F
F.set_kmap_mode("hashmap")
_config = F.conv_config.get_default_conv_config()
_config.kmap_mode = "hashmap"
F.conv_config.set_global_conv_config(_config)
```

### 4. ROCm Warmup (Critical Fix)
On AMD ROCm, torchsparse HIP kernels need to be initialized before running DINOv2 image encoding. Without warmup, kernels fail with `hipErrorLaunchFailure` after the image encoder runs.

The fix runs a small warmup convolution before `get_cond()`:

```python
from trellis.modules import sparse as sp
sp.warmup_rocm_sparse()  # Call before get_cond()
```

This is now automatically called in `TrellisImageTo3DPipeline.get_cond()`.

### 5. GPU Architecture (Critical Fix)
Extensions must be compiled for the correct GPU architecture. The AMD Radeon 8060S uses `gfx1151` (RDNA 3.5), but the default fallback is `gfx1100` (RDNA 3).

**Symptom**: Access violation (0xC0000005) in `hipLaunchKernel()` during rendering.

**Fix**: Set `GPU_ARCH` environment variable before building:

```python
# In Python before building:
import os
os.environ['GPU_ARCH'] = 'gfx1151'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1151'
```

Or use the rebuild script:
```bash
python rebuild_extensions.py
```

## Environment

- **Windows Version**: Windows 11
- **Python**: 3.12
- **GPU**: AMD Radeon 8060S (gfx1151)
- **ROCm**: 7.11.0a nightly (TheRock build)
- **TheRock Path**: C:\TheRock\build

## Rebuild Commands

```powershell
# Rebuild torchsparse with correct GPU arch
$env:GPU_ARCH = "gfx1151"
$env:PYTORCH_ROCM_ARCH = "gfx1151"
cd C:\Dev\TRELLIS-AMD\extensions\torchsparse
pip install -e . --no-build-isolation -v
```

## Quick Test

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"HIP: {torch.version.hip}")
print(f"CUDA available: {torch.cuda.is_available()}")

import nvdiffrast.torch as dr
print("nvdiffrast-hip: OK")

import diff_gaussian_rasterization
print("diff-gaussian-rasterization: OK")

import torchsparse
print(f"torchsparse: {torchsparse.__version__}")

# Test basic sparse conv
from torchsparse import SparseTensor
from torchsparse.nn import Conv3d as TSConv3d

coords = torch.randint(0, 16, (100, 4), dtype=torch.int32).cuda()
coords[:, 0] = 0
feats = torch.randn(100, 32).cuda()
st = SparseTensor(feats=feats, coords=coords)
conv = TSConv3d(32, 64, kernel_size=3, stride=1).cuda()
out = conv(st)
print(f"torchsparse Conv3d: OK (output: {out.feats.shape})")
```

## Full Pipeline Test

```python
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline

pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

image = Image.open("assets/example_image/T.png")
outputs = pipeline.run(
    image,
    seed=42,
    formats=["gaussian"],
    sparse_structure_sampler_params={"steps": 4},
    slat_sampler_params={"steps": 4},
)
print(f"Success! Output keys: {list(outputs.keys())}")
```
