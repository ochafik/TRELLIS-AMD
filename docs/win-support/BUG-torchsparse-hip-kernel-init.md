# Bug Report: TorchSparse HIP Kernel Initialization Failure After DINOv2

## Summary

On AMD ROCm, torchsparse HIP kernels fail with `hipErrorLaunchFailure` if they are first called AFTER running DINOv2 image encoding. However, if torchsparse kernels are "warmed up" (run once) BEFORE DINOv2, they work correctly.

## Environment

- **OS**: Windows 11 (WSL2 may have same issue)
- **GPU**: AMD Radeon 8060S (gfx1151 / RDNA 3.5)
- **ROCm**: 7.11.0a nightly (TheRock build for Windows)
- **PyTorch**: 2.11.0a0+rocm7.11.0a20251218
- **TorchSparse**: 2.1.0 (with HIP backend, built from source)
- **Python**: 3.12

## Error Message

```
torch.AcceleratorError: HIP error: unspecified launch failure
Search for `hipErrorLaunchFailure' in https://rocm.docs.amd.com/projects/HIP/en/latest/index.html
```

The error is reported at various points (async), typically at:
- `torch.sum(results != -1, dim=1)` in `hashmap.py:139`
- Or any subsequent GPU operation

## Reproduction Steps

### Minimal Reproduction

```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HIP_LAUNCH_BLOCKING'] = '1'

import torch
print(f"PyTorch: {torch.__version__}, HIP: {torch.version.hip}")

# Setup torchsparse with hashmap mode (required for AMD)
import torchsparse.nn.functional as F_ts
F_ts.set_kmap_mode("hashmap")
_config = F_ts.conv_config.get_default_conv_config()
_config.kmap_mode = "hashmap"
F_ts.conv_config.set_global_conv_config(_config)

from torchsparse import SparseTensor
from torchsparse.nn import Conv3d as TSConv3d

def test_conv():
    """Run a simple torchsparse convolution"""
    coords = torch.randint(0, 64, (3686, 4), dtype=torch.int32).cuda()
    coords[:, 0] = 0
    feats = torch.randn(3686, 128, dtype=torch.float16).cuda()
    st = SparseTensor(feats=feats, coords=coords)
    conv = TSConv3d(128, 128, kernel_size=3, stride=1).cuda().half()
    out = conv(st)
    torch.cuda.synchronize()
    return out.feats.shape

# Load DINOv2 model (like TRELLIS does)
print("Loading DINOv2...")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)
dinov2.eval().cuda()

# Create dummy image input
image = torch.randn(1, 3, 518, 518).cuda()

print("\n[Test 1] Conv BEFORE DINOv2 forward...")
try:
    result = test_conv()
    print(f"  SUCCESS: {result}")
except Exception as e:
    print(f"  FAILED: {e}")

print("\n[Test 2] Running DINOv2 forward...")
with torch.no_grad():
    features = dinov2(image, is_training=True)['x_prenorm']
    torch.cuda.synchronize()
print("  DINOv2 completed")

print("\n[Test 3] Conv AFTER DINOv2 forward...")
try:
    result = test_conv()
    print(f"  SUCCESS: {result}")
except Exception as e:
    print(f"  FAILED: {e}")
```

### Expected Result

All three tests should pass.

### Actual Result

- Test 1: SUCCESS (conv before DINOv2)
- Test 2: DINOv2 completed
- Test 3: FAILED with `hipErrorLaunchFailure`

### Workaround

Running a torchsparse convolution BEFORE DINOv2 makes subsequent convolutions work:

```python
# Warmup BEFORE DINOv2
test_conv()  # This "primes" the HIP kernels

# Now DINOv2
with torch.no_grad():
    features = dinov2(image)

# Now this works!
test_conv()  # SUCCESS
```

## Analysis

### What We Tested

1. **Memory pressure**: Not the cause. ~5.5GB of model weights loaded, 88GB total GPU memory available.

2. **Coordinate patterns**: Not the cause. Same coordinates work in isolation.

3. **Float16 vs Float32**: Not the cause. Both fail after DINOv2.

4. **Hashmap mode**: Required for AMD (default `hashmap_on_the_fly` crashes), but doesn't fix this issue.

5. **Memory cleanup**: `torch.cuda.empty_cache()` doesn't help.

6. **Order of operations**: The critical factor. DINOv2 must run AFTER torchsparse warmup.

### Hypothesis

The torchsparse HIP kernels have some lazy initialization (JIT compilation, memory allocation, or device state setup) that gets corrupted or preempted when DINOv2 runs first. Once torchsparse kernels are initialized (warmup), they remain functional even after DINOv2 runs.

Possible causes:
1. HIP JIT compilation conflict between DINOv2 attention kernels and torchsparse kernels
2. Device state (stream, context) corruption
3. Memory allocator state issue
4. ROCm driver bug specific to gfx1151

### Affected Kernel

The crash occurs in `build_mask_from_kmap` or related torchsparse backend functions:

```
File ".../torchsparse/nn/functional/conv/kmap/func/hashmap.py", line 139
    nbsizes = torch.sum(results != -1, dim=1)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.AcceleratorError: HIP error: unspecified launch failure
```

The actual failing kernel is likely in the hashmap lookup or mask building:
- `extensions/torchsparse/torchsparse/backend/others/sparsemapping_hip.hip`
- Functions: `get_masks_from_kmap_kernel`, `build_kmap_Gather_Scatter_hashmap`

## Files Changed for Workaround

1. `trellis/modules/sparse/__init__.py` - Added `warmup_rocm_sparse()` function
2. `trellis/pipelines/trellis_image_to_3d.py` - Call warmup before `get_cond()`

## Related Issues

- TorchSparse Issue #347: hashmap_on_the_fly crashes on AMD (different issue, but related)
- This appears to be a new/different issue specific to kernel initialization order

## Potential Fix Locations

1. **TorchSparse**: Add explicit kernel initialization on first use
2. **ROCm/HIP**: Investigate JIT compilation conflicts
3. **PyTorch ROCm**: Check if device state is properly preserved across different kernel types

## Debug Scripts

The following debug scripts in `C:\Dev\TRELLIS-AMD\` were used to isolate this issue:

- `debug_get_cond_trigger.py` - Shows conv works before/after get_cond if warmup done first
- `debug_bincount_trigger.py` - Rules out torch.bincount as cause
- `debug_warmup_helps.py` - **Best reproduction** - Confirms warmup fixes the issue
- `debug_order_matters.py` - Shows order of operations matters
- `debug_trellis_wrapper.py` - Rules out TRELLIS wrapper as cause
- `bug_repro_minimal.py` - Standalone reproduction (requires TRELLIS imports for proper init)

### Best Reproduction Script

Run `debug_warmup_helps.py` for the clearest demonstration:

```bash
cd C:\Dev\TRELLIS-AMD
.\.venv\Scripts\python.exe debug_warmup_helps.py
```

Expected output:
- WARMUP SUCCESS (conv before get_cond works)
- get_cond completes
- Conv AFTER get_cond: SUCCESS (because warmup was done)
- Full slat sampling: SUCCESS

To see the bug (without warmup), run `debug_full_pipeline.py` which calls get_cond before any torchsparse operations and fails.
