# spconv - Work in Progress

## Current Status: NOT YET BUILT

The application crashes at startup with:
```
ModuleNotFoundError: No module named 'spconv'
```

## What is spconv?

spconv (Sparse Convolution) is a library for efficient sparse convolutions on point clouds and voxel data. TRELLIS uses it for the 3D VAE decoder.

## Options

### Option 1: Build spconv for ROCm (Complex)

spconv has CUDA-specific code that needs porting:
- Sparse tensor operations
- Custom CUDA kernels
- Complex build system

This requires significant work similar to the other extensions.

### Option 2: Use torchsparse Backend

TRELLIS supports two sparse backends:
```python
BACKEND = 'spconv'      # Default
BACKEND = 'torchsparse' # Alternative
```

Set via environment variable:
```bash
set SPARSE_BACKEND=torchsparse
```

Or in code:
```python
import os
os.environ['SPARSE_BACKEND'] = 'torchsparse'
```

### Option 3: Modify Default for ROCm

Similar to attention backend, auto-detect ROCm and use torchsparse:

```python
_is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
BACKEND = 'torchsparse' if _is_rocm else 'spconv'
```

## Next Steps

1. Try using torchsparse backend instead of spconv
2. Verify all sparse operations work correctly with torchsparse
3. Document any differences in functionality or performance

## References

- spconv GitHub: https://github.com/traveller59/spconv
- torchsparse GitHub: https://github.com/mit-han-lab/torchsparse
