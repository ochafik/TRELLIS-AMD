# torchsparse Extension Build

## Overview

torchsparse provides efficient sparse tensor operations for 3D point cloud and voxel data.

## Key Files

- `extensions/torchsparse/setup.py` - Build configuration
- `extensions/torchsparse/torchsparse/backend/` - Backend implementations
- `extensions/torchsparse/third_party/sparsehash/` - Google sparsehash dependency

## Dependencies

### sparsehash-c11

torchsparse requires Google's sparsehash for hash map implementations.

```bash
cd extensions/torchsparse/third_party
git clone https://github.com/sparsehash/sparsehash-c11.git sparsehash
```

### Include Path Compatibility

The code expects `#include <google/dense_hash_map>` but sparsehash-c11 uses `sparsehash/dense_hash_map`.

**Solution**: Create compatibility headers in `third_party/sparsehash/sparsehash/google/`:

```cpp
// google/dense_hash_map
#include <sparsehash/dense_hash_map>
```

Setup.py must include both directories:
```python
sparsehash_base = os.path.join(base_dir, "third_party", "sparsehash")
sparsehash_include = os.path.join(sparsehash_base, "sparsehash")
include_dirs.append(sparsehash_base)      # For sparsehash/dense_hash_map
include_dirs.append(sparsehash_include)   # For google/dense_hash_map
```

## Fixes Applied

### 1. ROCm Detection Without CUDA_HOME

On ROCm, `CUDA_HOME` is None. Fixed CUDA detection:

```python
# Before:
if (torch.cuda.is_available() and CUDA_HOME is not None) or ...

# After:
if (torch.cuda.is_available() and (CUDA_HOME is not None or is_rocm)) or ...
```

### 2. Type Mismatch: long vs int64_t

On Windows, `long` is 32-bit but PyTorch's `ScalarType::Long` uses 64-bit.

**File**: `torchsparse/backend/others/downsample_cuda.cu`

```cpp
// Before:
_out_coords_transformed.data_ptr<long>()

// After:
_out_coords_transformed.data_ptr<int64_t>()
```

### 3. Excluded Files with PTX Assembly

Some files use CUDA-specific PTX inline assembly that won't work with HIP:

```python
HIP_EXCLUDED_FILES = [
    'implicit_gemm',    # Tensor core MMA ops
    'fetch_on_demand',  # PTX cvta.to.shared
]
```

### 4. HipFixBuildExtension

Custom build extension that fixes hipified files:
- Corrects `hipLaunchKernelGGL` syntax
- Injects HIP-compatible `atomicAdd` for `half2` types

## Build Command

```bash
cd extensions/torchsparse
pip install -e . --no-build-isolation
```

## Verification

```python
import torchsparse
print("torchsparse:", torchsparse.__version__)
```
