# diff-gaussian-rasterization Extension Build

## Overview

This extension provides 3D Gaussian splatting rasterization for neural rendering.

## Key Files

- `extensions/diff-gaussian-rasterization/setup.py` - Build configuration
- `extensions/diff-gaussian-rasterization/cuda_rasterizer/` - Original CUDA sources
- `extensions/diff-gaussian-rasterization/hip_rasterizer/` - Pre-hipified sources

## Dependencies

The extension requires:
- **rocPRIM** - ROCm primitive library
- **hipCUB** - HIP-compatible CUB library
- **rocThrust** - ROCm thrust implementation

These are provided by TheRock installation at `C:\TheRock\build\include`.

## Fixes Applied

### 1. TheRock Integration

```python
therock_include = r"C:\TheRock\build\include"
if os.path.exists(therock_include):
    # Add to nvcc_flags only (not include_dirs to avoid MSVC conflicts)
    nvcc_flags.append(f"-I{therock_include}")
```

### 2. Kernel Launch Syntax Fix

PyTorch's hipify sometimes produces malformed kernel launches:
```cpp
// Bad:
kernel<< <grid, block>> >(...);

// Good:
hipLaunchKernelGGL(kernel, grid, block, 0, 0, ...);
```

The `fix_hipify_issues()` function in setup.py corrects these.

### 3. Extra Argument Removal

Hipify sometimes adds extra `0, 0` arguments:
```cpp
// Bad:
hipLaunchKernelGGL(kernel, grid, block, 0, 0, 0, 0, args...);

// Good:
hipLaunchKernelGGL(kernel, grid, block, 0, 0, args...);
```

Fixed with regex replacement:
```python
fixed = re.sub(r',\s*0,\s*0,\s*0,\s*0,', ', 0, 0,', content)
```

## Build Command

```bash
cd extensions/diff-gaussian-rasterization
pip install -e . --no-build-isolation
```

## Verification

```python
import diff_gaussian_rasterization
print("diff-gaussian-rasterization: OK")
```
