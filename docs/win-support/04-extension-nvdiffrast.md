# nvdiffrast-hip Extension Build

## Overview

nvdiffrast is a differentiable rasterization library. The `nvdiffrast-hip` variant is a HIP port for AMD GPUs.

## Key Files

- `extensions/nvdiffrast-hip/setup.py` - Build configuration
- `extensions/nvdiffrast-hip/csrc/` - Source files

## Fixes Applied

### 1. Missing thrust/complex.h

The minimal ROCm SDK from pip doesn't include all thrust headers. Created a stub:

**File**: `csrc/thrust/complex.h`
```cpp
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <complex>

namespace thrust {
    template <typename T>
    using complex = std::complex<T>;
}
```

### 2. LOG Macro Conflict

`Defs.hpp` defined a `LOG` macro that conflicted with Windows headers.

**Fix**: Renamed to `CR_LOG_IMPL` with a wrapper macro:
```cpp
#ifndef LOG
#define LOG(...) CR_LOG_IMPL(__VA_ARGS__)
#endif
```

### 3. Narrowing Conversion

`torch_antialias_hip.hip` had `int64_t` narrowing issues.

**Fix**: Explicit casts to `int` where needed.

### 4. Library Path

The ROCm SDK library path needs to be added for linking:

```python
library_dirs = []
rocm_sdk_lib = os.path.join(sys.prefix, "Lib", "site-packages", "_rocm_sdk_core", "lib")
if os.path.exists(rocm_sdk_lib):
    library_dirs.append(rocm_sdk_lib)
```

## Build Command

```bash
cd extensions/nvdiffrast-hip
pip install -e . --no-build-isolation
```

## Verification

```python
import nvdiffrast.torch as dr
print("nvdiffrast-hip: OK")
```
