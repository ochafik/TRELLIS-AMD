# Attention Backend Configuration

## Problem: flash_attn Not Available on AMD

The `flash_attn` library is CUDA-only and doesn't work on AMD GPUs. TRELLIS defaults to flash_attn for attention operations.

## Solution: Use SDPA Backend

PyTorch provides `scaled_dot_product_attention` (SDPA) which works on all backends including ROCm.

### Automatic Detection

Modified attention modules to auto-detect ROCm:

**File**: `trellis/modules/attention/__init__.py`
```python
import torch

# Detect ROCm/AMD - flash_attn doesn't work on AMD, use sdpa instead
_is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
BACKEND = 'sdpa' if _is_rocm else 'flash_attn'
```

**File**: `trellis/modules/sparse/__init__.py`
```python
import torch

_is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
ATTN = 'sdpa' if _is_rocm else 'flash_attn'
```

### Manual Override

Users can still override via environment variables:

```bash
# Use specific backend
set ATTN_BACKEND=sdpa
set SPARSE_ATTN_BACKEND=sdpa

# Or via Python
import os
os.environ['ATTN_BACKEND'] = 'sdpa'
```

## Available Backends

| Backend | Description | AMD Support |
|---------|-------------|-------------|
| `flash_attn` | FlashAttention library | No |
| `xformers` | Meta's xformers | No (CUDA only) |
| `sdpa` | PyTorch native SDPA | Yes |
| `naive` | Reference implementation | Yes |

## Performance Notes

- `sdpa` uses PyTorch's optimized attention, with Flash Attention-like optimizations when available
- On ROCm, PyTorch may use hipBLAS-based attention or memory-efficient implementations
- For production, test both `sdpa` and `naive` to compare performance on your specific GPU
