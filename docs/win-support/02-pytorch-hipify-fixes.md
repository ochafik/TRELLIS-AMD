# PyTorch Hipify Fixes for Windows

## Problem: Path Mangling with Spaces

PyTorch's `torch.utils.cpp_extension` has a bug on Windows where paths containing spaces get mangled incorrectly.

### Root Cause

In `torch/utils/cpp_extension.py` around line 1032, the `_nt_quote_args` function replaces spaces in `-I` paths with backslashes instead of properly quoting them.

Example:
```
-IC:\Users\Olivier Chafik\...
becomes:
-IC:\Users\Olivier\Chafik\...  (WRONG!)
```

### Solution: Patch `_nt_quote_args`

Each extension's `setup.py` includes a patch that:

1. Detects mangled paths (paths that don't exist but would if spaces were restored)
2. Converts paths with spaces to Windows 8.3 short format (e.g., `C:\Users\OLIVIE~1\...`)
3. Applies the fix before PyTorch's quoting

```python
import ctypes
from ctypes import wintypes
from torch.utils import cpp_extension as cpp_ext

def _get_short_path_win(path):
    """Convert path to Windows 8.3 short format."""
    if not path or not os.path.exists(path):
        return path
    try:
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        GetShortPathNameW.restype = wintypes.DWORD
        buf_size = GetShortPathNameW(path, None, 0)
        if buf_size == 0:
            return path
        buf = ctypes.create_unicode_buffer(buf_size)
        GetShortPathNameW(path, buf, buf_size)
        return buf.value
    except Exception:
        return path

def _try_unmangle_path(mangled_path):
    """Try to recover a path that had spaces replaced with backslashes."""
    if os.path.exists(mangled_path):
        return mangled_path
    # Try combinations of backslash vs space
    parts = mangled_path.split('\\')
    # ... recursive combination logic
    return mangled_path

_orig_nt_quote_args = cpp_ext._nt_quote_args
def _patched_nt_quote_args(args):
    if not args:
        return []
    fixed_args = [_fix_include_arg(arg) if arg.startswith('-I') else arg for arg in args]
    return _orig_nt_quote_args(fixed_args)
cpp_ext._nt_quote_args = _patched_nt_quote_args
```

## Problem: Wrong hipcc Path

PyTorch looks for `hipcc` in `bin/` but on Windows pip installs it to `Scripts/`.

### Solution: Patch `_get_hipcc_path`

```python
_orig_get_hipcc_path = cpp_ext._get_hipcc_path
def _patched_get_hipcc_path():
    path = _orig_get_hipcc_path()
    if '\\bin\\' in path:
        fixed = path.replace('\\bin\\', '\\Scripts\\')
        if os.path.exists(fixed):
            return fixed
    return path
cpp_ext._get_hipcc_path = _patched_get_hipcc_path
```

## Files Modified

- `extensions/nvdiffrast-hip/setup.py`
- `extensions/diff-gaussian-rasterization/setup.py`
- `extensions/torchsparse/setup.py`
