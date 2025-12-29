# ABI Compatibility Between hipcc and MSVC

## Problem: Linker Errors with c10::ValueError

When mixing code compiled by hipcc (which uses clang) and MSVC, certain exception types cause linker errors:

```
error LNK2001: unresolved external symbol "class c10::ValueError"
```

### Root Cause

hipcc/clang and MSVC have different C++ ABIs. Exception classes with complex inheritance like `c10::ValueError` aren't binary compatible between the two compilers.

### Solution: STRIP_ERROR_MESSAGES Flag

PyTorch provides a `STRIP_ERROR_MESSAGES` preprocessor flag that replaces complex exception throwing with simpler alternatives.

```python
extra_compile_args = {
    "cxx": ["/O2", "-DSTRIP_ERROR_MESSAGES"],
    "nvcc": ["-O3", "-std=c++17", "-DSTRIP_ERROR_MESSAGES"],
}
```

### Implementation Notes

- Add to both `cxx` (MSVC) and `nvcc` (hipcc) flags
- This affects error messages in the compiled code but prevents ABI issues
- All extensions need this flag when building on Windows with ROCm

## Files That Need the Flag

- `extensions/nvdiffrast-hip/setup.py`
- `extensions/diff-gaussian-rasterization/setup.py`
- `extensions/torchsparse/setup.py`

## Header Isolation

Another approach to avoid ABI issues is to keep ROCm headers isolated from MSVC:

```python
# Add TheRock includes to nvcc_flags only (not include_dirs)
# This prevents MSVC from seeing ROCm headers
nvcc_flags = [
    f"-I{therock_include}",
    f"--offload-arch={gpu_arch}",
]
```

By adding `-I` paths only to `nvcc_flags` (hipcc), MSVC doesn't try to parse ROCm headers which can conflict with MSVC's standard headers.
