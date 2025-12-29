from typing import *
import torch

# Detect ROCm/AMD
# - flash_attn doesn't work on AMD, use sdpa instead
# - spconv not built for ROCm, use torchsparse instead
_is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None

BACKEND = 'torchsparse' if _is_rocm else 'spconv'
DEBUG = False
ATTN = 'sdpa' if _is_rocm else 'flash_attn'

def __from_env():
    import os

    global BACKEND
    global DEBUG
    global ATTN

    env_sparse_backend = os.environ.get('SPARSE_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn = os.environ.get('SPARSE_ATTN_BACKEND')
    if env_sparse_attn is None:
        env_sparse_attn = os.environ.get('ATTN_BACKEND')

    if env_sparse_backend is not None and env_sparse_backend in ['spconv', 'torchsparse']:
        BACKEND = env_sparse_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    if env_sparse_attn is not None and env_sparse_attn in ['xformers', 'flash_attn', 'sdpa', 'naive']:
        ATTN = env_sparse_attn

    print(f"[SPARSE] Backend: {BACKEND}, Attention: {ATTN}")

    # Apply torchsparse hashmap mode fix for AMD ROCm (Issue #347 workaround)
    # The default hashmap_on_the_fly mode causes kernel crashes on AMD GPUs
    if BACKEND == 'torchsparse' and _is_rocm:
        try:
            import torchsparse.nn.functional as F
            F.set_kmap_mode("hashmap")
            _config = F.conv_config.get_default_conv_config()
            _config.kmap_mode = "hashmap"
            F.conv_config.set_global_conv_config(_config)
            print(f"[SPARSE] Applied torchsparse hashmap mode fix for AMD")
        except Exception as e:
            print(f"[SPARSE] Warning: Could not apply hashmap mode fix: {e}")


# Warmup state for AMD ROCm
_rocm_warmup_done = False

def warmup_rocm_sparse():
    """
    Warmup torchsparse kernels for AMD ROCm.

    On AMD GPUs with ROCm, torchsparse HIP kernels need to be initialized
    before running certain PyTorch operations (like image encoding with DINOv2).
    Otherwise, the kernels fail with 'unspecified launch failure'.

    This function runs a small warmup convolution to initialize the kernels.
    Call this before get_cond() or any image preprocessing.
    """
    global _rocm_warmup_done

    if not _is_rocm or BACKEND != 'torchsparse' or _rocm_warmup_done:
        return

    try:
        from torchsparse import SparseTensor
        from torchsparse.nn import Conv3d as TSConv3d

        # Create small test tensors
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        coords = torch.randint(0, 16, (100, 4), dtype=torch.int32, device=device)
        coords[:, 0] = 0  # batch index
        feats = torch.randn(100, 32, dtype=torch.float16, device=device)

        # Run warmup convolution
        st = SparseTensor(feats=feats, coords=coords)
        conv = TSConv3d(32, 32, kernel_size=3, stride=1).to(device).half()
        out = conv(st)
        torch.cuda.synchronize()

        # Clean up
        del coords, feats, st, conv, out
        torch.cuda.empty_cache()

        _rocm_warmup_done = True
        print(f"[SPARSE] ROCm torchsparse warmup completed")
    except Exception as e:
        print(f"[SPARSE] Warning: ROCm warmup failed: {e}")
        

__from_env()
    

def set_backend(backend: Literal['spconv', 'torchsparse']):
    global BACKEND
    BACKEND = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

def set_attn(attn: Literal['xformers', 'flash_attn', 'sdpa', 'naive']):
    global ATTN
    ATTN = attn
    
    
import importlib

__attributes = {
    'SparseTensor': 'basic',
    'sparse_batch_broadcast': 'basic',
    'sparse_batch_op': 'basic',
    'sparse_cat': 'basic',
    'sparse_unbind': 'basic',
    'SparseGroupNorm': 'norm',
    'SparseLayerNorm': 'norm',
    'SparseGroupNorm32': 'norm',
    'SparseLayerNorm32': 'norm',
    'SparseReLU': 'nonlinearity',
    'SparseSiLU': 'nonlinearity',
    'SparseGELU': 'nonlinearity',
    'SparseActivation': 'nonlinearity',
    'SparseLinear': 'linear',
    'sparse_scaled_dot_product_attention': 'attention',
    'SerializeMode': 'attention',
    'sparse_serialized_scaled_dot_product_self_attention': 'attention',
    'sparse_windowed_scaled_dot_product_self_attention': 'attention',
    'SparseMultiHeadAttention': 'attention',
    'SparseConv3d': 'conv',
    'SparseInverseConv3d': 'conv',
    'SparseDownsample': 'spatial',
    'SparseUpsample': 'spatial',
    'SparseSubdivide' : 'spatial'
}

__submodules = ['transformer']

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    from .basic import *
    from .norm import *
    from .nonlinearity import *
    from .linear import *
    from .attention import *
    from .conv import *
    from .spatial import *
    import transformer
