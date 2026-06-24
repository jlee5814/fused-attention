import torch

try:
    from . import _C
    _HAS_EXT = True

except ImportError:
    _C = None
    _HAS_EXT = False

def fused_attention(q, k, v):
    """
    q: (B, H, S_q, d), k/v: (B, H, S_kv, d), BF16 on CUDA. Returns (B, H, S_q, d) BF16.
    """
    if not _HAS_EXT:
        raise RuntimeError("CUDA extension not built.")
    return _C.fused_attention(q, k, v)
