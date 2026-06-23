import pytest
import torch

from fused_attention import fused_attention, attention_reference
from bench.shapes import ALL_SHAPES, BATCH, HEADS

@pytest.mark.parametrize("Sq,Skv,d", ALL_SHAPES)
def test_matches_reference(Sq, Skv, d):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(0)
    q = torch.randn(BATCH, HEADS, Sq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(BATCH, HEADS, Skv, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(BATCH, HEADS, Skv, d, device="cuda", dtype=torch.bfloat16)
    ref = attention_reference(q, k, v)
    out = fused_attention(q, k, v)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=0)
