#include <torch/extension.h>
#include <cuda_bf16.h>
#include "decode.cuh"
#include "prefill.cuh"

at::Tensor fused_attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "inputs must be blfloat16");
  TORCH_CHECK(q.dim() == 4, "expected (B, H, S_q, d)");

  const int B   = q.size(0);
  const int H   = q.size(1);
  const int Sq  = q.size(2);
  const int d   = q.size(3);
  const int Skv = k.size(2);

  auto out = at::empty_like(q);

  if (Sq == 1) {
    launch_decode_attention(q, k, v, out, B, H, Skv, d);
  } else {
    launch_prefill_attention(q, k, v, out, B, H, Sq, Skv, d);
  }
  return out;
}
