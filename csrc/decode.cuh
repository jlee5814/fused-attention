#pragma once
#include <torch/extension.h>
#include <cstdlib>
#include <cuda_bf16.h>
#include "common.cuh"

// Split-KV decode path (S_q == 1). Memory-bound, ~1 FLOP/byte.
__global__ void decode_partials_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    float* __restrict__ m_out,
    float* __restrict__ l_out,
    float* __restrict__ acc_out,
    int H, int Skv, int d, int num_splits) {
    
    const int bh    = blockIdx.x;
    const int split = blockIdx.y;
    const int lane  = threadIdx.x;

    const int chunk = (Skv + num_splits - 1) / num_splits;
    const int lo    = split * chunk;
    const int hi    = min(lo + chunk, Skv);

    const int dpl = d / 32;
    const __nv_bfloat16* qx = q + (size_t)bh * d;
    float qreg[4];

#pragma unroll
    for (int r = 0; r < dpl; ++r)
      qreg[r] = __bfloat162float(qx[lane + r * 32]);

    // Phase A
    const __nv_bfloat16* kx = k + (size_t)bh * Skv * d;
    const __nv_bfloat16* vx = v + (size_t)bh * Skv * d;

    float m = -INFINITY, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int j = lo; j < hi; ++j) {
      const __nv_bfloat16* kj = kx + (size_t)j * d;
      const __nv_bfloat16* vj = vx + (size_t)j * d;
    
      float part = 0.f;

#pragma unroll
      for (int r = 0; r < dpl; ++r)
        part += qreg[r] * __bfloat162float(kj[lane + r * 32]);
      float s = warp_reduce_sum(part) * rsqrtf((float)d);

      float m_new = fmaxf(m, s);
      float alpha = expf(m - m_new);
      float p     = expf(s - m_new);
      l = l * alpha + p;

#pragma unroll
      for (int r = 0; r < dpl; ++r)
        acc[r] = acc[r] * alpha + p * __bfloat162float(vj[lane + r * 32]);
      m = m_new;
    }

    const int slot = bh * num_splits + split;

    if (lane == 0) { m_out[slot] = m; l_out[slot] = l; }
    float* acc_slot = acc_out + (size_t)slot * d;

#pragma unroll
    for (int r = 0; r < dpl; ++r)
      acc_slot[lane + r * 32] = acc[r];

}

__global__ void decode_combine_kernel(
    const float* __restrict__ m_in,
    const float* __restrict__ l_in,
    const float* __restrict__ acc_in,
    __nv_bfloat16* __restrict__ out,
    int H, int d, int num_splits) {
    
    const int bh   = blockIdx.x;
    const int base = bh * num_splits;

    float M = -INFINITY;
    for (int i = 0; i < num_splits; ++i)
        M = fmaxf(M, m_in[base + i]);

    float L = 0.f;
      for (int i = 0; i < num_splits; ++i)
        L += l_in[base + i] * expf(m_in[base + i] - M);

    for (int t = threadIdx.x; t < d; t += blockDim.x) {
      float O = 0.f;
      for (int i = 0; i <num_splits; ++i)
        O += acc_in[(size_t)(base + i) * d + t] * expf(m_in[base + i] - M);
      out[(size_t)bh * d + t] = __float2bfloat16(O / L);
    }
}

inline int decode_num_splits(int bh, int Skv) {
  if (const char* e = std::getenv("DECODE_NUM_SPLITS")) {
    int v = std::atoi(e);
    if (v > 0) return v;
  }
  const int kWarpsPerSMTarget = 24;
  const int kNumSMs = 108;
  const int kMinChunk = 128;
  int by_fill = (kWarpsPerSMTarget * kNumSMs + bh - 1) / bh;
  int cap = Skv / kMinChunk; if (cap < 1) cap = 1;
  int n = by_fill < 1 ? 1 : by_fill;
  if (n > cap) n = cap;
  return n;
}

inline void launch_decode_attention(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
    int B, int H, int Skv, int d) {
  const int bh = B * H;
  const int num_splits = decode_num_splits(bh, Skv);

  auto fopts = q.options().dtype(at::kFloat);
  auto m_scratch    = at::empty({B, H, num_splits},    fopts);
  auto l_scratch    = at::empty({B, H, num_splits},    fopts);
  auto acc_scratch  = at::empty({B, H, num_splits, d}, fopts);

  const auto* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  const auto* kp = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  const auto* vp = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  float* mp = m_scratch.data_ptr<float>();
  float* lp = l_scratch.data_ptr<float>();
  float* ap = acc_scratch.data_ptr<float>();
  
  // Phase A 
  dim3 gridA(bh, num_splits);
  dim3 blockA(32);
  decode_partials_kernel<<<gridA, blockA>>>(qp, kp, vp, mp, lp, ap, H, Skv, d, num_splits);

  // Phase B 
  dim3 gridB(bh);
  dim3 blockB(128);
  decode_combine_kernel<<<gridB, blockB>>>(mp, lp, ap, op, H, d, num_splits);

}
