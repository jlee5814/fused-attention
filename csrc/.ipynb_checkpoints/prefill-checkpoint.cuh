#pragma once
#include <torch/extension.h>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "common.cuh"

__global__ void prefill_kernel(
  const __nv_bfloat16* __restrict__ q,
  const __nv_bfloat16* __restrict__ k,
  const __nv_bfloat16* __restrict__ v,
  __nv_bfloat16* __restrict__ out,
  int Sq, int Skv, int d, int BLOCK_N) {
  
    const int bh    = blockIdx.x;
    const int lane  = threadIdx.x;
    const int row   = threadIdx.y;
    const int BM    = blockDim.y;
    const int q_idx = blockIdx.y * BM + row;
    const int dpl   = d / 32; 
    
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* Ks = smem;
    __nv_bfloat16* Vs = smem + BLOCK_N * d;

    float qreg[4] = {0.f, 0.f, 0.f, 0.f};
    if (q_idx < Sq) {
      const __nv_bfloat16* qx = q + (size_t)(bh * Sq + q_idx) * d;
#pragma unroll
      for (int r = 0; r < dpl; ++r)
        qreg[r] = __bfloat162float(qx[lane + r * 32]);
    }

    float m = -INFINITY, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};

    const __nv_bfloat16* kx = k + (size_t)bh * Skv * d;
    const __nv_bfloat16* vx = v + (size_t)bh * Skv * d;

    const int tid       = row * 32 + lane;
    const int nthreads  = BM * 32;

    for (int kv0 = 0; kv0 < Skv; kv0 += BLOCK_N) {
      
      const int tile_keys = min(BLOCK_N, Skv - kv0);
      
      for (int e = tid; e < tile_keys * d; e += nthreads) {
        Ks[e] = kx[(size_t)kv0 * d + e];
        Vs[e] = vx[(size_t)kv0 * d + e];
      }
      
      __syncthreads();

      for (int kk = 0; kk < tile_keys; ++kk) {
        const __nv_bfloat16* ks = Ks + kk * d;
        const __nv_bfloat16* vs = Vs + kk * d;

        float part = 0.f;
#pragma unroll
        for (int r = 0; r < dpl; ++r)
          part += qreg[r] * __bfloat162float(ks[lane + r * 32]);
        float s = warp_reduce_sum(part) * rsqrtf((float)d);

        float m_new = fmaxf(m, s);
        float alpha = expf(m - m_new);
        float p     = expf(s - m_new);
        l = l * alpha + p;
#pragma unroll 
        for (int r = 0; r < dpl; ++r)
          acc[r] = acc[r] * alpha + p * __bfloat162float(vs[lane + r * 32]);
        m = m_new;
      }

      __syncthreads();

    }

    if (q_idx < Sq) {
      __nv_bfloat16* ox = out + (size_t)(bh * Sq + q_idx) * d;
#pragma unroll
      for (int r = 0; r < dpl; ++r)
        ox[lane + r * 32] = __float2bfloat16(acc[r] / l);
    }
}

inline void prefill_tile(int& BM, int& BN) {
  BM = 16; BN = 64;
  if (const char* e = std::getenv("PREFILL_BLOCK_M")) { int x = std::atoi(e); if (x > 0) BM = x; }
  if (const char* e = std::getenv("PREFILL_BLOCK_N")) { int x = std::atoi(e); if (x > 0) BN = x; }
  if (BM > 32) BM = 32;
}

inline void launch_prefill_attention(
  const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
  int B, int H, int Sq, int Skv, int d) {
  int BM, BN;
  prefill_tile(BM, BN);
  
  const int bh = B * H;
  const int num_qtiles = (Sq + BM - 1) / BM;

  dim3 grid(bh, num_qtiles);
  dim3 block(32, BM);
  size_t shmem = (size_t)2 * BN * d * sizeof(__nv_bfloat16);

  if (shmem > 48 * 1024)
     cudaFuncSetAttribute(prefill_kernel,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);

  const auto* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  const auto* kp = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  const auto* vp = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

  prefill_kernel<<<grid, block, shmem>>>(qp, kp, vp, op, Sq, Skv, d, BN);
    
}
