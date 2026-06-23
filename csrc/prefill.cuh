#pragma once
#include <torch/extension.h>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include "common.cuh"
using namespace nvcuda;

__global__ void prefill_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    int Sq, int Skv, int d, int BLOCK_N) {

    const int bh   = blockIdx.x;
    const int q0   = blockIdx.y * 16;
    const int lane = threadIdx.x;
    const float scale = rsqrtf((float)d);

    extern __shared__ char smem[];
    __nv_bfloat16* Qs = (__nv_bfloat16*)smem;
    __nv_bfloat16* Ks = Qs + 16 * d;
    __nv_bfloat16* Vs = Ks + BLOCK_N * d;
    __nv_bfloat16* Ps = Vs + BLOCK_N * d;
    float* Ss  = (float*)(Ps + 16 * BLOCK_N);
    float* m   = Ss + 16 * BLOCK_N;
    float* l   = m + 16;
    float* Acc = l + 16;

    for (int e = lane; e < 16 * d; e += 32) {
        int r = e / d, c = e % d, qr = q0 + r;
        Qs[e] = (qr < Sq) ? q[(size_t)(bh * Sq + qr) * d + c] : __float2bfloat16(0.f);
    }

    for (int r = lane; r < 16; r += 32) { m[r] = -INFINITY; l[r] = 0.f; }
    for (int e = lane; e < 16 * d; e += 32) Acc[e] = 0.f;
    __syncthreads();

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
