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
    float* Ss   = (float*)(Ps + 16 * BLOCK_N);
    float* m    = Ss + 16 * BLOCK_N;
    float* l    = m + 16;
    float* Acc  = l + 16;
    float* Otmp = Acc + 16 * d; 

    for (int e = lane; e < 16 * d; e += 32) {
        int r = e / d, c = e % d, qr = q0 + r;
        Qs[e] = (qr < Sq) ? q[(size_t)(bh * Sq + qr) * d + c] : __float2bfloat16(0.f);
    }

    for (int r = lane; r < 16; r += 32) { m[r] = -INFINITY; l[r] = 0.f; }
    for (int e = lane; e < 16 * d; e += 32) Acc[e] = 0.f;
    __syncthreads();

    const int n_sub = BLOCK_N / 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;

    for (int kv0 = 0; kv0 < Skv; kv0 += BLOCK_N) {
        const int tile_keys = min(BLOCK_N, Skv - kv0);

        for (int e = lane; e < BLOCK_N * d; e += 32) {
            int kr = e / d;
            bool valid = kr < tile_keys;
            Ks[e] = valid ? k[(size_t)(bh * Skv + kv0 + kr) * d + (e % d)] : __float2bfloat16(0.f);
            Vs[e] = valid ? v[(size_t)(bh * Skv + kv0 + kr) * d + (e % d)] : __float2bfloat16(0.f);
        }

        __syncthreads();

        for (int sub = 0; sub < n_sub; ++sub) {
            wmma::fill_fragment(s_frag, 0.0f);
            for (int k0 = 0; k0 < d; k0 += 16) {
                wmma::load_matrix_sync(q_frag, Qs + k0,                d);
                wmma::load_matrix_sync(k_frag, Ks + sub * 16 * d + k0, d);
                wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
            }
            wmma::store_matrix_sync(Ss + sub * 16, s_frag, BLOCK_N, wmma::mem_row_major);
        }
        __syncthreads();

        if (lane < 16) {
            int r = lane;
            float* Srow = Ss + r * BLOCK_N;

            float m_prev = m[r];
            float m_cur  = m_prev;
            for (int j = 0; j < tile_keys; ++j) {
                float s = Srow[j] * scale;
                m_cur = fmaxf(m_cur, s);
            }

            float alpha = expf(m_prev - m_cur);

            float l_cur = l[r] * alpha;
            for (int j = 0; j < BLOCK_N; ++j) {
                float p = 0.f;
                if (j < tile_keys) {
                    p = expf(Srow[j] * scale - m_cur);
                    l_cur += p;
                }
                Ps[r * BLOCK_N + j] = __float2bfloat16(p);
            }

            for (int c = 0; c < d; ++c)
                Acc[r * d + c] *= alpha;

            m[r] = m_cur;
            l[r] = l_cur;

        }
    
        __syncthreads();

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag;

            for (int dt = 0; dt < d; dt += 16) {
                wmma::fill_fragment(o_frag, 0.0f);
                for (int kt = 0; kt < n_sub; ++kt) {
                    wmma::load_matrix_sync(p_frag, Ps + kt * 16,        BLOCK_N);
                    wmma::load_matrix_sync(v_frag, Vs + kt * 16 * d + dt, d);
                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                }
                wmma::store_matrix_sync(Otmp, o_frag, 16, wmma::mem_row_major);

                __syncthreads();
                for (int e = lane; e < 16 * 16; e += 32) {
                    int rr = e / 16, cc = e % 16;
                    Acc[rr * d + dt + cc] += Otmp[rr * 16 + cc];
                }
                __syncthreads();

            }

        }
    }

    for (int e = lane; e < 16 * d; e += 32) {
        int r = e / d, c = e % d, qr = q0 + r;
        if (qr < Sq)
            out[(size_t)(bh * Sq + qr) * d + c] = __float2bfloat16(Acc[r * d + c] / l[r]);
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
  const int num_qtiles = (Sq + 16 - 1) / 16;
  dim3 grid(bh, num_qtiles);
  dim3 block(32);

  size_t bf16_elems = (size_t)16 *d + 2*BN*d + 16*BN;
  size_t float_elems = (size_t)16*BN + 16 + 16 + 16*d + 16*16;
  size_t shmem = bf16_elems * 2 + float_elems * 4;

  if (shmem > 48 * 1024)
     cudaFuncSetAttribute(prefill_kernel,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
  const auto* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  const auto* kp = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  const auto* vp = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  prefill_kernel<<<grid, block, shmem>>>(qp, kp, vp, op, Sq, Skv, d, BN);
}
