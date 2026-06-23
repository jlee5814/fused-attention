#pragma once
#include <cuda_bf16.h>

__device__ inline float warp_reduce_sum(float x) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    x += __shfl_xor_sync(0xffffffff, x, off);
  return x;
}
