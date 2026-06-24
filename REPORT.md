# Online Softmax + Scaled Dot-Product Attention Report

## I. AI Usage Disclosure

Claude was used as a reviewer and sounding board throughout this project. All kernel code was written by me; no portion of the kernel implementation was AI-generated. Claude's role was to talk through conceptual foundations, review code for correctness, and to discuss technical design/hardware concepts.

## II. Development log

### Design & alternatives

I built a two-pass PyTorch reference first (scores materialized, then normalized), deliberately avoiding SDPA, so the fused kernel is validated against an independent implementation rather than another fused one. The fused kernel never materializes the score matrix in global memory: scores are produced per KV-tile, consumed immediately by the online softmax, and discarded. Decode and prefill are dispatched as separate kernels on S_q=1 as their bottlenecks differ (bandwidth vs compute). Optimizing a code path for one starves the other. For the tensor-core path, I chose the wmma over raw mma.sync PTX for a first correct version. I de-risked the mechanics in two standalone probe kernels: a basic 16x16x16 MMA, then a QKᵀ from shared kernel exercising the column-major-K transpose, before integrating. Validating each mechanic in isolation meant integration bugs had only one possible source.

### Tile/block configuration 

Decode: one warp (32 threads) per (batch head, KV-split) block, grid (BH, 16). The 16 splits were chosen by sweep as the bandwidth optimum, with lanes mapped across the head dimension for coalesced KV loads.

Prefill: one warp per 16-query-row tile, W=6 warps per block (32 W threads), grid (BH, [S_q/16]/W). The 16-row tile matches the 16x16x16 wmma shape; BLOCK_N=64 and W=6 were selected by sweep. W=6 maximizes occupancy within the A100’s 164 KB shared-memory budget, whereas W=8 would exceed it. Shared memory holds the reused K/V tile (loaded once per block, shared across warps) plus private per-warp softmax state.

### Debugging 

The dominant failure mode was silent correctness bugs that compile and run but corrupt output: index-arithmetic slips and wrong leading dimensions in fragment loads. The last is especially dangerous as a wrong stride reads the right amount of data with the wrong layout, producing plausible but incorrect numbers. The 1e-2 reference gate caught these at the shape level; code review caught many before compile. 

### Optimization iterations

**Decode: split-KV**

A single warp streaming the KV cache cannot generate enough outstanding memory requests to saturate HBM as bandwidth is only reached when many requests are in flight at once. Split-KV partitions the KV range across thread blocks. Each block computes a partial result (running max, sum, weighted-V) over its slice, and a second combine kernel merges the partials. This multiplies the number of concurrent load streams, hiding any single stream’s latency behind the others. A split-count sweep found a bandwidth optimum at 16 splits (0.05 -> 0.66 TB/s on 4096-key shape, 12x), plateauing at ~33% of peak.

**Prefill: tensor-core MMA with occupancy tuning**

A flat BLOCK_M x BLOCK_N tile sweep (0.7-1.8 TFLOP/s) first ruled out tiling as the limiter, redirecting to MMA. The matmuls were running on the wrong execution units. A scalar FMA issues one multiply-add per instruction on a CUDA core, whereas one MMA instruction computes a full 16x16x16 multiply-accumulate on the tensor cores (312 TFLOP/s vs 19.5 for FP32). Rewriting QKᵀ and PV as wmma operations was therefore the highest-leverage change. The first variant regressed to 0.59 TFLOP/s, slower than scalar: a single-warp-per-tile kernel occupies ~1/64 of an SM’s warp slots, so the tensor cores stalled on shared-memory latency with no resident warps to hide it. Throughput here is gated by occupancy, not by arithmetic capability. Restructuring to multiple resident warps per block (shared K/V, private per-warp softmax state) and sweeping the warp count reached 3.03 TFLOP/s at W=6 (1.75x over the scalar baseline), rising at every warp count tested.

Two further experiments bounded the remaining headroom: a softmax stub showed softmax is only ~20% of runtime, capping the value of overlapping it. Double-buffered K/V gained 22% at fixed occupancy (2.19 -> 2.68 at W=4) but lost to higher occupancy, as W=6 reaches 3.03. The second buffer consumes shared memory that would otherwise fund resident warps and on this SM occupancy is worth more than the latency-hiding it buys. Prefetch and occupancy compete for the same on-chip budget, a constraint a production kernel resolves by shrinking the per-warp footprint so both can coexist.

## III. Validation & Performance Results 

All ten required shapes pass against the two-pass PyTorch reference within 1e-2 absolute error. Benchmarks were run on an A100-SXM4-80GB; throughput is the dominant metric per regime, with achieved bandwidth for decode (memory-bound), estimated FLOP/s for prefill (compute-bound).

**Decode (S_q=1)**

| S_q | S_kv | d | latency (µs) | TFLOP/s | TB/s |
|----:|-----:|----:|-----:|-----:|-----:|
| 1 | 512 | 64 | 58.6 | 0.29 | 0.29 |
| 1 | 512 | 128 | 70.5 | 0.48 | 0.48 |
| 1 | 2048 | 64 | 350.1 | 0.19 | 0.19 |
| 1 | 2048 | 128 | 232.2 | 0.58 | 0.58 |
| 1 | 4096 | 128 | 453.5 | 0.59 | 0.59 |

**Chunked prefill (S_q << S_kv)**

| S_q | S_kv | d | latency (µs) | TFLOP/s | TB/s |
|----:|-----:|----:|-----:|-----:|-----:|
| 64 | 1024 | 64 | 1779.7 | 1.21 | 0.02 |
| 64 | 2048 | 128 | 5203.5 | 1.65 | 0.03 |
| 128 | 2048 | 64 | 4297.1 | 2.00 | 0.02 |
| 128 | 4096 | 128 | 14769.9 | 2.33 | 0.02 |
| 256 | 4096 | 128 | 22662.0 | 3.03 | 0.01 |

Decode achieves up to 0.59 TB/s (~30% of the 2.0 TB/s HBM peak); the bandwidth split between d=128 (0.48-0.59 TB/s) and d=64 (0.19-0.29) indicates sub-128-bit per-thread loads. Prefill reaches 3.03 TFLOP/s at the largest shape after the tensor-core rewrite, with ~1% of the 312 TFLOP/s tensor-core ceiling, utilization-limited rather than throughput-limited. 

## IV. Analysis

### Arithmetic intensity for decode vs. chunked prefill

These are the two inference-serving phases: prefill processes the prompt to build the KV cache, decode then generates tokens autoregressively. The two regimes hit different hardware limits because of reuse. Decode has a single query, streaming the entire KV cache, giving ~1 FLOP/byte, with each byte used once, no reuse. Prefill reuses each loaded K/V tile across all S_q rows, so arithmetic intensity scales with the query-tile size. Decode is intrinsically bandwidth-bound and since it must stream the whole cache per token, it is the throughput-critical regime for serving, whereas prefill is compute-bound.

### Roofline positioning

The A100's ridge (2.0 TB/s vs 19.5 TFLOP/s FP32) is ~9.8 FLOP/byte. Decode at ~1 FLOP/byte sits far left, pinned against the bandwidth ceiling. Best achieved bandwidth is ~0.59 TB/s, which is only 30% of peak, confirming it is bandwidth-bound and under-utilizing the available bandwidth. Prefill sits right of the ridge (compute-bound) but reaches only 3.03 TFLOP/s, which is 15% of the CUDA-core ceiling and ~1% of the 312 TFLOP/s tensor-core ceiling. It’s limited by unit utilization, not available compute.

### Dominant bottlenecks

The dominant bottleneck is memory bandwidth; with a single query there is no data reuse, so runtime is bounded by KV_bytes / achieved_bandwidth. Since the bytes moved are intrinsic, the only lever is raising achieved bandwidth toward the 2.0 TB/s ceiling. Two factors cap it. First, insufficient memory-level parallelism: a single warp streaming KV serially does not keep enough requests in flight (addressed by split-KV). Second, narrow memory transactions: d=64 hits 0.19 TB/s vs d=128’s 0.59 TB/s, indicating sub-128-bit loads.

### Tiling strategy discussion

Both kernels tile over the KV sequence, but for opposite reasons dictated by their bottlenecks. Decode partitions the KV range across separate thread blocks (split-KV) to expose parallelism and saturate bandwidth. With only one query there is nothing to reuse, so tiling exists purely for concurrency, with partials combined afterwards. Prefill tiles query rows and KV within a block, staging each K/V tile in shared memory for reuse across all BLOCK_M rows. Tiling exists for locality to raise arithmetic intensity and keep the compute units fed. Decode spreads tiles across blocks to saturate memory bandwidth; prefill keeps tiles local to a block to maximize reuse and feed compute.

### What would you improve with an additional day of work

For decode, vectorized 128-bit loads (float4 / __half2): achieved bandwidth scales with d (0.19 TB/s at d=64 vs 0.59 at d=128), the signature of per-thread loads below the 128-bit transaction width, widening them targets the regime that matters most. For prefill, closing the tensor-core gap needs a smaller per-warp shared-memory footprint so prefetch and high occupancy coexist.

## V. Attribution
Tensor-core matmuls use NVIDIA’s wmma API (mma.h). Online-softmax follows Milakov & Gimelshein (2018). No cuBLAS, cuDNN, FlashAttention or high-level attention libraries were used.
