import csv
import os

import torch

from fused_attention import fused_attention
from bench.shapes import ALL_SHAPES, BATCH, HEADS

def make_inputs(Sq, Skv, d, device="cuda", dtype=torch.bfloat16):
    q = torch.randn(BATCH, HEADS, Sq, d, device=device, dtype=dtype)
    k = torch.randn(BATCH, HEADS, Skv, d, device=device, dtype=dtype)
    v = torch.randn(BATCH, HEADS, Skv, d, device=device, dtype=dtype)
    return q, k, v

def bench_one(fn, q, k, v, warmup=10, iters=50):
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(q, k, v)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3

def metrics(Sq, Skv, d, seconds):
    # FLOPs: QK^T (2*Sq*Skv*d) + PV (2*Sq*Skv*d) = 4*Sq*Skv*d, per (batch, head).
    flops = 4 * Sq * Skv * d * BATCH * HEADS 
    bytes_moved =  2 * (Skv * d) * 2 * BATCH * HEADS
    return flops / seconds / 1e12, bytes_moved / seconds / 1e12 

def main():
    assert torch.cuda.is_available(), "no CUDA device"
    os.makedirs("results", exist_ok=True)
    rows = []
    for (Sq, Skv, d) in ALL_SHAPES:
        q, k, v = make_inputs(Sq, Skv, d)
        sec = bench_one(fused_attention, q, k, v)
        tflops, tbps = metrics(Sq, Skv, d, sec)
        regime = "decode" if Sq == 1 else "prefill"
        rows.append([regime, Sq, Skv, d, sec * 1e6, tflops, tbps])
        print(f"{regime:8s} Sq={Sq:4d} Skv={Skv:5d} d={d:4d}    "
              f"{sec*1e6:8.1f} us   {tflops:7.2f} TFLOP/s  {tbps:6.2f} TB/s")
        with open("results/benchmark.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["regime", "S_q", "S_kv", "d", "us", "TFLOP_s", "TB_s"])
            w.writerows(rows)
        print("\nwrote results/benchmark.csv")

if __name__ == "__main__":
    main()

