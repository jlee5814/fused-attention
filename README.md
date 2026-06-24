# Online Softmax + Scaled Dot-Product Attention

Single-pass fused scaled dot-product attention that computes attention for a multi-head workload, dispatched to unique kernels for the decode and prefill phases since the two are bounded by different resources. Inputs are BF16. Accumulation is FP32. Output is BF16. 

## Environment

- GPU: NVIDIA A100-SXM4-80GB
- Driver: 580.126.16
- CUDA: 12.8
- PyTorch: 2.8.0+cu128
- Cloud: RunPod, A100 SXM pod

## Build

```
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

## Run

```
pytest tests/ -q
python -m bench.benchmark
```

## Attribution

Tensor-core matmuls use NVIDIA’s wmma API (mma.h). Online-softmax follows Milakov & Gimelshein (2018). No cuBLAS, cuDNN, FlashAttention or high-level attention libraries were used.
