# Fused Online-Softmax Attention

Single-pass fused scaled dot-product attention 

## Environment
- GPU: NVIDIA A100-SXM4-80GB
- Driver: 580.126.16
- CUDA: 12.8
- Cloud: RunPod, A100 SXM pod

## Build

pip install -r requirements.txt
pip install -e .

## Run 
pytest
python -m bench.benchmark

