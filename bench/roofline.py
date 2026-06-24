import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# A100 SXM ceilings
HBM_BW      = 2.0e12     
CUDA_PEAK   = 19.5e12    
TENSOR_PEAK = 312e12      

decode = [
    ("dec 512/64",   1, 512,  64,  0.24),
    ("dec 512/128",  1, 512,  128, 0.40),
    ("dec 2048/64",  1, 2048, 64,  0.19),
    ("dec 2048/128", 1, 2048, 128, 0.55),
    ("dec 4096/128", 1, 4096, 128, 0.58),
]
prefill = [
    ("pre 1024/64",   64,  1024, 64,  1.43),
    ("pre 2048/128",  64,  2048, 128, 1.66),
    ("pre 2048/64",   128, 2048, 64,  2.00),
    ("pre 4096/128",  128, 4096, 128, 2.33),
    ("pre 4096/128b", 256, 4096, 128, 3.03),
]

def intensity(Sq, Skv, d):
    flops = 4.0 * Sq * Skv * d
    bytes_ = 2.0 * (2*Sq*d + 2*Skv*d)
    return flops / bytes_

fig, ax = plt.subplots(figsize=(9, 6))

x = np.logspace(-1, 3, 500)
ax.plot(x, np.minimum(HBM_BW * x, CUDA_PEAK) / 1e12, 'k-', lw=1.5, label="CUDA-core roofline (19.5 TF)")
ax.axhline(TENSOR_PEAK / 1e12, color='tab:red', ls='--', lw=1.5, label="Tensor-core peak (312 TF)")
ax.axhline(CUDA_PEAK / 1e12, color='gray', ls=':', lw=1)

# ridge point (CUDA core)
ridge = CUDA_PEAK / HBM_BW
ax.axvline(ridge, color='gray', ls=':', lw=0.8)
ax.text(ridge*1.1, 0.15, f"ridge ~{ridge:.1f}", fontsize=8, color='gray')

# data points
for lbl, Sq, Skv, d, tf in decode:
    ai = intensity(Sq, Skv, d)
    ax.plot(ai, tf, 'o', color='tab:blue', ms=7)
for lbl, Sq, Skv, d, tf in prefill:
    ai = intensity(Sq, Skv, d)
    ax.plot(ai, tf, 's', color='tab:green', ms=7)

ax.plot([], [], 'o', color='tab:blue', label='decode (S_q=1)')
ax.plot([], [], 's', color='tab:green', label='prefill (S_q>>1)')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
ax.set_ylabel("Performance (TFLOP/s)")
ax.set_title("Fused Attention on A100 (Roofline)")
ax.legend(loc='lower right', fontsize=8)
ax.grid(False)
plt.tight_layout()
plt.savefig("results/roofline.png", dpi=130)
print("wrote results/roofline.png")