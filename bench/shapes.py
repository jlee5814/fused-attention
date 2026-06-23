# (S_q, S_kv, d)
# Inputs BF16, Accumulate FP32, Output BF16

BATCH = 16
HEADS = 8

DECODE_SHAPES = [
    (1, 512, 64),       # short KV cache
    (1, 512, 128),      # short KV cache
    (1, 2048, 64),      # medium KV cache
    (1, 2048, 128),     # medium KV cache
    (1, 4096, 128),     # long KV cache
]

PREFILL_SHAPES = [
    (64, 1024, 64),     # small chunk
    (64, 2048, 128),    # small chunk
    (128, 2048, 64),    # medium chunk
    (128, 4096, 128),   # medium chunk
    (256, 4096, 128),   # large chunk
]

ALL_SHAPES = DECODE_SHAPES + PREFILL_SHAPES
