import math 
import torch

def attention_reference(q, k, v):
    """
    Two-pass

    q: (B, H, S_q, d), k/v: (B, H, S_kv, d). Returns (B, H, S_q, d).
    """
    d = q.shape[-1]
    qf, kf, vf = q.float(), k.float(), v.float()
    scores = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(d)
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, vf)
    return out.to(q.dtype)
