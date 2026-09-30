
import math
import torch
import torch.nn as nn
'''
we will have (Batch_size, seq_len, n_heads, head_dims)

m = token position
i = 2d pair sequence position | 0...head_dims/2
θ_i = 10000^(-2i/d)

angle[m, i] = m * θ_i


'''

class RotaryPositionEncoding(nn.Module):

    def __init__(
        self, max_seq_len: int,
        head_dims: int,
        base: float = 10_000.0,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.head_dims = head_dims
        self.base = base
        self.precomputes()

    # computes the sin and cos since it only depends on m and head_dims
    def precomputes(
            self
    ):
        i = torch.arange(0, self.head_dims//2, dtype=torch.float32)

        freqs = 1.0 / (self.base ** (2*i / self.head_dims))

        m = torch.arange(0, self.max_seq_len, dtype=torch.float32)

        angles = torch.outer(m, freqs)

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)


    def forward(
            self,
            q,
            k
            
    ):
        seq_len = q.size(1)

        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(2)   # [1, seq_len, 1, head_dims/2]
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(2)

        xq = q.float()
        xq1 = xq[..., 0::2] # even position
        xq2 = xq[..., 1::2] # odd position

        xk = k.float()
        xk1 = xk[..., 0::2] # even position
        xk2 = xk[..., 1::2] # odd position

        outq1 = xq1 * cos - xq2 * sin
        outq2 = xq1 * sin + xq2 * cos

        outk1 = xk1 * cos - xk2 * sin
        outk2 = xk1 * sin + xk2 * cos

        outq = torch.stack((outq1, outq2), dim=-1).flatten(-2)  # [batch, seq_len, n_heads, head_dims]
        outk = torch.stack((outk1, outk2), dim=-1).flatten(-2)  # [batch, seq_len, n_heads, head_dims]

        return outq.type_as(q), outk.type_as(k)
    
            


    
