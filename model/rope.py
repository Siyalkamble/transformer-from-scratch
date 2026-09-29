
import math
import torch
import torch.nn as nn
'''
we will have (Batch_size, seq_len, n_heads, head_dims)

m = token position
i = 2d pair sequence position | 0...head_dims/2
θ_i = 10000^(-2i/d)

angle[m, i] = m * i_θ


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
            x
            
    ):



    
