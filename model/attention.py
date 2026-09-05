import torch
import torch.nn as nn

batch = 32
seq_len = 64
d_model = 256
n_heads = 4

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model:int ,n_heads: int):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.head_dims = self.d_model // self.n_heads # 64 
        assert (self.d_model % n_heads) == 0, "d_model must be divisible by n_heads"

        self.scale = self.head_dims ** -0.5 # not by n_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):

        seq_len = x.shape[1]

        Q = self.Wq(x) # (batch, seq_len, d_model)
        K = self.Wk(x) # no heads yet 
        V = self.Wv(x)

        # (batch, seq_len, n_heads, head_dims) -> (batch, n_heads, seq_len, head_dims)
        Q = Q.view(-1, seq_len, self.n_heads, self.head_dims).transpose(1,2) 
        K = K.view(-1, seq_len, self.n_heads, self.head_dims).transpose(1,2)
        V = V.view(-1, seq_len, self.n_heads, self.head_dims).transpose(1,2)

        scores = Q @ K.transpose(-2, -1) * self.scale
        attention = scores.softmax(dim=-1)
        out = attention @ V

        concatenated  = out.transpose(1,2).contigious().view(-1, seq_len, self.d_model)

        out_proj = self.Wo(concatenated)

        return out_proj # (batch, seq_len, d_model)
