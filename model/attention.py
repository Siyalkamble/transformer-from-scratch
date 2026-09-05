import torch
import torch.nn as nn

batch = 32
seq_len = 64
d_model = 256
n_heads = 4
max_seq_len = 512

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model:int ,n_heads: int, max_seq_len:int):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        self.head_dims = self.d_model // self.n_heads # 64 
        assert (self.d_model % n_heads) == 0, "d_model must be divisible by n_heads"

        self.scale = self.head_dims ** -0.5 # not by n_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):

        seq_len = x.shape[1]

        Q = self.Wq(x) # (batch, seq_len, d_model)
        K = self.Wk(x) # no heads yet 
        V = self.Wv(x)

        # (batch, seq_len, d_model).view -> (batch, seq_len, n_heads, head_dims).T -> (batch, n_heads, seq_len, head_dims)
        Q = Q.view(-1, seq_len, self.n_heads, self.head_dims).transpose(1,2) 
        K = K.view(-1, seq_len, self.n_heads, self.head_dims).transpose(1,2)
        V = V.view(-1, seq_len, self.n_heads, self.head_dims).transpose(1,2)

        # (batch, n_heads, seq_len, head_dims) @ (batch, n_heads, head_dims, seq_len).T -> (batch, n_heads, seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1) * self.scale

        mask = self.causal_mask[:seq_len, :seq_len] # type: ignore # (seq_len, seq_len)

        scores = scores.masked_fill(mask, float('-inf')) # (batch, n_heads, seq_len, seq_len) 
        
        attention = scores.softmax(dim = -1) # (batch, n_heads, seq_len, seq_len)

        out = attention @ V
        # attention: (batch, n_heads, seq_len, seq_len)
        # V: (batch, n_heads, seq_len, head_dims)
        # out: (batch, n_heads, seq_len, head_dims)

        concatenated  = out.transpose(1,2).contiguous().view(-1, seq_len, self.d_model)
        #  (batch, seq_len, n_heads, head_dims) -> (batch, seq_len, d_model)

        out_proj = self.Wo(concatenated) # (batch, seq_len, d_model) 

        return out_proj # (batch, seq_len, d_model)

