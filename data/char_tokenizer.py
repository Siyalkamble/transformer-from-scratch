import torch
import torch.nn as nn

with open("data/tinyshakespeare.txt", 'r') as f:
    text = f.read()

class CharTokenizer:

    def __init__(self, text):
        vocab = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)
       
    def encode(self, string):
        return [self.stoi[c] for c in string]

    def decode(self, integers):
        return ''.join(self.itos[i] for i in integers)


tok = CharTokenizer(text)
emb = nn.Embedding(tok.vocab_size, 32)
sample = torch.tensor(tok.encode(text[:50]))
out = emb(sample)
print(tok.vocab_size) # 65
print(out.shape)  

token_ids = torch.tensor(tok.encode(text), dtype=torch.long)
print(token_ids[:100])

n = int(0.9 * len(token_ids))
X = token_ids[:n]
y = token_ids[n:]

test_str = "hello world"
assert tok.decode(tok.encode(test_str)) == test_str
print("round-trip passed")