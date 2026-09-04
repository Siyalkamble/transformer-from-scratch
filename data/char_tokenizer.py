
class CharTokenizer:

    def __init__(self, text:str) -> None:
        vocab = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)
       
    def encode(self, string):
        return [self.stoi[c] for c in string]

    def decode(self, integers):
        return ''.join(self.itos[i] for i in integers)

