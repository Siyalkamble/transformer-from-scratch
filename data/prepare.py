from char_tokenizer import CharTokenizer  
import torch

def load(path:str) -> str:
    with open(path, "r", encoding='utf-8') as f:
        text = f.read()
        return text

class PrepareData():

    def __init__(self, text:str, seq_len:int, split_ratio:float):

        self.seq_len = seq_len
        self.split_ratio = split_ratio
        assert 0.0 < self.split_ratio < 1.0, "split_ratio must be in (0, 1)"

        tok = CharTokenizer(text)
        self.token_ids = torch.tensor(tok.encode(text), dtype=torch.long)
        assert self.token_ids.dtype == torch.long, "token_ids dtype must be torch.long"


    def _chunk(self, token_ids:torch.Tensor):
        usable_data = (len(token_ids) // self.seq_len) * self.seq_len
        token_ids = token_ids[:usable_data]
        seqs = token_ids.view(-1, self.seq_len)
        inputs = seqs[:, :-1]   # columns 0 to 62 → (num_samples, 63)
        targets = seqs[:, 1: ]  # columns 1 to 63 → (num_samples, 63)
        return inputs, targets

    def prepare(self):
        n = int(len(self.token_ids) * self.split_ratio)
        train_data = self.token_ids[:n]
        val_data = self.token_ids[n:]
        train_inp, train_targ = self._chunk(train_data)
        val_inp, val_targ = self._chunk(val_data)

        return train_inp, train_targ, val_inp, val_targ

if __name__ == "__main__":
    text = load("data/tinyshakespeare.txt")
    prep = PrepareData(text, seq_len=64, split_ratio=0.9)
    train_inp, train_targ, val_inp, val_targ = prep.prepare()

        


