from char_tokenizer import CharTokenizer  
import torch
import torch.nn as nn

def prepare_data():

    # Load data
    with open("data/tinyshakespeare.txt", 'r', encoding='utf-8') as f:
        text = f.read()

    tok = CharTokenizer(text)
    token_ids = torch.tensor(tok.encode(text), dtype=torch.long)

    # convert this to embedding of size 
    emb = nn.Embedding(tok.vocab_size, 256)
    sample = torch.tensor(tok.encode(text[:50]))
    out = emb(sample)

    #s split data
    n = int(0.9 * len(token_ids))
    train_tok = token_ids[:n]
    val_tok = token_ids[n:]


