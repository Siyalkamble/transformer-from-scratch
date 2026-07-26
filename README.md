# Transformer From Scratch — Anchor Project

## Problem Statement

Build a decoder-only (GPT-style) autoregressive language model from scratch in PyTorch, train it on a small text corpus on CPU, and be able to defend every tensor shape and design decision in the forward pass without looking at the code.

**Formal definition:**
Given a sequence of tokens `x_1, x_2, ..., x_T`, learn a model `P(x_t | x_1, ..., x_{t-1})` — the probability distribution over the next token conditioned only on preceding tokens (causal/autoregressive). Train via next-token cross-entropy loss (teacher forcing) and generate new sequences via sampling at inference time.

**Constraints:**
- No pre-built attention/transformer layers (`nn.MultiheadAttention`, `nn.TransformerEncoderLayer`, etc. are NOT allowed)
- `nn.Linear`, `nn.LayerNorm`, `nn.Embedding`, `nn.Dropout` ARE allowed as primitives — attention math, masking, multi-head split/concat, and the transformer block itself must be hand-implemented
- Must run and be debuggable on CPU (8GB RAM laptop) for development; GPU (Colab) used only for the final scaled-up training run, not for dev/debug
- No reference implementation (e.g. nanoGPT) consulted until Phase 5 passes

**Definition of done:**
1. Model overfits a single small batch to near-zero loss (sanity check)
2. Model trains on a real (small) dataset with visibly decreasing loss
3. Model generates non-garbage text via sampling (greedy → temperature → top-k/top-p)
4. Every phase has at least one entry in `BUGS.md` demonstrating independent diagnosis
5. RoPE swapped in for sinusoidal positional encoding, with an A/B comparison
6. Repo is clean: modular `.py` files, not a notebook dump

---

## Architecture Blueprint

```
Input tokens (B, T)
    │
    ▼
Token Embedding (vocab_size, d_model)  +  Positional Encoding
    │
    ▼
┌─────────────────────────────────────┐
│  Transformer Block (× N layers)      │
│  ┌─────────────────────────────┐    │
│  │ LayerNorm (pre-LN)          │    │
│  │        │                    │    │
│  │        ▼                    │    │
│  │ Masked Multi-Head Attention │    │
│  │        │                    │    │
│  │        ▼                    │    │
│  │ Residual Add ────────────┐  │    │
│  │        │                 │  │    │
│  │        ▼                 │  │    │
│  │ LayerNorm (pre-LN)        │  │    │
│  │        │                  │  │    │
│  │        ▼                  │  │    │
│  │ FeedForward (4x expand,   │  │    │
│  │ GELU, dropout)            │  │    │
│  │        │                  │  │    │
│  │        ▼                  │  │    │
│  │ Residual Add ──────────────┘ │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
    │
    ▼
Final LayerNorm
    │
    ▼
Linear Head (tied to embedding weights) → vocab logits
    │
    ▼
Softmax → next-token distribution
```

---

## Phase-by-Phase Blueprint

### Phase 0 — Primitives & Data Pipeline
- Confirm shape fluency: `nn.Linear`, `nn.LayerNorm`, `nn.Embedding` (predict shapes on paper before running)
- Build char-level tokenizer (skip BPE — don't learn two new things at once)
- Build `Dataset`/batching: input `x`, target `y` = `x` shifted by 1 token
- **Gate to pass:** predict every tensor shape in the primitives snippet without running code

### Phase 1 — Single Attention Head
- Scaled dot-product attention, one head, no batching complexity
- Causal mask applied before softmax
- Test in isolation with tiny hand-computable tensors, print every intermediate shape
- **Gate to pass:** can explain why divide by `sqrt(d_k)`, and what the causal mask blocks and why

### Phase 2 — Multi-Head Attention
- Split into heads `(B, nh, T, hs)`, run attention per head, concatenate, project through `W_O`
- **Known trap:** incorrect reshape when splitting/merging heads
- *(Optional checkpoint with mentor here if going fully blind proves too costly — verify shapes only, not implementation)*

### Phase 3 — Full Transformer Block
- Attention + residual + pre-LN LayerNorm
- FFN (4x expansion, GELU) + residual + pre-LN LayerNorm
- Dropout included (previously flagged gap — do not skip)
- Mini overfit check on this block alone before moving on

### Phase 4 — Stack Blocks + Positional Encoding
- Start with sinusoidal or learned absolute positional embeddings (not RoPE yet)
- Stack N transformer blocks
- **Gate to pass:** full forward pass runs end-to-end without shape errors

### Phase 5 — Sanity Check (non-negotiable)
- Overfit one tiny batch (1–2 examples) until loss → near zero
- If this doesn't pass, architecture is broken — do not proceed
- This is where independent debugging matters most — no reference code, no shortcuts

### Phase 6 — Real Training Loop
- AdamW, LR warmup + cosine decay, gradient clipping, checkpointing
- Train on small corpus (Tiny Shakespeare acceptable for this phase only — not the final portfolio dataset)
- Scale up and run full training on Colab GPU using the already-debugged local code

### Phase 7 — Generation
- Greedy decoding → temperature sampling → top-k / top-p sampling

### Phase 8 — Extensions (differentiation from generic tutorial clones)
- Swap sinusoidal → RoPE, A/B compare
- KV-cache for faster inference

---

## Repo Structure

```
transformer-from-scratch/
├── README.md
├── BUGS.md
├── data.py           # tokenizer, Dataset, batching
├── model.py          # attention, MHA, transformer block, full model
├── train.py          # training loop, optimizer, scheduler
├── generate.py        # sampling strategies
├── configs/
│   └── toy_config.py # tiny debug config (d_model=8, n_heads=2, n_layers=1)
└── checkpoints/
```

---

## Debugging Protocol (applies to every phase)

1. Shape assertions on every forward pass, written before running the code
2. Test every component in isolation with tiny, hand-computable inputs before integrating
3. Mini overfit-check after each major component, not just at Phase 5
4. Time-box: 45 minutes stuck on the same bug → write a bug report entry, then move on
5. Bug report format (in `BUGS.md`): expected vs. actual, what's been ruled out, relevant code — written *before* asking for outside help

## Rules of Engagement (self-imposed)

- No nanoGPT or reference code consulted until Phase 5 passes
- Any AI/mentor help before Phase 5 is Socratic (diagnostic questions only) — not code fixes
- Dev and debug on CPU locally with the toy config; GPU used only for the scaled final run in Phase 6
