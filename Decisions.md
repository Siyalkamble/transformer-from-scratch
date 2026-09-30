## DECISIONS.md

Date: [today's date]
Change: Project scope expanded from "build from-scratch transformer" to
"build + run controlled architecture ablation suite + anchor baseline +
conditionally run mini scaling-law study."
Trigger: [the REAL reason — not "sounds impressive"]
Time budget check: [actual hours/week for this project, honestly]
Gate: Stage 3 does not start until Stage 2 is fully complete.

### 26-08-2026 — Char tokenizer: .lower() added then reverted

Change: Added `.lower()` to tokenizer during Stage 1, then removed it.
Why: I thought that will make the vocab size smaller
Trigger: capitilised words do have meaning (starting word, etc)
Current state: tokenizer is case-sensitive, vocab_size includes upper/lowercase separately.


## Decision: RoPE pairing convention — adjacent-pair chosen for Stage 1

**Date:** 2026-09-29
**Change:** Implementing RoPE using adjacent-pair convention (`(x0,x1), (x2,x3), ...`) rather than split-half convention (`(x0, x_{d/2}), ...`).
**Trigger:** Stage 1 is CPU-only with no wall-clock speed constraint driving the choice; adjacent-pair matches RoFormer §3.2's ground-truth equations directly, letting `rope.py` be verified against the paper's math with no index-remapping step. Split-half is the GPU-throughput-oriented convention used by LLaMA/GPT-NeoX — that reasoning doesn't apply until GPU training (Stage 2+) is actually happening and speed is actually a measured bottleneck, not a hypothetical one.
**Time budget check:** n/a — this is an implementation detail, not a scope change to the project timeline.
**Gate:** Split-half is NOT implemented now. It stays deferred until (a) Stage 2 GPU training is underway, AND (b) wall-clock speed is measured and shown to matter at that point. If both hold, log a new entry here documenting the switch and the permutation-equivalence argument for why both conventions produce the same model behavior. Until then, `rope.py` uses adjacent-pair only — no parallel implementation.
