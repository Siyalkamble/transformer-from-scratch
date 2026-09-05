## Bug #N: [one-line description]

**Date:**
**Component:** (e.g., attention.py — causal mask)

**Expected behavior:**
What should happen. Be specific — not "attention should work," but "token at
position 5 should have zero attention weight on positions 6-9."

**Actual behavior:**
What actually happens. Include the concrete evidence — a printed tensor, a loss
value, an assertion failure. Not "it's broken," show the number.

**Causes ruled out:**

- [ ] Checked X, wasn't it because Y
- [ ] Checked Z, ruled out because W
  (This list grows as you debug within the 45-min box — it's proof you
  investigated before asking, not just staring at the screen.)

**Root cause:**
The actual mechanism that was wrong. One sentence.

**Fix:**
What you changed.

**Why this matters / what it taught you:**
1-2 sentences on the *general* lesson, not just this specific fix.

## Bug: MultiHeadSelfAttention — missing seq_len attribute, wrong instance-state design, mislabeled shape comment

**Date:** 2026-09-05

**Expected behavior:** `forward(x)` should reshape Q/K/V from `(batch, seq_len, d_model)` into per-head form using shape values available at call time.

**Actual behavior:**

1. `forward()` references `self.seq_len`, which is never set in `__init__` — this raises `AttributeError` on first call.
2. `self.batch` is set from a constructor argument and used inside `forward()` — will silently produce wrong reshapes (or crash) whenever the actual input batch size differs from what was passed at init (e.g. last batch of an epoch, inference with batch=1).
3. Comment on the `.view()` call claims output shape is `(batch, n_heads, seq_len, head_dims)`, but `.view(batch, seq_len, n_heads, head_dims)` actually produces `(batch, seq_len, n_heads, head_dims)` — heads and seq_len are not transposed yet at this point.

**Root cause:** Treated `batch` and `seq_len` as architectural constants (like `d_model`, `n_heads`) instead of recognizing them as properties of the input tensor `x`, which change per forward call. Also conflated "reshape into head-split view" with "reshape + transpose into per-head batched-matmul view" — same gap flagged during the Q3 shape-tracing exercise, resurfaced in actual code.

**Fix:** Derive `batch` and `seq_len` from `x.shape` inside `forward()`, not from `__init__` args. Correct the shape comment to reflect what `.view()` alone produces, and add the explicit `.transpose(1, 2)` step (with its own correct comment) before QK^T.

**Status:** Done

## Bug: MultiHeadSelfAttention — .view() after .transpose() without .contiguous()

**Date:** 2026-09-05

**Expected behavior:** Head-merge step should reshape `(batch, n_heads, seq_len, head_dims)` back to `(batch, seq_len, d_model)` after attention output.

**Actual behavior:** `out.transpose(1,2).view(-1, seq_len, self.d_model)` raises `RuntimeError: view size is not compatible with input tensor's size and stride` — `.transpose()` changes strides without moving memory, so the tensor is non-contiguous, and `.view()` requires contiguous memory to reinterpret shape.

**Root cause:** Conflated "logically rearranged shape" with "physically rearranged memory." Already identified this correctly at the conceptual level in the Q9 shape-trace answer ("transpose back + contiguous().view()") but it didn't carry over into the actual implementation.

**Fix:** Insert `.contiguous()` between `.transpose(1,2)` and `.view(...)`, or replace `.view()` with `.reshape()` (which contiguous-copies internally when needed).

**Status:** Fixed.
