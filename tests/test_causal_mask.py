'''
1. Direct mask inspection. Forget the model for a second — can you assert something directly about self.causal_mask itself? What should mask[0, 0], mask[0, 5], and mask[5, 0] each be, and why? (Position 0 attending to position 5 — future — vs. position 5 attending to position 0 — past.)

2. Attention weights, not just mask values. After a real forward pass, attention (post-softmax) should have actual zeros in specific places. For query position i, what should be true about attention[..., i, j] for every j > i? This is testing the effect of the mask, not just its construction.

3. The "does the future leak through gradients" test. This is the sharpest one and the one most people skip. If you change a future token's input embedding and rerun forward, should the output at an earlier position change at all? How would you actually check that with two forward passes and a comparison — not eyeballing, an assertion?

4. Softmax sanity under masking. Each row of attention should still sum to 1 (softmax property preserved even with -inf masking, per the mechanism you already know: exp(-inf)=0). What's the assertion, and what tolerance do you need for floating point (torch.allclose, not ==)?

5. Shape/no-NaN sanity. With -inf in the mix, is there any position where an entire row could end up all -inf (which would make softmax produce NaN)? Think about position 0 — how many valid (unmasked) keys does it have? Is that number ever zero? Worth asserting torch.isnan(attention).any() is False.

6. Different seq_len values. You built this to slice self.causal_mask[:seq_len, :seq_len] for variable-length input. Does it still work correctly at seq_len=1? At seq_len=max_seq_len? Edge cases like seq_len=1 are where slicing bugs hide.'''


# tests/test_causal_mask.py
from model.attention import MultiHeadSelfAttention
import torch
import pytest


@pytest.fixture
def mha():
    """Fresh MHA instance for each test — no cross-test leakage."""
    return MultiHeadSelfAttention(d_model=8, n_heads=2, max_seq_len=6)


def test_causal_mask_structure(mha):
    """Test 1: Direct mask inspection — no forward pass needed."""
    mask = mha.causal_mask  # shape: [max_seq_len, max_seq_len] = [6, 6]

    # Position 0 attending to position 5 (future) → should be masked (True)
    assert mask[0, 5] is True, "Future token (pos 5) should be masked for query at pos 0"

    # Position 5 attending to position 0 (past) → should NOT be masked (False)
    assert mask[5, 0] is False, "Past token (pos 0) should NOT be masked for query at pos 5"

    # Diagonal (self-attention) → never masked
    assert mask[0, 0] is False, "Self-attention (pos 0 → pos 0) should NOT be masked"
    assert mask[3, 3] is False, "Self-attention (pos 3 → pos 3) should NOT be masked"


def test_attention_weights_have_zeros_for_future(mha):
    """Test 2: After forward pass, attention weights should be exactly 0 for masked positions."""
    x = torch.randn(1, 6, 8)  # [batch, seq_len=6, d_model=8]
    _ = mha.forward(x)
    attn = mha.last_attention  # shape: [batch, seq_len, seq_len] = [1, 6, 6]

    # For every query position i, all future keys j > i should have attention == 0
    for i in range(6):
        for j in range(i + 1, 6):
            assert attn[0, i, j] == 0.0, f"Attention[{i}, {j}] (future) should be 0.0, got {attn[0, i, j]}"


def test_future_token_change_does_not_affect_past_output(mha):
    """Test 3: Causal isolation — changing future token should NOT change past outputs."""
    x = torch.randn(1, 6, 8)
    y = x.clone()

    # Modify only the last token (position 5, the furthest future)
    y[0, 5, :] = torch.randn(8)

    out1 = mha.forward(x)
    out2 = mha.forward(y)

    # Output at position 0 should be identical (it can't attend to position 5)
    assert torch.allclose(out1[0, 0, :], out2[0, 0, :], atol=1e-6), \
        "Changing future token (pos 5) should NOT affect output at pos 0"

    # Output at position 5 SHOULD change (it attends to itself)
    assert not torch.allclose(out1[0, 5, :], out2[0, 5, :], atol=1e-6), \
        "Changing token at pos 5 SHOULD affect output at pos 5"


def test_softmax_rows_sum_to_one(mha):
    """Test 4: Each attention row should sum to 1 (softmax property preserved)."""
    x = torch.randn(2, 6, 8)  # batch=2 to test multiple sequences
    _ = mha.forward(x)
    attn = mha.last_attention  # [2, 6, 6]

    # Each row (query position) should sum to 1.0 within float tolerance
    row_sums = attn.sum(dim=-1)  # [2, 6]
    expected = torch.ones_like(row_sums)
    assert torch.allclose(row_sums, expected, atol=1e-6), \
        f"Attention rows should sum to 1.0, got max deviation: {(row_sums - expected).abs().max().item()}"


def test_no_nan_in_attention(mha):
    """Test 5: No NaN values — even with -inf masking, softmax should be stable."""
    x = torch.randn(1, 6, 8)
    _ = mha.forward(x)
    attn = mha.last_attention

    assert not torch.isnan(attn).any(), "Attention weights contain NaN — check masking + softmax stability"

    # Extra: position 0 has only 1 valid key (itself), so its row should be [1, 0, 0, 0, 0, 0]
    # This ensures no row is all -inf before softmax (which would cause NaN)
    assert torch.allclose(attn[0, 0, :], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), atol=1e-6), \
        "Position 0 should attend only to itself (all mass on key 0)"


def test_variable_seq_len_and_edge_cases(mha):
    """Test 6: Sliced mask should work for seq_len=1, seq_len=max, and reject >max."""
    # Edge case: seq_len=1 (single token — should work, no masking needed)
    x1 = torch.randn(1, 1, 8)
    out1 = mha.forward(x1)
    assert out1.shape == (1, 1, 8), f"Output shape mismatch for seq_len=1: expected (1,1,8), got {out1.shape}"
    assert not torch.isnan(out1).any(), "NaN output for seq_len=1 — slicing bug?"

    # Edge case: seq_len=max_seq_len=6 (full context — should work)
    x6 = torch.randn(1, 6, 8)
    out6 = mha.forward(x6)
    assert out6.shape == (1, 6, 8), f"Output shape mismatch for seq_len=6: expected (1,6,8), got {out6.shape}"

    # seq_len > max_seq_len should raise an error (you added this assertion in forward())
    x_invalid = torch.randn(1, 513, 8)
    with pytest.raises(AssertionError, match="seq_len"):
        # Your forward() has: assert seq_len <= self.max_seq_len, "..."
        # So this should raise AssertionError, not silently run
        _ = mha.forward(x_invalid)