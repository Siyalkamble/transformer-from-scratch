## Ablation #N: [comparison name]
**Date hypothesis written:**
**Date result recorded:**

**Setup:** exact config difference (e.g., "identical model, only norm 
placement differs: pre-LN vs post-LN, 6 layers, 4 heads, seq_len=256")

**Hypothesis (written BEFORE running):**
What you expect and WHY — the mechanism, not just the outcome.
e.g. "post-LN will show higher loss variance / harder convergence at this 
depth because gradients through the unnormalized residual stream compound 
across layers, whereas pre-LN keeps the residual path clean."

**Result:**
Actual numbers/curves. Link to the plot.

**Hypothesis held?** Yes / No / Partially

**Analysis (written AFTER seeing result):**
If it held: why the mechanism you predicted is confirmed by this data.
If it didn't: this is the valuable part — what did you get wrong about 
the mechanism? Don't hand-wave it, dig into why your mental model was off.