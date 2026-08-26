## DECISIONS.md

### 26-08-2026 — Char tokenizer: .lower() added then reverted
Change: Added `.lower()` to tokenizer during Stage 1, then removed it.
Why: I thought that will make the vocab size smaller 
Trigger: capitilised words do have meaning (starting word, etc)
Current state: tokenizer is case-sensitive, vocab_size includes upper/lowercase separately.



