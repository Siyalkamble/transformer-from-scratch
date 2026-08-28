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



