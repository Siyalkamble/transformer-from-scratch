## ADR — Architecture Decision Record

### The actual checklist (use this every time, not just for RoPE)

1. **What's the invariant that must hold regardless of choice?** (Here: the relative-position math must work — both pairings satisfy it. If a choice breaks the invariant, it's not a trade-off, it's disqualified.)
2. **What are ALL the candidate approaches?** — you can't weigh options you don't know exist. This is the actual bottleneck for you right now, not the weighing.
3. **What axis am I comparing on?** — speed, memory, code simplicity, readability, compatibility with a reference implementation, hardware constraints. Name the axis explicitly, don't compare vaguely.
4. **What's MY specific constraint that breaks the tie?** — for you: CPU-only (favors fewer ops/vectorized code over reshapes), and LLaMA's `apply_rotary_emb` is your stated oracle (favors matching its convention so your diff-check later is clean).


The ranked list — with trade-offs, not a single answer

1. Hypothesis-driven debugging (the general form of what you asked about) — highest leverage for you specifically.
   This is: state a falsifiable hypothesis before touching code, run the smallest experiment that could kill it, log the result, repeat. You already have the scaffolding — your "attempt-first, specific-question" protocol and BUGS.md are literally this, applied to asking me for help. The reason this ranks above derive-build-diff: it's testable live. A 45-minute technical screen or take-home review can watch whether you thrash randomly at a bug or narrow it systematically. Nobody live-tests "did you rederive RoPE." Concrete example: your loss goes to NaN. Weak candidate tries five random fixes. Strong candidate says "either LR is too high or there's an inf in preprocessing — let me check gradient norms first before touching LR," and can tell you which branch it was and why, after. That sentence is worth more in an interview than the whole derive-build-diff paragraph.
2. Eval/measurement rigor for GenAI systems — you've already correctly identified this, don't let it get displaced.
   Your own portfolio-projects.md already names this as the highest-leverage differentiator (before/after reranker precision, eval leaderboard). This is a real market gap: everyone can prompt an LLM, almost nobody can rigorously show a RAG change helped or hurt with numbers instead of vibes. Non-obvious point most people miss: this skill is undersupplied specifically because it's boring — writing an eval harness has none of the dopamine of "I built attention from scratch," so most self-taught engineers skip it. That's exactly why it's rarer and worth more. Keep this as your #1 hiring signal, not derive-build-diff.
3. Derive-build-diff / mechanistic first-principles understanding — real, but scope it down.
   Use it on the 4–5 components in Project 1 where the design choice itself is the content (attention, RoPE, KV-cache) — as already planned. Don't extend it to optimizers, tokenizers, or anything where the reference is just numerical bookkeeping tricks. This is a "top 10%" skill; skill #1 and #2 above are closer to top 1% because almost nobody actually executes them consistently, even among people who know they should.

Non-obvious pick, deliberately not trendy: writing that lets a senior engineer trust your result without re-deriving it themselves. Most CS students think the code is the artifact. In practice, a BUGS.md entry structured as hypothesis → experiment → result → decision is a hiring signal on its own — separate from whether the code works — because it shows you can be handed ambiguous debugging work and trusted to report back honestly. You already produce this artifact. Nobody has told you that the artifact itself, not just the project it documents, is the thing that's rare.
