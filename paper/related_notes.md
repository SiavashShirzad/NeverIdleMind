# Verified related-work notes (from arXiv html, first ~100k chars of each; appendices not read)

- li2026never (arXiv:2609.17416, page shows 11 Jul 2026 v1; cite by ID): single think/say/tool_call stream; ~200-line external orchestrator "alone knows the clock", force-appends </think>, injects events at interrupt points as chat turns; Qwen3-8B/4B; SFT + distillation + RFT + GRPO/DAPO; ReactiveBench; streaming completion 48% -> 73±5%. No idle token, no memory/eviction described.
- su2026multistream (arXiv:2605.12460): H parallel role-fixed streams decoded in lockstep, per-stream position counters + learnable stream embeddings, masked "-" empty cells; fixed tick for inputs; no learned routing ("fixed streams do not require learning when to branch or merge"); no eviction.
- yakushev2025async (arXiv:2512.10931, v3 May 2026): training-free thinker/writer views of one KV cache via RoPE rotation; writer gated by an inserted yes/no probe every paragraph/T=20 tokens; inputs only at paragraph boundaries; no idle, no memory.
- gander2026duplex (arXiv:2609.08977, "Multimodal Duplex Interaction Agent", Gander, MiniCPM-o 4.5): 1-s chunks, learned per-chunk control token listen/speak/interrupt; separate "brain" agent via task_* tool tokens; sliding window of 128 chunks + separate memory_search tool.
- 2604.00892 title is "When Users Change Their Mind: Evaluating Interruptible Agents in Long-Horizon Web Navigation" (not "InterruptBench").
- Other 18 entries: metadata only verified (refs.bib, @misc with eprint).
