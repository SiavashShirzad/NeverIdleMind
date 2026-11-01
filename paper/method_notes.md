# Method notes (frozen design, written before main results)

## Pre-registered hypotheses
- H1 (routing vs tags): with asynchronous delivery fixed, a learned per-token router (R) matches or beats in-stream channel tags (T) on task success, with fewer generated tokens and lower interrupt latency.
- H2 (in-model vs orchestrated control): R beats a rule-based boundary/preemptive orchestrator (O) on urgent-interrupt success and quick-response latency at equal training compute.
- H3 (vs turn-based): R and T beat a turn-based agent loop (D) on interrupt categories (quick/cancel). Non-interrupt tasks are expected to be similar.
- H4 (memory by routing): over long horizons with a fixed 1024-token context, keeping mem-routed tokens on eviction beats a plain sliding window on revision/cancellation tasks whose originating event has been evicted.

## Fixed choices
- Base: Qwen3-0.6B (full fine-tune, bf16 autocast, AdamW lr 5e-5, new params 1e-3, 1 epoch, 16k tokens/batch, cosine schedule).
- Training data: 12,000 'id' episodes + 600 'long' episodes x 4 eviction views (mem or plain), identical episodes across arms; each arm is trained on oracle traces under its own protocol.
- Test: 300 id episodes (seeds 0..299), 300 stress episodes (seeds 100000..), 32 long episodes (seeds 200000..). Greedy decoding.
- Metrics: task success; quick (deadline 24 ticks); cancel compliance (grace 10 ticks); revise correctness; plain tasks; median quick latency; median completion latency; generated tokens; ticks.
- Statistics: 3 training seeds/arm; mean over seeds; 95% CI from 2000 bootstrap resamples of test episodes (paired across arms).
