# NeverIdleMind: an always-on LLM with a learned output router

Code, data generators, results and paper for

> **An Always-On Language Model with a Learned Output Router: Routing Thought, Speech, Writing, Code and Action from a Single Stream**
> Siavash Shirzadeh Barough, 2026 (preprint, [`paper/main.pdf`](paper/main.pdf))

A decoder-only LLM emits one token stream. A small router head assigns every generated token to one of seven
channels: `think`, `say`, `write`, `code`, `exec`, `mem` and `idle`.

- There is no stop token: `idle` waits for the next event.
- Events (user messages, tool results, images) are inserted into the stream at the moment they arrive, even mid-thought.
- Tokens routed to `mem` are kept when old context is evicted.

No external agent loop decides when the model thinks, speaks, calls a tool or waits.

## Main results (Qwen3-0.6B, AsyncWorld, 3 seeds, 300 test episodes per split)

| | Router | In-stream tags | Rule-based orchestrator | Turn-based loop |
|---|---|---|---|---|
| Task success, in distribution | 99.6% | 99.4% | 97.6% | 66.5% |
| Task success, stress (6-8 tasks) | 97.7% | 96.8% | 98.0% | 60.3% |
| Quick tasks within 24-tick deadline (in distribution) | 98.4% | 98.2% | 91.5% | 11.7% |
| Median quick-task latency, stress (ticks) | 7 | 9 | 8 | 487 |

In the multimodal pilot (frozen CLIP-ConvNeXt encoder, GSM8K rationales on the think channel), one model:

- answered 95.3% of visual questions on held-out synthetic scenes;
- solved 52.0% of GSM8K problems; the base model's thinking mode solved 53.7%.

See the paper for confidence intervals, ablations and limitations. All tasks are synthetic and all models imitate
scripted policies.

## Layout

| Path | Contents |
|---|---|
| `src/world.py` | AsyncWorld environment, scripted oracle policies for 4 protocols, scoring |
| `src/model.py` | `ChannelLM`: base LLM + channel embedding + router head |
| `src/data.py`, `src/train.py` | training-sequence builder, fine-tuning |
| `src/rollout.py` | batched closed-loop evaluation (one token per sequence per step) |
| `src/analyze.py`, `src/tables.py`, `src/figs.py` | statistics (paired bootstrap), LaTeX tables, figures |
| `src_mm/` | multimodal pilot: `see` channel, synthetic scenes, GSM8K self-distillation, linear probes |
| `scripts/` | run queues used for the paper |
| `runs/` | per-episode results (JSONL) behind every number in the paper |
| `paper/` | LaTeX source, figures, generated tables, the hypotheses document written before the main runs |

## Reproduce

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python src/data.py                                  # builds data/train.pt (all 4 protocols, identical episodes)
python src/train.py --arm router --seed 0           # also: tags | orch | turn | router_noemb
python src/rollout.py --arm router --ckpt ckpt/router_s0.pt --split id --out runs/router_s0_id.jsonl
python src/analyze.py id                            # table with 95% bootstrap CIs
python src/tables.py && python src/figs.py          # regenerate paper/gen and paper/figs
```

Multimodal pilot:

1. `scripts/gpu0_prep.sh` builds the image features and GSM8K rationales.
2. `scripts/gpu0_mmv.sh` runs two-stage training and evaluation.

One training run takes about 13 minutes on a single RTX 4090.

## Model weights

Trained checkpoints (bf16 `state_dict`, load with `ChannelLM(...).load_state_dict`) are attached to the
[GitHub release](../../releases).

## Citation

```bibtex
@misc{shirzadehbarough2026alwayson,
  title  = {An Always-On Language Model with a Learned Output Router: Routing Thought, Speech, Writing, Code and Action from a Single Stream},
  author = {Shirzadeh Barough, Siavash},
  year   = {2026}
}
```

## Licence

Code: MIT. Base model Qwen3-0.6B: Apache-2.0. GSM8K: MIT.
