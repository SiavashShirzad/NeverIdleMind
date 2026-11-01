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

Further results (see the paper for details):

- **Scale.** With Qwen3-1.7B and Qwen3-4B (LoRA), all arms improved under heavy load; router and tags were equal, and the orchestrator still missed more deadlines (router minus orchestrator on quick tasks: +6.8 pp at 4B).
- **Memory.** When revisions arrived after the original request had been evicted, keeping mem-routed tokens raised correct revisions from 43.8% (sliding window) to 67.0%, and to 99.0% after long-gap episodes were added to training.
- **Seeing, hearing and reasoning in one stream.** With frozen CLIP-ConvNeXt (see) and Whisper-base (hear) encoders, one model answered about 96% of visual questions on held-out scenes, added spoken digits from an unseen speaker with 99% accuracy, and solved about 50% of GSM8K problems on its think channel (base model's thinking mode: 53.7%). Interrupting a thought with urgent requests cost 5.0 ± 1.2 pp of math accuracy (3 seeds).

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
| `src_av/` | `src_mm` plus a `hear` channel (Free Spoken Digit Dataset, frozen Whisper-base encoder) |
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

Larger base models: `BASE_MODEL=Qwen/Qwen3-4B LORA_R=64 SUF=_4b scripts/run_q.sh "router tags orch" "0"` (see `scripts/gpu1_scale.sh`).

One 0.6B training run takes about 13 minutes on a single RTX 4090.

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

Code: MIT. Base models Qwen3-0.6B/1.7B/4B: Apache-2.0. GSM8K: MIT. Free Spoken Digit Dataset: CC BY-SA 4.0. Whisper: MIT.
