"""STaR-style self-distillation: Qwen3-0.6B writes its own short rationales for GSM8K train problems;
only rationales reaching the gold answer are kept as think-channel content. Also scores the base model on test."""
import json, re, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
split, n_samp, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
think_mode = len(sys.argv) > 4 and sys.argv[4] == 'think'
tk = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); tk.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.bfloat16).cuda().eval()
rows = [json.loads(l) for l in open(f'data/gsm8k/{split}.jsonl')]
if split == 'test':
    rows = rows[:300]
gold = lambda r: r['answer'].split('####')[-1].strip().replace(',', '')
INSTR = "Solve the problem with short, clear reasoning steps. End with a final line of the form 'Answer: 42' (just the number).\n\n"

def ans_of(t):
    m_ = re.findall(r'Answer:\s*[<\$\\{boxed]*\s*(-?[\d,]*\.?\d+)', t)
    return m_[-1].replace(',', '').rstrip('.') if m_ else None

res, bs = [], 64
for i in range(0, len(rows), bs):
    chunk = rows[i:i + bs]
    prompts = [tk.apply_chat_template([{'role': 'user', 'content': INSTR + r['question']}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=think_mode) for r in chunk]
    enc = tk(prompts, return_tensors='pt', padding=True).to('cuda')
    kept = [None] * len(chunk)
    for s in range(n_samp):
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=1536 if think_mode else 384, do_sample=s > 0, temperature=0.7, top_p=0.95)
        for j, seq in enumerate(g[:, enc.input_ids.shape[1]:]):
            t = tk.decode(seq, skip_special_tokens=True).strip()
            a = ans_of(t)
            ok = a is not None and abs(float(a) - float(gold(chunk[j]))) < 1e-6
            if s == 0 and split == 'test':
                res.append(dict(q=chunk[j]['question'], gold=gold(chunk[j]), pred=a, ok=ok))
            if ok and kept[j] is None:
                kept[j] = t
    if split == 'train':
        for r, t in zip(chunk, kept):
            if t:
                res.append(dict(q=r['question'], gold=gold(r), rationale=t))
    print(i + len(chunk), len(res), flush=True)
with open(out, 'w') as f:
    for r in res: f.write(json.dumps(r) + '\n')
if split == 'test':
    print('base accuracy', sum(r['ok'] for r in res) / len(res))
