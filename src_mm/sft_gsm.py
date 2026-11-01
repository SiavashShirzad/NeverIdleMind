"""Matched baseline: plain chat SFT of Qwen3-0.6B on the same self-distilled GSM8K rationales (no router, no stream),
same number of math examples as the router_mmv training mixture, then greedy test on the first 300 test problems."""
import json, math, random, re, sys, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

N_EX, LR, BS, out = int(sys.argv[1]), 5e-5, 16, sys.argv[2]
torch.manual_seed(0); random.seed(0)
tk = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); tk.padding_side = 'left'
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float32).cuda()
m.gradient_checkpointing_enable(); m.config.use_cache = False
INSTR = "Solve the problem with short, clear reasoning steps. End with a final line of the form 'Answer: 42' (just the number).\n\n"
prompt = lambda q: tk.apply_chat_template([{'role': 'user', 'content': INSTR + q}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
rat = [json.loads(l) for l in open('data/gsm8k/train_rationales.jsonl')]
ex = [rat[i % len(rat)] for i in random.sample(range(len(rat)), len(rat))] * (N_EX // len(rat) + 1)
ex = ex[:N_EX]; random.shuffle(ex)
opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
n_steps = math.ceil(len(ex) / BS)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * s / n_steps)))
m.train()
for step in range(n_steps):
    b = ex[step * BS:(step + 1) * BS]
    for k in range(0, len(b), 4):  # micro-batches of 4, gradient accumulation to the batch of 16
        mb = b[k:k + 4]
        p = [tk(prompt(r['q']), add_special_tokens=False).input_ids for r in mb]
        a = [tk(r['rationale'] + tk.eos_token, add_special_tokens=False).input_ids for r in mb]
        L = max(len(x) + len(y) for x, y in zip(p, a))
        ids = torch.full((len(mb), L), tk.pad_token_id); lab = torch.full((len(mb), L), -100); att = torch.zeros(len(mb), L, dtype=torch.long)
        for j, (x, y) in enumerate(zip(p, a)):
            n = len(x) + len(y); ids[j, :n] = torch.tensor(x + y); lab[j, len(x):n] = torch.tensor(y); att[j, :n] = 1
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = m(input_ids=ids.cuda(), attention_mask=att.cuda()).logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1).cuda(),
                               ignore_index=-100, reduction='sum') / sum(len(y) for y in a) * len(mb) / len(b)
        loss.backward(); del logits
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
    if step % 50 == 0:
        print(step, n_steps, round(loss.item(), 4), flush=True)

m.eval(); m.config.use_cache = True
ans_of = lambda t: (lambda r: r[-1].replace(',', '').rstrip('.') if r else None)(re.findall(r'Answer:\s*[<\$\\{boxed]*\s*(-?[\d,]*\.?\d+)', t))
rows = [json.loads(l) for l in open('data/gsm8k/test.jsonl')][:300]
res = []
for i in range(0, 300, 64):
    chunk = rows[i:i + 64]
    enc = tk([prompt(r['question']) for r in chunk], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        g = m.generate(**enc, max_new_tokens=384, do_sample=False)
    for r, seq in zip(chunk, g[:, enc.input_ids.shape[1]:]):
        t = tk.decode(seq, skip_special_tokens=True); a = ans_of(t); gold = r['answer'].split('####')[-1].strip().replace(',', '')
        res.append(dict(gold=gold, pred=a, ok=a is not None and abs(float(a) - float(gold)) < 1e-6, n_tok=int((seq != tk.pad_token_id).sum())))
with open(out, 'w') as f:
    for r in res: f.write(json.dumps(r) + '\n')
print('sft accuracy', sum(r['ok'] for r in res) / len(res))
