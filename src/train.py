import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from model import ChannelLM, load_tokenizer
from world import CH

ap = argparse.ArgumentParser()
ap.add_argument('--arm', required=True, help='router | tags | orch | turn | router_noemb')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--data', default='data/train.pt')
ap.add_argument('--tok_per_batch', type=int, default=16384)
ap.add_argument('--lr', type=float, default=5e-5)
ap.add_argument('--lr_new', type=float, default=1e-3)
ap.add_argument('--epochs', type=float, default=1.0)
ap.add_argument('--out', default='ckpt')
ap.add_argument('--suffix', default='', help='checkpoint/log name suffix, e.g. _1.7b')
a = ap.parse_args()

torch.manual_seed(a.seed); random.seed(a.seed)
torch.set_num_threads(8)
proto = 'router' if a.arm.startswith('router') else a.arm
dev = 'cuda'
tk = load_tokenizer()
m = ChannelLM(router=proto == 'router', tk=tk, dtype=torch.bfloat16 if os.environ.get('LORA_R') else torch.float32,
              chan_emb=a.arm != 'router_noemb').to(dev)
m.base.gradient_checkpointing_enable()
m.base.config.use_cache = False
data = torch.load(a.data)[proto]

# length-bucketed batches with a token budget
order = sorted(range(len(data)), key=lambda i: len(data[i]['ids']))
batches, cur = [], []
for i in order:
    if cur and (len(cur) + 1) * len(data[i]['ids']) > a.tok_per_batch:
        batches.append(cur); cur = []
    cur.append(i)
batches.append(cur)
n_steps = int(len(batches) * a.epochs)

new = list(m.chan_emb.parameters()) + list(m.router.parameters()) + ([m.tag_delta] if m.lora else [])
opt = torch.optim.AdamW([{'params': [p for p in m.base.parameters() if p.requires_grad], 'lr': a.lr},
                         {'params': new, 'lr': a.lr_new}], weight_decay=0.0, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(s, n_steps) / n_steps)))


from torch.utils.checkpoint import checkpoint


def _ce(h, y):
    return F.cross_entropy(m.logits(h).float(), y, reduction='sum')


def lm_loss(h, y, chunk=2048):  # chunked + recomputed so the 152k-way logits never sit in memory at once
    tot = sum(checkpoint(_ce, h[i:i + chunk], y[i:i + chunk], use_reentrant=False) for i in range(0, len(y), chunk))
    return tot / max(1, len(y))


def collate(idx):
    L = max(len(data[i]['ids']) for i in idx)
    ids = torch.zeros(len(idx), L, dtype=torch.long); ch = torch.zeros_like(ids)
    mod = torch.zeros(len(idx), L, dtype=torch.bool); att = torch.zeros_like(ids)
    for j, i in enumerate(idx):
        n = len(data[i]['ids'])
        ids[j, :n] = data[i]['ids']; ch[j, :n] = data[i]['chans'].long(); mod[j, :n] = data[i]['model']; att[j, :n] = 1
    return ids.to(dev), ch.to(dev), mod.to(dev), att.to(dev)


log = open(f'logs/train_{a.arm}{a.suffix}_s{a.seed}.jsonl', 'w')
t0, step, tokseen = time.time(), 0, 0
m.train()
ep_order = []
while step < n_steps:
    if not ep_order:
        ep_order = list(range(len(batches))); random.shuffle(ep_order)
    ids, ch, mod, att = collate(batches[ep_order.pop()])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h, _ = m.hidden(ids, ch, att)
        tgt = mod[:, 1:]
        hs = h[:, :-1][tgt]
        nxt_tok, nxt_ch = ids[:, 1:][tgt], ch[:, 1:][tgt]
        if proto == 'router':
            keep = nxt_ch != CH['idle']  # the idle token is implied by the router, not predicted by the LM
            lt = lm_loss(hs[keep], nxt_tok[keep])
            lr_ = F.cross_entropy(m.router(hs).float(), nxt_ch)
            loss = lt + lr_
        else:
            lt = lm_loss(hs, nxt_tok); lr_ = torch.zeros(())
            loss = lt
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step(); sched.step(); step += 1
    tokseen += int(att.sum())
    if step % 20 == 0 or step == n_steps:
        rec = dict(step=step, n=n_steps, loss_tok=lt.item(), loss_route=lr_.item(), tok=tokseen,
                   tps=tokseen / (time.time() - t0), mem=torch.cuda.max_memory_allocated() / 1e9)
        log.write(json.dumps(rec) + '\n'); log.flush()
        if step % 100 == 0 or step == n_steps:
            print(rec, flush=True)
os.makedirs(a.out, exist_ok=True)
torch.save(m.trainable_state(), f'{a.out}/{a.arm}{a.suffix}_s{a.seed}.pt')
print('saved', time.time() - t0)
