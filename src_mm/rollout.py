"""Batched closed-loop evaluation: every sequence advances exactly one token per forward step,
either a forced input token (an event being 'heard') or a model-generated token."""
import argparse
import json
import random
import time
from collections import deque

import torch
from transformers import DynamicCache

from data import L_MAX, W_LEN, evict_view, sys_tokens
from model import ChannelLM, load_tokenizer
from world import CH, N_OUT, N_VIS, TAG_FORMAT, Env, Tok, gen_episode, score
from data import set_pools

FIRST_SPECIAL = 151643


class Seq:
    def __init__(self, ep, protocol, T, max_steps):
        self.ep, self.T, self.tagfmt = ep, T, TAG_FORMAT[protocol]
        self.env = Env(ep, protocol)
        self.queue, self.hist = deque(), []
        self.cur_chan = CH['think']
        self.buf = {c: [] for c in range(N_OUT)}
        self.n_model = self.n_think = self.steps = 0
        self.done, self.max_steps = False, max_steps
        self.tag_chan = {v: CH[k] for k, v in T.tag.items()}

    def enqueue(self, evs):
        for e in evs:
            ch = 'tool' if e['type'] == 'result' else 'user'
            if self.tagfmt:
                self.queue.append((self.T.tag[ch], CH[ch], -1))
            for x in self.T.line(e['text']):
                self.queue.append((x, CH[ch], -1))
            if 'img' in e:
                for k in range(N_VIS):
                    self.queue.append((self.T.img, CH['see'], e['img'] * N_VIS + k))

    def generated(self, tok, chan):
        """Apply a model token to the environment. chan is the output channel (router) or None (tags)."""
        env, T = self.env, self.T
        env.clock += 1; self.n_model += 1; self.steps += 1
        tag_chan = self.tag_chan.get(tok)
        if chan is None:  # tag formats: a tag token switches channel
            chan = tag_chan if tag_chan is not None else self.cur_chan
        if chan == CH['idle']:
            self.enqueue(env.on_idle())
            if env.done:
                self.done = True
            return CH['idle']
        boundary = False
        if tag_chan is not None:
            self.cur_chan = chan
        else:
            if chan == CH['think']:
                self.n_think += 1
            self.buf[chan].append(tok)
            text = T.tk.decode(self.buf[chan])
            if '\n' in text:
                env.on_line(chan, text.split('\n', 1)[0].strip())
                self.buf[chan] = []
                boundary = True
        self.enqueue(env.deliver('tok', boundary, chan))
        if self.steps >= self.max_steps:
            self.done = True
        return chan

    def result(self):
        return score(self.ep, self.env.lines, self.n_model, self.n_think, self.env.clock)


@torch.no_grad()
def run_batch(m, T, eps, protocol, evict='full', max_steps=3000, dev='cuda'):
    router = protocol == 'router'
    B = len(eps)
    seqs = [Seq(ep, protocol, T, max_steps) for ep in eps]
    sys_t = sys_tokens(T)
    V = m.base.config.vocab_size
    mask = torch.zeros(V, device=dev)
    mask[FIRST_SPECIAL:] = float('-inf')
    if not router:
        for k in ['think', 'say', 'write', 'code', 'exec', 'mem', 'idle']:
            mask[T.tag[k]] = 0
    idle_id = T.tag['idle']

    def prefill(views):
        L = max(len(v) for v in views)
        ids = torch.zeros(B, L, dtype=torch.long); ch = torch.zeros_like(ids); att = torch.zeros_like(ids)
        pos = torch.zeros_like(ids)
        img = torch.full_like(ids, -1)
        for j, v in enumerate(views):
            n = len(v); o = L - n
            ids[j, o:] = torch.tensor([x[0] for x in v]); ch[j, o:] = torch.tensor([x[1] for x in v])
            img[j, o:] = torch.tensor([x[3] for x in v])
            att[j, o:] = 1; pos[j, o:] = torch.arange(n)
        past = DynamicCache()
        h, past = m.hidden(ids.to(dev), ch.to(dev), att.to(dev), pos.to(dev), past, True, img.to(dev))
        return past, att.to(dev), torch.tensor([len(v) for v in views], device=dev), h[:, -1]

    past, att, npos, hlast = prefill([[(t, CH['sys'], False, -1) for t in sys_t]] * B)
    while not all(s.done for s in seqs):
        if evict != 'full' and max(len(s.hist) for s in seqs) + len(sys_t) > L_MAX:
            views = []
            for s in seqs:
                if len(s.hist) + len(sys_t) > L_MAX - 64:
                    cut = len(s.hist) - W_LEN
                    s.hist = evict_view(s.hist, cut, evict) + s.hist[cut:]
                views.append([(t, CH['sys'], False, -1) for t in sys_t] + list(s.hist))
            del past; torch.cuda.empty_cache()
            past, att, npos, hlast = prefill(views)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, rl = m.heads(hlast)
        tok_pred = (logits.float() + mask).argmax(-1).tolist()
        ch_pred = rl.argmax(-1).tolist() if router else [None] * B
        ids, chs, live, imgs = [], [], [], []
        for j, s in enumerate(seqs):
            if s.done:
                ids.append(0); chs.append(0); live.append(0); imgs.append(-1); continue
            im = -1
            if s.queue:
                t, c, im = s.queue.popleft(); s.hist.append((t, c, False, im))
            else:
                if router and ch_pred[j] == CH['idle']:
                    t = idle_id
                else:
                    t = tok_pred[j]
                c = s.generated(t, ch_pred[j])
                s.hist.append((t, c, True, -1))
            ids.append(t); chs.append(c); live.append(1); imgs.append(im)
        live_t = torch.tensor(live, device=dev)
        att = torch.cat([att, live_t[:, None]], 1)
        pos = npos[:, None].clone()
        npos = npos + live_t
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h, past = m.hidden(torch.tensor(ids, device=dev)[:, None], torch.tensor(chs, device=dev)[:, None],
                               att, pos, past, True, torch.tensor(imgs, device=dev)[:, None])
        hlast = h[:, -1]
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--split', default='id')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--evict', default='full')
    ap.add_argument('--protocol', default=None, help='override delivery protocol at test time')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    seed0 = {'id': 0, 'stress': 100_000, 'long': 200_000, 'math_async': 300_000, 'math_solo': 300_000, 'vis': 400_000}[a.split]
    set_pools('test')
    proto = a.protocol or ('router' if a.arm.startswith('router') else a.arm)
    T = Tok(load_tokenizer())
    m = ChannelLM(router=proto == 'router', tk=T.tk, dtype=torch.bfloat16,
                  chan_emb='noemb' not in a.arm).cuda().eval()
    if a.ckpt:
        m.load_state_dict(torch.load(a.ckpt, map_location='cuda'))
    m.vis_bank = torch.load('data/vis/test.pt')['feats'].flatten(0, 1).cuda()
    math = a.split.startswith('math')
    eps = [gen_episode(random.Random(seed0 + i), a.split, math_idx=i if math else None) for i in range(a.n)]
    res, t0 = [], time.time()
    max_steps = 9000 if a.split == 'long' else 3000
    with open(a.out, 'w') as f:
        for i in range(0, a.n, a.bs):
            seqs = run_batch(m, T, eps[i:i + a.bs], proto, a.evict, max_steps)
            for k, s in enumerate(seqs):
                r = s.result(); r['ep'] = i + k
                if i == 0 and k < 3:  # keep a few raw traces for qualitative inspection
                    r['trace'] = [(T.tk.decode([t]), c, mm) for t, c, mm, _ in s.hist]
                f.write(json.dumps(r) + '\n'); res.append(r)
            ok = sum(t['ok'] for r in res for t in r['tasks']) / sum(len(r['tasks']) for r in res)
            print(f'{len(res)}/{a.n} task_ok={ok:.3f} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
