"""Build oracle training sequences for each protocol (identical episodes across arms)."""
import argparse
import random
from multiprocessing import Pool

import torch

from world import CH, PROTOCOLS, SYS_PROMPT, Oracle, Tok, gen_episode

W_LEN, M_LEN, L_MAX = 640, 192, 1024  # eviction window, retained-memory budget, context cap
_T = None


def _init():
    global _T
    from model import load_tokenizer
    _T = Tok(load_tokenizer())


def sys_tokens(T):
    return T.tk(SYS_PROMPT, add_special_tokens=False).input_ids


def evict_view(stream, s, mode):
    """Context after eviction at index s: retained mem-channel model tokens (mode='mem') or none ('plain')."""
    if mode != 'mem':
        return []
    return [x for x in stream[:s] if x[2] and x[1] == CH['mem']][-M_LEN:]


def pack(T, items, loss_from=0):
    st = [(t, CH['sys'], False) for t in sys_tokens(T)]
    seq = st + items
    off = len(st) + loss_from
    return ([x[0] for x in seq], [x[1] for x in seq], [bool(x[2]) and i >= off for i, x in enumerate(seq)])


def _job(args):
    seed, cfg, n_views = args
    ep = gen_episode(random.Random(seed), cfg)
    out = {}
    for p in PROTOCOLS:
        stream = Oracle(ep, p, _T, random.Random(seed * 7 + 1)).run()
        if not cfg.startswith('long'):
            out[p] = [pack(_T, stream)]
            continue
        rng, ex = random.Random(seed * 13 + 5), []
        for _ in range(n_views):
            e = rng.randint(W_LEN, len(stream))
            s = e - W_LEN
            mode = 'mem' if cfg == 'longfar' else rng.choice(['mem', 'plain'])  # long-gap views only with retention
            kept = evict_view(stream, s, mode)
            ex.append(pack(_T, kept + stream[s:e], loss_from=len(kept)))
        out[p] = ex
    return out


def build(n, cfg, seed0, n_views=4, procs=40):
    with Pool(procs, initializer=_init) as pool:
        res = pool.map(_job, [(seed0 + i, cfg, n_views) for i in range(n)], chunksize=16)
    return {p: [dict(ids=torch.tensor(a, dtype=torch.int32), chans=torch.tensor(b, dtype=torch.int8),
                     model=torch.tensor(c, dtype=torch.bool)) for r in res for a, b, c in r[p]] for p in PROTOCOLS}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_id', type=int, default=12000)
    ap.add_argument('--n_long', type=int, default=600)
    ap.add_argument('--n_longfar', type=int, default=0)
    ap.add_argument('--base', default=None, help='existing data file to extend')
    ap.add_argument('--out', default='data/train.pt')
    a = ap.parse_args()
    if a.base:
        data = torch.load(a.base)
    else:
        d1 = build(a.n_id, 'id', 10_000_000)
        d2 = build(a.n_long, 'long', 20_000_000)
        data = {p: d1[p] + d2[p] for p in PROTOCOLS}
    if a.n_longfar:
        d3 = build(a.n_longfar, 'longfar', 25_000_000)
        data = {p: data[p] + d3[p] for p in PROTOCOLS}
    for p in PROTOCOLS:
        L = [len(x['ids']) for x in data[p]]
        print(p, len(L), 'tokens', sum(L), 'max', max(L), 'model-tok', sum(int(x['model'].sum()) for x in data[p]))
    torch.save(data, a.out)
