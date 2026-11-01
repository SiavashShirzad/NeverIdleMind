"""Build oracle training sequences for each protocol (identical episodes across arms)."""
import argparse
import random
from multiprocessing import Pool

import torch

from world import CH, PROTOCOLS, SYS_PROMPT, Oracle, Tok, gen_episode

W_LEN, M_LEN, L_MAX = 640, 192, 1024  # eviction window, retained-memory budget, context cap
_T = None


def set_pools(split='train'):
    import json, os
    import world
    if split == 'train':
        world.POOLS['math'] = [json.loads(l) for l in open(os.environ.get('RATIONALES', 'data/gsm8k/train_rationales.jsonl'))]
    else:
        rows = [json.loads(l) for l in open('data/gsm8k/test.jsonl')][:300]
        world.POOLS['math'] = [dict(q=r['question'], gold=r['answer'].split('####')[-1].strip().replace(',', '')) for r in rows]
    oj = f'data/vis/{split}_objs.json'
    if not os.path.exists(oj):
        json.dump(torch.load(f'data/vis/{split}.pt')['objs'], open(oj, 'w'))
    world.POOLS['vis'] = [[tuple(o) for o in x] for x in json.load(open(oj))]
    world.POOLS['aud'] = [tuple(x) for x in json.load(open(f'data/aud/{split}_meta.json'))]


def _init():
    global _T
    from model import load_tokenizer
    _T = Tok(load_tokenizer())
    set_pools('train')


def sys_tokens(T):
    return T.tk(SYS_PROMPT, add_special_tokens=False).input_ids


def evict_view(stream, s, mode):
    """Context after eviction at index s: retained mem-channel model tokens (mode='mem') or none ('plain')."""
    if mode != 'mem':
        return []
    return [x for x in stream[:s] if x[2] and x[1] == CH['mem']][-M_LEN:]


def pack(T, items, loss_from=0):
    st = [(t, CH['sys'], False, -1) for t in sys_tokens(T)]
    seq = st + items
    off = len(st) + loss_from
    return ([x[0] for x in seq], [x[1] for x in seq], [bool(x[2]) and i >= off for i, x in enumerate(seq)], [x[3] for x in seq])


def _job(args):
    seed, cfg, n_views = args
    ep = gen_episode(random.Random(seed), cfg)
    out = {}
    for p in PROTS:
        stream = Oracle(ep, p, _T, random.Random(seed * 7 + 1)).run()
        if cfg != 'long':
            out[p] = [pack(_T, stream)]
            continue
        rng, ex = random.Random(seed * 13 + 5), []
        for _ in range(n_views):
            e = rng.randint(W_LEN, len(stream))
            s = e - W_LEN
            mode = rng.choice(['mem', 'plain'])
            kept = evict_view(stream, s, mode)
            ex.append(pack(_T, kept + stream[s:e], loss_from=len(kept)))
        out[p] = ex
    return out


PROTS = ['router']


def build(n, cfg, seed0, n_views=4, procs=40):
    with Pool(procs, initializer=_init) as pool:
        res = pool.map(_job, [(seed0 + i, cfg, n_views) for i in range(n)], chunksize=16)
    return {p: [dict(ids=torch.tensor(a, dtype=torch.int32), chans=torch.tensor(b, dtype=torch.int8),
                     model=torch.tensor(c, dtype=torch.bool), img=torch.tensor(d, dtype=torch.int32))
                for r in res for a, b, c, d in r[p]] for p in PROTS}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_id', type=int, default=12000)
    ap.add_argument('--n_long', type=int, default=600)
    ap.add_argument('--n_vis', type=int, default=0)
    ap.add_argument('--n_look', type=int, default=0)
    ap.add_argument('--n_aud', type=int, default=0)
    ap.add_argument('--n_senses', type=int, default=0)
    ap.add_argument('--mix', default='av')
    ap.add_argument('--out', default='data/train_av.pt')
    a = ap.parse_args()
    set_pools('train')
    parts = [build(n, cfg, s0) for n, cfg, s0 in [(a.n_id, a.mix, 30_000_000), (a.n_vis, 'vis', 40_000_000),
                                                  (a.n_look, 'look', 50_000_000), (a.n_aud, 'aud', 60_000_000),
                                                  (a.n_senses, 'senses', 70_000_000)] if n]
    data = {p: sum((d[p] for d in parts), []) for p in PROTS}
    for p in PROTS:
        L = [len(x['ids']) for x in data[p]]
        print(p, len(L), 'tokens', sum(L), 'max', max(L), 'model-tok', sum(int(x['model'].sum()) for x in data[p]))
    torch.save(data, a.out)
