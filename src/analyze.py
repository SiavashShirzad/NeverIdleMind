"""Aggregate rollout results: per-category success, latency, compute; paired bootstrap 95% CIs over episodes."""
import glob
import json
import re
import sys
from collections import defaultdict

import numpy as np

RNG = np.random.default_rng(0)
NB = 2000


def load(path):
    return [json.loads(l) for l in open(path)]


def ep_metrics(r):
    t = r['tasks']
    cat = lambda f: [x['ok'] for x in t if f(x)]
    plain = lambda x: not x['cancelled'] and not x['revised'] and x['kind'] != 'quick'
    ql = [x['latency'] if x['latency'] is not None else np.nan for x in t if x['kind'] == 'quick']
    cl = [x['latency'] for x in t if plain(x) and x.get('latency') is not None]
    return dict(task=np.mean([x['ok'] for x in t]), episode=float(r['ep_ok']),
                quick=np.mean(cat(lambda x: x['kind'] == 'quick')) if cat(lambda x: x['kind'] == 'quick') else np.nan,
                cancel=np.mean(cat(lambda x: x['cancelled'])) if cat(lambda x: x['cancelled']) else np.nan,
                revise=np.mean(cat(lambda x: x['revised'])) if cat(lambda x: x['revised']) else np.nan,
                plain=np.mean(cat(plain)) if cat(plain) else np.nan,
                quick_lat=np.nanmedian(ql) if ql and not all(np.isnan(ql)) else np.nan,
                done_lat=np.median(cl) if cl else np.nan,
                tokens=r['n_tok'], think=r['n_think'], ticks=r['ticks'])


KEYS = ['task', 'episode', 'quick', 'cancel', 'revise', 'plain', 'quick_lat', 'done_lat', 'tokens', 'ticks']


def matrix(files):
    """seeds x episodes x metrics"""
    runs = [[ep_metrics(r) for r in sorted(load(f), key=lambda r: r['ep'])] for f in files]
    n = min(len(r) for r in runs)
    return np.array([[[e[k] for k in KEYS] for e in r[:n]] for r in runs], dtype=float)


def stat(M, idx=None):
    """mean over seeds of nan-mean over episodes; bootstrap over episodes (paired across seeds)."""
    f = lambda X: np.nanmean(np.nanmean(X, 1), 0)
    est = f(M)
    n = M.shape[1]
    bs = np.array([f(M[:, RNG.integers(0, n, n)]) for _ in range(NB)])
    return est, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)


def paired_diff(A, B):
    n = min(A.shape[1], B.shape[1]); A, B = A[:, :n], B[:, :n]
    f = lambda X, ix: np.nanmean(np.nanmean(X[:, ix], 1), 0)
    est = f(A, np.arange(n)) - f(B, np.arange(n))
    bs = []
    for _ in range(NB):
        ix = RNG.integers(0, n, n); bs.append(f(A, ix) - f(B, ix))
    bs = np.array(bs)
    p = 2 * np.minimum((bs <= 0).mean(0), (bs >= 0).mean(0))
    return est, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0), p


def fmt(e, lo, hi, k):
    if k in ('quick_lat', 'done_lat', 'tokens', 'ticks'):
        return f'{e:.0f} [{lo:.0f}, {hi:.0f}]'
    return f'{100 * e:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]'


if __name__ == '__main__':
    split = sys.argv[1] if len(sys.argv) > 1 else 'id'
    pat = sys.argv[2] if len(sys.argv) > 2 else None
    groups = defaultdict(list)
    files = sorted(glob.glob(f'runs/*_{split}.jsonl'))
    full = max(sum(1 for _ in open(f)) for f in files)
    for f in files:
        if sum(1 for _ in open(f)) < full:
            continue
        name = re.sub(rf'_{split}\.jsonl$', '', f.split('/')[-1])
        if name.startswith(('pilot', 'tiny')) or (pat and not re.search(pat, name)):
            continue
        groups[re.sub(r'_s\d+$', '', name)].append(f)
    out = {}
    print('| arm | seeds | ' + ' | '.join(KEYS) + ' |')
    print('|' + '---|' * (len(KEYS) + 2))
    for g, fs in groups.items():
        M = matrix(fs); e, lo, hi = stat(M); out[g] = M
        print(f'| {g} | {len(fs)} | ' + ' | '.join(fmt(e[i], lo[i], hi[i], k) for i, k in enumerate(KEYS)) + ' |')
    np.save(f'runs/summary_{split}.npy', {k: v for k, v in out.items()}, allow_pickle=True)
