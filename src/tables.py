"""Write LaTeX tables and number macros (paper/gen/) from runs/*.jsonl."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, 'src')
from analyze import KEYS, NB, RNG, load, matrix, paired_diff, stat

os.makedirs('paper/gen', exist_ok=True)
ARMS = [('router', 'Router (ours)'), ('tags', 'In-stream tags'), ('orch', 'Orchestrator'), ('turn', 'Turn-based')]
PCT = {'task', 'episode', 'quick', 'cancel', 'revise', 'plain'}
macros = {}


def complete(pat, n):
    return [f for f in sorted(glob.glob(pat)) if sum(1 for _ in open(f)) >= n]


def cell(e, lo, hi, k, ci=True):
    if k in PCT:
        s = f'{100 * e:.1f}'
        return s + (f' {{\\scriptsize[{100 * lo:.1f}, {100 * hi:.1f}]}}' if ci else '')
    return f'{e:.0f}' + (f' {{\\scriptsize[{lo:.0f}, {hi:.0f}]}}' if ci else '')


def mac(name, v):
    macros[name] = v


def main_table():
    cols = ['task', 'episode', 'quick', 'cancel', 'revise', 'quick_lat', 'tokens']
    head = 'Arm & Seeds & Task & Episode & Quick & Cancel & Revise & Quick lat. & Tokens \\\\'
    out = []
    for split, title in [('id', 'In-distribution (300 episodes, 3--5 tasks)'), ('stress', 'Stress (300 episodes, 6--8 tasks)')]:
        out.append(f'\\midrule\\multicolumn{{9}}{{l}}{{\\textit{{{title}}}}}\\\\')
        for arm, lab in ARMS:
            fs = complete(f'runs/{arm}_s[0-9]_{split}.jsonl', 300)
            if not fs:
                continue
            e, lo, hi = stat(matrix(fs))
            for k in cols:
                mac(f'{arm}{split}{k}'.replace('_', ''), cell(e[KEYS.index(k)], 0, 0, k, False))
            out.append(f'{lab} & {len(fs)} & ' + ' & '.join(cell(e[KEYS.index(k)], lo[KEYS.index(k)], hi[KEYS.index(k)], k) for k in cols) + ' \\\\')
            o = stat(matrix([f'runs/oracle_{arm}_{split}.jsonl']))[0]
            for k in cols:
                mac(f'oracle{arm}{split}{k}'.replace('_', ''), cell(o[KEYS.index(k)], 0, 0, k, False))
            out.append(f'\\quad oracle & -- & ' + ' & '.join('{\\color{gray}' + cell(o[KEYS.index(k)], 0, 0, k, False) + '}' for k in cols) + ' \\\\')
    open('paper/gen/tab_main.tex', 'w').write(
        '\\begin{tabular}{llrrrrrrr}\n\\toprule\n' + head + '\n' + '\n'.join(out) + '\n\\bottomrule\n\\end{tabular}\n')


def diff_table():
    cols = ['task', 'episode', 'quick', 'quick_lat', 'tokens']
    out = []
    for split in ['id', 'stress']:
        R = matrix(complete(f'runs/router_s[0-9]_{split}.jsonl', 300))
        for arm, lab in ARMS[1:] + [('router_noemb', 'Router w/o channel emb.')]:
            fs = complete(f'runs/{arm}_s[0-9]_{split}.jsonl', 300)
            if not fs:
                continue
            e, lo, hi, p = paired_diff(R, matrix(fs))
            row = []
            for k in cols:
                j = KEYS.index(k); s = 100 if k in PCT else 1
                pp = '$<$0.001' if p[j] < 1e-3 else f'{p[j]:.3f}'
                row.append(f'{s * e[j]:+.1f} {{\\scriptsize[{s * lo[j]:+.1f}, {s * hi[j]:+.1f}]; p={pp}}}')
                mac(f'd{arm}{split}{k}'.replace('_', ''), f'{s * e[j]:+.1f}')
                mac(f'd{arm}{split}{k}ci'.replace('_', ''), f'[{s * lo[j]:+.1f}, {s * hi[j]:+.1f}]')
            out.append(f'{split} & {lab} & ' + ' & '.join(row) + ' \\\\')
    open('paper/gen/tab_diff.tex', 'w').write(
        '\\begin{tabular}{llrrrrr}\n\\toprule\nSplit & Router minus & $\\Delta$Task (pp) & $\\Delta$Episode (pp) & '
        '$\\Delta$Quick (pp) & $\\Delta$Quick lat. & $\\Delta$Tokens \\\\\n\\midrule\n' + '\n'.join(out) + '\n\\bottomrule\n\\end{tabular}\n')


def long_table():
    cols = ['task', 'episode', 'revise', 'cancel', 'plain', 'tokens']
    out, mats = [], {}
    for arm, lab in ARMS[:2]:
        for ev, el in [('full', 'none (full context)'), ('mem', 'keep mem-routed'), ('plain', 'sliding window')]:
            fs = complete(f'runs/{arm}_s[0-9]_long_{ev}.jsonl', 32)
            if not fs:
                continue
            M = matrix(fs); mats[(arm, ev)] = M
            e, lo, hi = stat(M)
            for k in cols:
                mac(f'long{arm}{ev}{k}', cell(e[KEYS.index(k)], 0, 0, k, False))
            out.append(f'{lab} & {el} & {len(fs)} & ' + ' & '.join(cell(e[KEYS.index(k)], lo[KEYS.index(k)], hi[KEYS.index(k)], k) for k in cols) + ' \\\\')
    for arm in ['router', 'tags']:
        if (arm, 'mem') in mats and (arm, 'plain') in mats:
            e, lo, hi, p = paired_diff(mats[(arm, 'mem')], mats[(arm, 'plain')])
            for k in ['task', 'revise', 'cancel', 'episode']:
                j = KEYS.index(k)
                mac(f'dlong{arm}{k}', f'{100 * e[j]:+.1f}')
                mac(f'dlong{arm}{k}ci', f'[{100 * lo[j]:+.1f}, {100 * hi[j]:+.1f}]')
                mac(f'dlong{arm}{k}p', '$<$0.001' if p[j] < 1e-3 else f'{p[j]:.3f}')
    open('paper/gen/tab_long.tex', 'w').write(
        '\\begin{tabular}{llrrrrrrr}\n\\toprule\nArm & Eviction & Seeds & Task & Episode & Revise & Cancel & Plain & Tokens \\\\\n\\midrule\n'
        + '\n'.join(out) + '\n\\bottomrule\n\\end{tabular}\n')


def boot(x):
    x = np.asarray(x, float); n = len(x)
    b = [x[RNG.integers(0, n, n)].mean() for _ in range(NB)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def pct(t):
    e, lo, hi = t
    return f'{100 * e:.1f} {{\\scriptsize[{100 * lo:.1f}, {100 * hi:.1f}]}}'


def mm_table():
    rows = []
    for fn, lab, key in [('base_test_nothink', 'non-thinking mode', 'gsmbase'), ('base_test_think', 'thinking mode', 'gsmbasethink')]:
        p = f'data/gsm8k/{fn}.jsonl'
        if os.path.exists(p) and sum(1 for _ in open(p)) >= 300:
            base = [json.loads(l) for l in open(p)][:300]
            bk = 'ok' if 'ok' in base[0] else 'correct'
            b = boot([float(r[bk]) for r in base])
            rows.append((f'GSM8K, base Qwen3-0.6B, {lab}', 300, pct(b), '--')); mac(key, f'{100 * b[0]:.1f}')
    acc = lambda rs, f: [float(t['ok']) for r in rs for t in r['tasks'] if f(t)]
    res = {}
    for arm, pre in [('router_mmv', 'v'), ('router_mm', 'n')]:
        for split in ['math_solo', 'math_async', 'vis', 'id']:
            f = f'runs/{arm}_s0_{split}.jsonl'
            if os.path.exists(f) and sum(1 for _ in open(f)) >= 300:
                res[(pre, split)] = sorted(load(f), key=lambda r: r['ep'])
    items = [('GSM8K, think channel, no interruptions', 'math_solo', lambda t: t['kind'] == 'math', 'gsmsolo'),
             ('GSM8K, 2 quick tasks arrive mid-thought', 'math_async', lambda t: t['kind'] == 'math', 'gsmasync'),
             ('\\quad quick tasks within deadline', 'math_async', lambda t: t['kind'] == 'quick', 'gsmasyncquick'),
             ('Visual QA, all questions', 'vis', lambda t: t['kind'] == 'look', 'visall')]
    items += [(f'\\quad {ql}', 'vis', (lambda q: lambda t: t.get('qk') == q)(qk), 'vis' + qk.replace('_', ''))
              for qk, ql in [('count_color', 'count by colour'), ('count_shape', 'count by shape'),
                             ('exists', 'existence'), ('color_of', 'colour of unique shape')]]
    items += [('\\quad concurrent non-visual tasks', 'vis', lambda t: t['kind'] != 'look', 'visother'),
              ('AsyncWorld id episodes, all tasks', 'id', lambda t: True, 'mmidtask')]
    for lab, split, f, key in items:
        cells, n = [], 0
        for pre in ['v', 'n']:
            if (pre, split) in res:
                x = acc(res[(pre, split)], f); t = boot(x); n = len(x)
                cells.append(pct(t)); mac(key + ('' if pre == 'v' else 'noalign'), f'{100 * t[0]:.1f}')
            else:
                cells.append('--')
        rows.append((lab, n, cells[0], cells[1]))
    for pre in ['v', 'n']:
        if (pre, 'math_solo') in res and (pre, 'math_async') in res:
            a = np.array(acc(res[(pre, 'math_async')], lambda t: t['kind'] == 'math'))
            s_ = np.array(acc(res[(pre, 'math_solo')], lambda t: t['kind'] == 'math'))
            d = boot(a - s_); sfx = '' if pre == 'v' else 'noalign'
            mac('gsmdiff' + sfx, f'{100 * d[0]:+.1f}'); mac('gsmdiffci' + sfx, f'[{100 * d[1]:+.1f}, {100 * d[2]:+.1f}]')
    p = 'data/gsm8k/base_test_think.jsonl'
    if ('v', 'math_solo') in res and os.path.exists(p):
        bt = [json.loads(l) for l in open(p)][:300]
        mac('gsmthinktrunc', f'{100 * np.mean([r["pred"] is None for r in bt]):.0f}')
        m_ = np.array(acc(res[('v', 'math_solo')], lambda t: t['kind'] == 'math'))
        d = boot(m_ - np.array([float(r['ok']) for r in bt]))
        mac('gsmvsthink', f'{100 * d[0]:+.1f}'); mac('gsmvsthinkci', f'[{100 * d[1]:+.1f}, {100 * d[2]:+.1f}]')
        if 'gsmbasethink' in macros:
            mac('gsmthinkclause', f' and {macros["gsmbasethink"]}\\% in its built-in thinking mode (up to 1536 generated tokens; {macros.get("gsmthinktrunc", "?")}\\% of outputs without a parseable answer)')
        mac('gsmthinktok', f'{np.median([r["n_think"] for r in res[("v", "math_solo")]]):.0f}')
    open('paper/gen/tab_mm.tex', 'w').write(
        '\\begin{tabular}{lrrr}\n\\toprule\nEvaluation & $n$ & router\\_mm (two-stage) & no alignment stage \\\\\n\\midrule\n'
        + '\n'.join(f'{a} & {n} & {c} & {d} \\\\' for a, n, c, d in rows) + '\n\\bottomrule\n\\end{tabular}\n')


def seed_table():
    from scipy import stats
    out = []
    for split in ['id', 'stress']:
        for arm, lab in ARMS + [('router_noemb', 'Router w/o channel emb.')]:
            fs = complete(f'runs/{arm}_s[0-9]_{split}.jsonl', 300)
            if not fs:
                continue
            v = np.array([100 * np.nanmean(matrix([f])[0, :, 0]) for f in fs])
            h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            mac(f'seed{arm}{split}sd'.replace('_', ''), f'{v.std(ddof=1):.1f}')
            out.append(f'{split} & {lab} & {len(v)} & ' + ', '.join(f'{x:.1f}' for x in v) +
                       f' & {v.mean():.1f} $\\pm$ {v.std(ddof=1):.1f} & [{v.mean() - h:.1f}, {v.mean() + h:.1f}] \\\\')
    open('paper/gen/tab_seeds.tex', 'w').write(
        '\\begin{tabular}{llrlrr}\n\\toprule\nSplit & Arm & Seeds & Task success per seed (\\%) & Mean $\\pm$ SD & 95\\% CI (seeds, $t$) \\\\\n\\midrule\n'
        + '\n'.join(out) + '\n\\bottomrule\n\\end{tabular}\n')


def holm():
    # confirmatory family: router vs each comparator on in-distribution task and quick success
    R = matrix(complete('runs/router_s[0-9]_id.jsonl', 300))
    ps = []
    for arm in ['tags', 'orch', 'turn']:
        e, lo, hi, p = paired_diff(R, matrix(complete(f'runs/{arm}_s[0-9]_id.jsonl', 300)))
        for k in ['task', 'quick']:
            ps.append((f'holm{arm}{k}', max(p[KEYS.index(k)], 1 / NB)))
    order = sorted(range(len(ps)), key=lambda i: ps[i][1]); adj, run = {}, 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[i][1])); adj[ps[i][0]] = run
    for k, v in adj.items():
        mac(k, '$<$0.003' if v <= 6 / NB else f'{v:.3f}')


if __name__ == '__main__':
    main_table(); diff_table(); long_table(); mm_table(); seed_table(); holm()
    with open('paper/gen/numbers.tex', 'w') as f:
        for k, v in macros.items():
            f.write(f'\\newcommand{{\\{"".join(c for c in k if c.isalpha())}}}{{{v}}}\n')
    print(len(macros), 'macros')
