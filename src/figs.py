"""Paper figures from runs/*.jsonl. Palette validated with dataviz validate_palette.js (contrast WARN -> direct labels)."""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, 'src')
from analyze import KEYS, load, matrix, stat

COL = dict(router='#2a78d6', tags='#eb6834', orch='#1baf7a', turn='#eda100')
LAB = dict(router='Router (ours)', tags='In-stream tags', orch='Orchestrator', turn='Turn-based')
INK, MUTED, GRID = '#1f2328', '#59636e', '#e6e8eb'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'DejaVu Sans', 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
os.makedirs('paper/figs', exist_ok=True)


def complete(pat, n):
    return [f for f in sorted(glob.glob(pat)) if sum(1 for _ in open(f)) >= n]


def fig_success():
    cats = ['task', 'quick', 'cancel', 'revise', 'plain']
    names = ['All tasks', 'Quick\n(24-tick deadline)', 'Cancel', 'Revise', 'Plain']
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, split in zip(axs, ['id', 'stress']):
        w = 0.19
        for k, arm in enumerate(COL):
            fs = complete(f'runs/{arm}_s[0-9]_{split}.jsonl', 300)
            if not fs:
                continue
            e, lo, hi = stat(matrix(fs))
            orc = stat(matrix([f'runs/oracle_{arm}_{split}.jsonl']))[0]
            for i, c in enumerate(cats):
                j = KEYS.index(c); x = i + (k - 1.5) * w
                ax.bar(x, 100 * e[j], w * 0.9, color=COL[arm], label=LAB[arm] if i == 0 else None, zorder=2)
                ax.errorbar(x, 100 * e[j], yerr=[[100 * (e[j] - lo[j])], [100 * (hi[j] - e[j])]], color=INK, lw=0.8,
                            capsize=1.5, zorder=3)
                ax.plot([x - w * 0.45, x + w * 0.45], [100 * orc[j]] * 2, color=INK, lw=0.8, ls=':', zorder=4)
        ax.set_xticks(range(len(cats)), names)
        ax.set_title({'id': 'In-distribution (3-5 tasks)', 'stress': 'Stress (6-8 tasks, OOD)'}[split], color=INK)
        ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_ylim(0, 104)
    axs[0].set_ylabel('Success (%)')
    h, l = axs[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=INK, ls=':', lw=0.8)); l.append('Oracle ceiling (same protocol)')
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.savefig('paper/figs/success.pdf'); fig.savefig('paper/figs/success.png')


def fig_latency():
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for arm in COL:
        lat = []
        for f in complete(f'runs/{arm}_s[0-9]_stress.jsonl', 300):
            for r in load(f):
                lat += [t['latency'] if t.get('latency') is not None else np.inf for t in r['tasks'] if t['kind'] == 'quick']
        if not lat:
            continue
        x = np.sort(np.array(lat, float)); y = np.arange(1, len(x) + 1) / len(x)
        med = np.median(x)
        ax.step(np.minimum(x, 1000), y, where='post', color=COL[arm], lw=1.6,
                label=f'{LAB[arm]} (median {med:.0f})')
    ax.axvline(24, color=MUTED, lw=0.8, ls='--'); ax.text(25, 0.08, 'deadline', color=MUTED, fontsize=7)
    ax.set_xscale('log'); ax.set_xlim(3, 1000); ax.set_ylim(0, 1.02)
    ax.set_xlabel('Quick-task response latency (ticks, log)'); ax.set_ylabel('Cumulative fraction')
    ax.yaxis.grid(True, color=GRID)
    ax.legend(frameon=False, fontsize=6.5, loc='center right', bbox_to_anchor=(1.0, 0.62))
    fig.savefig('paper/figs/latency.pdf'); fig.savefig('paper/figs/latency.png')


def long_rows():
    rows = {}
    for arm in ['router', 'tags']:
        for ev in ['mem', 'plain', 'full']:
            fs = complete(f'runs/{arm}_s[0-9]_long_{ev}.jsonl', 32)
            if fs:
                rows[(arm, ev)] = (len(fs), stat(matrix(fs)))
    return rows


def fig_long():
    rows = long_rows()
    if not rows:
        return
    cats = ['task', 'revise', 'cancel', 'plain']
    evs = [('full', 'Full context (no eviction)'), ('mem', 'Evict, keep mem-routed'), ('plain', 'Evict, sliding window')]
    shade = dict(full=1.0, mem=0.75, plain=0.45)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
    for ax, arm in zip(axs, ['router', 'tags']):
        for k, (ev, lab) in enumerate(evs):
            if (arm, ev) not in rows:
                continue
            _, (e, lo, hi) = rows[(arm, ev)]
            for i, c in enumerate(cats):
                j = KEYS.index(c); x = i + (k - 1) * 0.26
                ax.bar(x, 100 * e[j], 0.24, color=COL[arm], alpha=shade[ev], label=lab if i == 0 else None, zorder=2)
                ax.errorbar(x, 100 * e[j], yerr=[[100 * (e[j] - lo[j])], [100 * (hi[j] - e[j])]], color=INK, lw=0.8,
                            capsize=1.5, zorder=3)
        ax.set_xticks(range(len(cats)), ['All tasks', 'Revise', 'Cancel', 'Plain'])
        ax.set_title(LAB[arm], color=INK)
        ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_ylim(0, 104)
    axs[0].set_ylabel('Success (%)')
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=shade[e]) for e, _ in evs], [l for _, l in evs], loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.1))
    fig.savefig('paper/figs/long.pdf'); fig.savefig('paper/figs/long.png')


if __name__ == '__main__':
    fig_success(); fig_latency(); fig_long()
    print('ok')
