import sys, random, statistics as S
sys.path.insert(0, 'src')
from transformers import AutoTokenizer
from world import *
tk = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); tk.add_special_tokens({'additional_special_tokens': TAGS})
T = Tok(tk)
for cfg in ['id', 'stress', 'long']:
    for p in PROTOCOLS:
        res = []; lens = []
        for i in range(60 if cfg != 'long' else 8):
            ep = gen_episode(random.Random(i), cfg)
            o = Oracle(ep, p, T, random.Random(i + 7)); s = o.run(); lens.append(len(s)); res.append(o.result())
        tasks = [t for r in res for t in r['tasks']]
        q = [t['latency'] for t in tasks if t['kind'] == 'quick' and t['latency'] is not None]
        print(f"{cfg:6s} {p:6s} len={S.mean(lens):6.0f} max={max(lens):5d} task_ok={S.mean(t['ok'] for t in tasks):.3f} "
              f"ep_ok={S.mean(r['ep_ok'] for r in res):.2f} quick_lat_med={S.median(q) if q else None} ticks={S.mean(r['ticks'] for r in res):.0f}")
ep = gen_episode(random.Random(3), 'id'); o = Oracle(ep, 'router', T, random.Random(1)); s = o.run()
out = []; prev = None
for tok, ch, m in s:
    if ch != prev: out.append(f"\n[{CH_NAMES[ch]}{'' if m else '*'}] "); prev = ch
    out.append(tk.decode([tok]).replace('\n', '⏎'))
print(''.join(out)[:2500])
