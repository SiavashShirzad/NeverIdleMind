import sys, random, collections
sys.path.insert(0, 'src')
from transformers import AutoTokenizer
from world import *
tk = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); tk.add_special_tokens({'additional_special_tokens': TAGS})
T = Tok(tk); C = collections.Counter(); shown = 0
for i in range(200):
    ep = gen_episode(random.Random(i), 'id'); o = Oracle(ep, 'router', T, random.Random(i + 7)); o.run(); r = o.result()
    for t in r['tasks']:
        if not t['ok']:
            C[(t['kind'], t['cancelled'], t['revised'])] += 1
            if shown < 3:
                shown += 1; d = ep['tasks'][t['tid']]; print(t, {k: v for k, v in d.items() if k != 'lat'}, 'exp', expected(d, ep['kb']))
                print([l for l in o.env.lines if l[1].startswith(f"T{t['tid']}")])
print(C)
