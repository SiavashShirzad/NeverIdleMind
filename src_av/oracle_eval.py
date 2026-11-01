"""Score the scripted oracle on the exact test episodes (upper bound per protocol)."""
import json, random, sys
from model import load_tokenizer
from world import PROTOCOLS, Oracle, Tok, gen_episode
T = Tok(load_tokenizer())
for split, n in [('id', 300), ('stress', 300), ('long', 32)]:
    seed0 = {'id': 0, 'stress': 100_000, 'long': 200_000}[split]
    eps = [gen_episode(random.Random(seed0 + i), split) for i in range(n)]
    for p in PROTOCOLS:
        with open(f'runs/oracle_{p}_{split}.jsonl', 'w') as f:
            for i, ep in enumerate(eps):
                o = Oracle(ep, p, T, random.Random(i)); o.run(); r = o.result(); r['ep'] = i
                f.write(json.dumps(r) + '\n')
print('done')
