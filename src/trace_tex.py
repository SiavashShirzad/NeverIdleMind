"""Render the first test-episode trace of a run as a LaTeX table (paper appendix)."""
import json
import sys

sys.path.insert(0, 'src')
from world import CH_NAMES

f = sys.argv[1] if len(sys.argv) > 1 else 'runs/router_s0_id.jsonl'
tr = json.loads(open(f).readline())['trace']
segs = []
for tok, c, model in tr:
    n = CH_NAMES[c]
    if segs and segs[-1][0] == n and segs[-1][1] == model:
        segs[-1][2] += tok
    else:
        segs.append([n, model, tok])
esc = lambda t: ''.join({'\\': r'\textbackslash{}', '_': r'\_', '%': r'\%', '&': r'\&', '#': r'\#', '$': r'\$', '{': r'\{',
                         '}': r'\}', '^': r'\^{}', '~': r'\~{}', '<': r'\textless{}', '>': r'\textgreater{}'}.get(ch, ch) for ch in t)
rows = []
for n, model, t in segs:
    if n == 'sys':
        continue
    body = esc(t).replace('\n', r' $\hookleftarrow$').strip()
    who = r'\textbf{model}' if model else r'\textit{input}'
    rows.append(f'{who} & \\texttt{{{n}}} & \\texttt{{{body}}} \\\\')
open('paper/gen/trace.tex', 'w').write('\\begin{tabular}{llp{11.5cm}}\n\\toprule\nSource & Channel & Tokens \\\\\n\\midrule\n'
                                       + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')
print(len(rows), 'rows')
