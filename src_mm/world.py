"""AsyncWorld: procedurally generated asynchronous multi-task environment,
an oracle policy that writes ideal channel-labelled traces under each protocol,
and the scoring used for every arm."""
import os
import random
import re

CH_NAMES = ['think', 'say', 'write', 'code', 'exec', 'mem', 'idle', 'user', 'tool', 'sys', 'see']
CH = {n: i for i, n in enumerate(CH_NAMES)}
N_OUT = 7  # model-output channels (think..idle)
TAGS = ['<|%s|>' % n for n in CH_NAMES[:9]] + ['<|see|>', '<|img|>']
N_VIS = 16
POOLS = {}  # 'math': list of dict(q, gold, rationale?), 'vis': list of objs per image (set by caller)
PROTOCOLS = ['router', 'tags', 'orch', 'turn']
TAG_FORMAT = {'router': False, 'tags': True, 'orch': True, 'turn': True}
QUICK_DEADLINE = 24
CANCEL_GRACE = 10  # an output line already in flight when the cancel arrives is not penalised
SYS_PROMPT = ("System: always-on assistant. Channels: think say write code exec mem idle. "
              "Tools via exec: T<n> calc <expr> | T<n> lookup <word> | T<n> run.\n")
WORDS = ['falcon', 'otter', 'zebra', 'lynx', 'heron', 'bison', 'gecko', 'koala', 'llama', 'moose',
         'newt', 'okapi', 'panda', 'quail', 'raven', 'shark', 'tapir', 'viper', 'walrus', 'yak',
         'badger', 'camel', 'dingo', 'eagle', 'ferret', 'gibbon', 'hyena', 'ibis', 'jackal', 'kiwi',
         'lemur', 'marten', 'narwhal', 'ocelot', 'puffin', 'robin', 'stork', 'toucan', 'urchin', 'vole']
CALC_FORMS = ['{a}*{b}+{c}', '{a}+{b}*{c}', '({a}+{b})*{c}', '{a}*{b}-{c}', '{a}-{b}+{c}']
CODE_KINDS = {
    'sq': ('sum of squares from 1 to {n}', 'print(sum(i*i for i in range(1,{n}+1)))'),
    'cube': ('sum of cubes from 1 to {n}', 'print(sum(i**3 for i in range(1,{n}+1)))'),
    'mult': ('count multiples of {k} below {n}', 'print(len([i for i in range(1,{n}) if i%{k}==0]))'),
}
THINK = {
    'calc': ["T{t} wants {x}. I should not do this in my head, so I will call calc and wait for the result.",
             "T{t} is arithmetic: {x}. Use the calc tool, keep going, and report the number when it returns.",
             "For T{t} the expression is {x}. Calling calc is safer than mental math; answer once it replies."],
    'lookup': ["T{t} asks for the code of {x}. I do not know it, so I will look it up and answer when the tool replies.",
               "T{t} needs the code for {x}. That needs the lookup tool; I will report it when the result comes back."],
    'write': ["T{t} wants the numbers {x} sorted. I can do this myself: smallest first, then write it to the document.",
              "T{t} is a sorting job for {x}. No tool needed; put them in increasing order and write them down."],
    'code': ["T{t} needs the {x}. I will write one line of python, run it, and say the output.",
             "T{t}: compute the {x} with a short python print, execute it, then report the printed value."],
}


# ---------------------------------------------------------------- safe tools
def safe_calc(expr):
    if not re.fullmatch(r'[0-9+\-*() ]{1,40}', expr or '') or '**' in expr:
        return None
    try:
        return str(eval(expr, {'__builtins__': {}}, {}))
    except Exception:
        return None


def _lim_range(*a):
    r = range(*a)
    if len(r) > 100000:
        raise ValueError
    return r


def safe_run(code):
    m = re.fullmatch(r'print\((.*)\)', (code or '').strip())
    if not m or '__' in code or len(code) > 120 or re.search(r'\*\*\s*(?![23]\b)', code):
        return 'error'
    try:
        return str(eval(m.group(1), {'__builtins__': {}}, {'sum': sum, 'range': _lim_range, 'len': len}))
    except Exception:
        return 'error'


# ---------------------------------------------------------------- episodes
CFG = {
    'id': dict(n_tasks=(3, 5), horizon=150, n_quick=(1, 2), n_cancel=(0, 1), n_revise=(0, 1),
               lat=(10, 50), off=(5, 60)),
    'stress': dict(n_tasks=(6, 8), horizon=120, n_quick=(2, 3), n_cancel=(1, 2), n_revise=(1, 2),
                   lat=(10, 60), off=(3, 40)),
    'mm': dict(n_tasks=(2, 4), horizon=150, n_quick=(1, 2), n_cancel=(0, 1), n_revise=(0, 1),
               lat=(10, 50), off=(5, 60), n_math=(0, 1), n_look=(0, 2)),
    'math_async': dict(n_tasks=(2, 2), horizon=120, n_quick=(2, 2), n_cancel=(0, 0), n_revise=(0, 0),
                       lat=(10, 50), off=(5, 60), n_math=(1, 1), n_look=(0, 0), math_first=True, quick_win=(15, 120)),
    'math_solo': dict(n_tasks=(0, 0), horizon=0, n_quick=(0, 0), n_cancel=(0, 0), n_revise=(0, 0),
                      lat=(10, 50), off=(5, 60), n_math=(1, 1), n_look=(0, 0), math_first=True),
    'vis': dict(n_tasks=(2, 2), horizon=150, n_quick=(1, 1), n_cancel=(0, 0), n_revise=(0, 0),
                lat=(10, 50), off=(5, 60), n_math=(0, 0), n_look=(2, 2)),
    'look': dict(n_tasks=(0, 0), horizon=60, n_quick=(0, 0), n_cancel=(0, 0), n_revise=(0, 0),
                 lat=(10, 50), off=(5, 60), n_math=(0, 0), n_look=(4, 4)),
    'long': dict(n_tasks=(28, 32), horizon=3000, n_quick=(5, 7), n_cancel=(3, 5), n_revise=(3, 5),
                 lat=(20, 150), off=(40, 300)),
}


def gen_episode(rng, cfg, math_idx=None, img_pool='vis'):
    c = CFG[cfg] if isinstance(cfg, str) else cfg
    n = rng.randint(*c['n_tasks'])
    kinds = [rng.choice(['calc', 'calc', 'lookup', 'write', 'code']) for _ in range(n)]
    raw = [(rng.randint(0, c['horizon']), k) for k in kinds]
    n_math = rng.randint(*c.get('n_math', (0, 0)))
    math_t = 0 if c.get('math_first') else rng.randint(0, c['horizon'])
    raw += [(math_t, 'math') for _ in range(n_math)]
    raw += [(rng.randint(0, c['horizon']), 'look') for _ in range(rng.randint(*c.get('n_look', (0, 0))))]
    qw = c.get('quick_win')
    raw += [(math_t + rng.randint(*qw) if qw else rng.randint(10, c['horizon'] + 40), 'quick')
            for _ in range(rng.randint(*c['n_quick']))]
    raw.sort(key=lambda x: x[0])
    words = rng.sample(WORDS, len(WORDS))
    kb = {w: ''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(3)) for w in WORDS}
    tasks, events = {}, []
    for i, (tick, kind) in enumerate(raw):
        t = i + 1
        d = dict(tid=t, kind=kind, arrival=tick, lat=[rng.randint(*c['lat']) for _ in range(4)])
        if kind == 'calc':
            a, b, cc = rng.sample(range(2, 100), 3)
            d['expr'] = rng.choice(CALC_FORMS).format(a=a, b=b, c=cc)
            txt = f"T{t} ask: what is {d['expr']}?"
        elif kind == 'lookup':
            d['word'] = words.pop()
            txt = f"T{t} ask: what is the code for {d['word']}?"
        elif kind == 'write':
            d['nums'] = rng.sample(range(1, 100), rng.randint(4, 6))
            txt = f"T{t} write: sort {' '.join(map(str, d['nums']))}"
        elif kind == 'code':
            ck = rng.choice(list(CODE_KINDS))
            d['ck'], d['n'], d['k'] = ck, rng.randint(5, 40), rng.randint(2, 9)
            txt = f"T{t} code: {CODE_KINDS[ck][0].format(n=d['n'], k=d['k'])}"
        elif kind == 'math':
            pool = POOLS['math']
            d['mi'] = math_idx if math_idx is not None else rng.randrange(len(pool))
            d['gold'] = pool[d['mi']]['gold']
            txt = f"T{t} solve: {' '.join(pool[d['mi']]['q'].split())}"
        elif kind == 'look':
            from scenes import question
            d['img'] = rng.randrange(len(POOLS[img_pool]))
            d['qk'], q, d['ans'] = question(rng, POOLS[img_pool][d['img']])
            txt = f"T{t} look: {q}"
        else:
            d['a'], d['b'] = rng.randint(1, 9), rng.randint(1, 9)
            txt = f"T{t} quick: {d['a']}+{d['b']}?"
        tasks[t] = d
        ev = dict(tick=tick, type='task', tid=t, text=txt)
        if kind == 'look':
            ev['img'] = d['img']
        events.append(ev)
    normal = [t for t, d in tasks.items() if d['kind'] not in ('quick', 'math', 'look')]
    cancelled = set(rng.sample(normal, min(len(normal), rng.randint(*c['n_cancel']))))
    for t in cancelled:
        tasks[t]['cancel_at'] = tasks[t]['arrival'] + rng.randint(*c['off'])
        events.append(dict(tick=tasks[t]['cancel_at'], type='cancel', tid=t, text=f"cancel T{t}"))
    calcs = [t for t in normal if tasks[t]['kind'] == 'calc' and t not in cancelled]
    for t in rng.sample(calcs, min(len(calcs), rng.randint(*c['n_revise']))):
        d = tasks[t]
        nums = re.findall(r'\d+', d['expr'])
        a = rng.choice(nums)
        b = str(rng.choice([v for v in range(2, 100) if str(v) not in nums]))
        d['final_expr'] = re.sub(r'\b%s\b' % a, b, d['expr'])
        d['revise_at'] = d['arrival'] + rng.randint(*c['off'])
        events.append(dict(tick=d['revise_at'], type='revise', tid=t, a=a, b=b,
                           text=f"T{t} change {a} to {b}"))
    events.sort(key=lambda e: (e['tick'], e['type'] != 'task'))
    return dict(tasks=tasks, events=events, kb=kb)


def expected(d, kb):
    k = d['kind']
    if k == 'calc':
        return safe_calc(d.get('final_expr', d['expr']))
    if k == 'lookup':
        return kb[d['word']]
    if k == 'write':
        return ' '.join(map(str, sorted(d['nums'])))
    if k == 'code':
        return safe_run(CODE_KINDS[d['ck']][1].format(n=d['n'], k=d['k']))
    if k == 'math':
        return d['gold']
    if k == 'look':
        return d['ans']
    return str(d['a'] + d['b'])


def same(a, b):
    try:
        return abs(float(a.replace(',', '')) - float(b.replace(',', ''))) < 1e-6
    except Exception:
        return a == b


# ---------------------------------------------------------------- environment
class Env:
    """Clock, event delivery rules (the protocol), tool execution and logging."""

    def __init__(self, ep, protocol):
        self.ep, self.p = ep, protocol
        self.clock = 0
        self.uev = list(ep['events'])
        self.ui = 0
        self.tool_q = []
        self.outstanding = 0
        self.code = {}
        self.calls = {}
        self.lines = []
        self.done = False

    def _arrived_tools(self):
        out = [e for e in self.tool_q if e['tick'] <= self.clock]
        self.tool_q = [e for e in self.tool_q if e['tick'] > self.clock]
        self.outstanding -= len(out)
        return out

    def _arrived_users(self, limit=None):
        out = []
        while self.ui < len(self.uev) and self.uev[self.ui]['tick'] <= self.clock and (limit is None or len(out) < limit):
            out.append(self.uev[self.ui]); self.ui += 1
        return out

    def deliver(self, ctx, boundary=False, cur_chan=None):
        p = self.p
        if p in ('router', 'tags') or (p == 'orch' and (ctx == 'idle' or boundary or cur_chan == CH['think'])):
            evs = self._arrived_tools() + self._arrived_users()
        elif p == 'turn' and ctx == 'idle':
            evs = self._arrived_tools()
            if self.outstanding == 0 and not evs:
                evs = self._arrived_users(limit=1)
        else:
            return []
        evs.sort(key=lambda e: e['tick'])
        for e in evs:
            e['delivered'] = self.clock
        return evs

    def _next_wake(self):
        nt = min((e['tick'] for e in self.tool_q), default=None)
        nu = self.uev[self.ui]['tick'] if self.ui < len(self.uev) else None
        if self.p == 'turn' and self.outstanding > 0:
            return nt
        c = [x for x in (nt, nu) if x is not None]
        return min(c) if c else None

    def on_idle(self):
        evs = self.deliver('idle')
        if evs:
            return evs
        t = self._next_wake()
        if t is None:
            self.done = True
            return []
        self.clock = max(self.clock, t)
        return self.deliver('idle')

    def on_line(self, chan, text):
        self.lines.append((chan, text, self.clock))
        if chan == CH['code']:
            m = re.match(r'^T(\d+): (.*)$', text)
            if m:
                self.code[int(m.group(1))] = m.group(2)
        elif chan == CH['exec']:
            self._exec(text)

    def _exec(self, text):
        m = re.match(r'^T(\d+) (calc|lookup|run)\s*(.*)$', text.strip())
        if not m:
            res, t = dict(text='tool error: bad call', tid=None), None
        else:
            t, op, arg = int(m.group(1)), m.group(2), m.group(3).strip()
            if op == 'calc':
                v = safe_calc(arg)
                res = dict(text=f"T{t} calc {arg} = {v if v is not None else 'error'}", expr=arg, value=v)
            elif op == 'lookup':
                v = self.ep['kb'].get(arg, 'unknown')
                res = dict(text=f"T{t} lookup {arg} = {v}", value=v)
            else:
                v = safe_run(self.code.get(t, ''))
                res = dict(text=f"T{t} output: {v}", value=v)
            res['tid'] = t
        d = self.ep['tasks'].get(t)
        k = self.calls.get(t, 0); self.calls[t] = k + 1
        lat = d['lat'][k % len(d['lat'])] if d else 10
        res.update(type='result', tick=self.clock + lat)
        self.tool_q.append(res)
        self.outstanding += 1


# ---------------------------------------------------------------- scoring
def score(ep, lines, n_model_tokens, n_think_tokens, clock):
    out = dict(tasks=[], n_tok=n_model_tokens, n_think=n_think_tokens, ticks=clock)
    for t, d in ep['tasks'].items():
        chan = CH['write'] if d['kind'] == 'write' else CH['say']
        ans = [(txt, tk) for c, txt, tk in lines if c == chan and re.match(r'^T%d:' % t, txt)]
        r = dict(tid=t, kind=d['kind'], cancelled='cancel_at' in d, revised='revise_at' in d)
        if 'cancel_at' in d:
            r['ok'] = not any(tk > d['cancel_at'] + CANCEL_GRACE for _, tk in ans)
        else:
            exp = expected(d, ep['kb'])
            final = ans[-1][0].split(':', 1)[1].strip() if ans else None
            r['ok'] = final is not None and same(final, exp)
            if d['kind'] == 'look':
                r['qk'] = d['qk']
            good = [tk for txt, tk in ans if same(txt.split(':', 1)[1].strip(), exp)]
            r['latency'] = (good[0] - d['arrival']) if good else None
            if d['kind'] == 'quick':
                r['ok'] = r['ok'] and r['latency'] is not None and r['latency'] <= QUICK_DEADLINE
        out['tasks'].append(r)
    out['ep_ok'] = all(r['ok'] for r in out['tasks'])
    return out


# ---------------------------------------------------------------- oracle
class Tok:
    def __init__(self, tokenizer):
        self.tk = tokenizer
        self.cache = {}
        self.tag = {n: tokenizer.convert_tokens_to_ids(TAGS[i]) for i, n in enumerate(CH_NAMES[:9])}
        self.nl = 198
        self.img = tokenizer.convert_tokens_to_ids('<|img|>')

    def line(self, text):
        if text not in self.cache:
            self.cache[text] = self.tk(text, add_special_tokens=False).input_ids + [self.nl]
        return self.cache[text]


PRIO = {'quick': 0, 'cancel': 1, 'revise': 1, 'result': 2, 'task': 3}


class Oracle:
    """Scripted ideal policy; emits one token per tick under the given protocol."""

    def __init__(self, ep, protocol, tok, rng, max_steps=6000):
        self.ep, self.p, self.T, self.rng = ep, protocol, tok, rng
        self.tagfmt = TAG_FORMAT[protocol]
        self.env = Env(ep, protocol)
        self.st = {t: dict(started=False, closed=False, cancelled=False,
                           expr=d.get('expr')) for t, d in ep['tasks'].items()}
        self.items, self.order = [], 0
        self.stream = []  # (tok, chan, is_model)
        self.cur, self.plan, self.cur_item = None, [], None
        self.stack = []  # suspended (item, segment, plan) for urgent interleaving
        self.max_steps = max_steps
        self.n_think = 0
        self.buf = {}

    # -- knowledge
    def know(self, e):
        if e['type'] == 'task':
            it = ('quick' if self.ep['tasks'][e['tid']]['kind'] == 'quick' else 'task', e['tid'])
        elif e['type'] == 'result':
            it = ('result', e['tid'], e)
        elif e['type'] == 'cancel':
            it = ('cancel', e['tid'])
        else:
            it = ('revise', e['tid'], e['a'], e['b'])
        self.items.append((PRIO[it[0]], self.order, it)); self.order += 1

    def insert(self, evs):
        for e in evs:
            ch = 'tool' if e['type'] == 'result' else 'user'
            if self.tagfmt:
                self.stream.append((self.T.tag[ch], CH[ch], False, -1))
            for x in self.T.line(e['text']):
                self.stream.append((x, CH[ch], False, -1))
            if 'img' in e:
                for k in range(N_VIS):
                    self.stream.append((self.T.img, CH['see'], False, e['img'] * N_VIS + k))
            self.know(e)

    def pick(self):
        if not self.items:
            return None
        self.items.sort(key=lambda x: (x[0], x[1]))
        return self.items.pop(0)

    # -- plans
    def think(self, kind, t, x):
        return ('think', self.rng.choice(THINK[kind]).format(t=t, x=x))

    def segments(self, it):
        kind, t = it[0], it[1]
        st, d = self.st[t], self.ep['tasks'][t]
        if kind == 'quick':
            st['closed'] = True
            return [('say', f"T{t}: {d['a'] + d['b']}")]
        if kind == 'task':
            if st['cancelled']:
                return []
            st['started'] = True
            k = d['kind']
            if k == 'math':
                r = POOLS['math'][d['mi']]['rationale']
                r = '\n'.join(l for l in r.strip().split('\n') if not l.strip().startswith('Answer:')).strip()
                return [('think', f"T{t}: " + r), ('say', f"T{t}: {d['gold']}"), ('mem', f"T{t} closed")]
            if k == 'look':
                return [('think', f"T{t}: I look at the picture to check this."), ('say', f"T{t}: {d['ans']}")]
            if k == 'calc':
                return [self.think('calc', t, st['expr']), ('mem', f"T{t} open calc {st['expr']}"),
                        ('exec', f"T{t} calc {st['expr']}")]
            if k == 'lookup':
                return [self.think('lookup', t, d['word']), ('mem', f"T{t} open lookup {d['word']}"),
                        ('exec', f"T{t} lookup {d['word']}")]
            if k == 'write':
                return [self.think('write', t, ' '.join(map(str, d['nums']))),
                        ('write', f"T{t}: {' '.join(map(str, sorted(d['nums'])))}"), ('mem', f"T{t} closed")]
            desc = CODE_KINDS[d['ck']][0].format(n=d['n'], k=d['k'])
            return [self.think('code', t, desc), ('mem', f"T{t} open code"),
                    ('code', f"T{t}: {CODE_KINDS[d['ck']][1].format(n=d['n'], k=d['k'])}"), ('exec', f"T{t} run")]
        if kind == 'cancel':
            if st['closed']:
                return [('think', f"T{t} is already finished, so there is nothing to cancel.")]
            st['cancelled'] = True
            return [('think', f"The user cancelled T{t}. I drop it and will ignore any result for it."),
                    ('mem', f"T{t} cancelled")]
        if kind == 'revise':
            a, b = it[2], it[3]
            new = re.sub(r'\b%s\b' % a, b, st['expr'])
            st['expr'] = new
            if not st['started']:
                return [('think', f"T{t} changed {a} to {b}; I will use {new} when I start it.")]
            st['closed'] = False
            return [('think', f"T{t} changed {a} to {b}, so it is now {new}. I need to recompute."),
                    ('mem', f"T{t} open calc {new}"), ('exec', f"T{t} calc {new}")]
        e = it[2]  # result
        if st['cancelled']:
            return [('think', f"A result for T{t} arrived, but T{t} was cancelled, so I ignore it.")]
        if d['kind'] == 'calc' and e.get('expr') != st['expr']:
            return [('think', f"This result is for the old T{t} expression; I wait for the new one.")]
        st['closed'] = True
        return [('say', f"T{t}: {e['value']}"), ('mem', f"T{t} closed")]

    # -- emission
    def emit(self, tok, chan):
        self.stream.append((tok, CH[chan], True, -1))
        self.env.clock += 1
        if chan == 'think':
            self.n_think += 1

    def run(self):
        env, steps = self.env, 0
        while not env.done and steps < self.max_steps:
            steps += 1
            if self.cur is None:
                if not self.plan and self.stack and not (self.items and min(self.items, key=lambda x: (x[0], x[1]))[0] <= (1 if self.stack[-1][1]["chan"] == "think" else 0)):
                    self.cur_item, self.cur, self.plan = self.stack.pop()
                    self.cur['tagged'] = False
                    if os.environ.get('REORIENT') and self.cur['chan'] == 'think' and self.cur['i'] > 0:
                        self.reorient()
                    if self.st[self.cur_item[2][1]]['cancelled']:
                        self.cur, self.plan = None, []
                    continue
                if not self.plan:
                    it = self.pick()
                    if it is None:
                        self.emit(self.T.tag['idle'], 'idle')
                        self.insert(env.on_idle())
                        continue
                    self.cur_item, self.plan = it, self.segments(it[2])
                    if not self.plan:
                        continue
                ch, text = self.plan.pop(0)
                self.cur = dict(chan=ch, text=text, toks=self.T.line(text), i=0, tagged=False)
            c = self.cur
            boundary = False
            if self.tagfmt and not c['tagged']:
                self.emit(self.T.tag[c['chan']], c['chan']); c['tagged'] = True
            else:
                tok = c['toks'][c['i']]
                self.emit(tok, c['chan']); c['i'] += 1
                ch = CH[c['chan']]
                self.buf.setdefault(ch, []).append(tok)
                if tok == self.T.nl:  # parse exactly as model outputs are parsed: per-channel line buffers
                    line = self.T.tk.decode(self.buf[ch]).split('\n', 1)[0].strip(); self.buf[ch] = []
                    env.on_line(ch, line)
                    mt = re.match(r'T(\d+):', line)
                    if c['chan'] == 'write' and mt and int(mt.group(1)) in self.st:
                        self.st[int(mt.group(1))]['closed'] = True
                if c['i'] == len(c['toks']):
                    self.cur, boundary = None, True
            evs = env.deliver('tok', boundary, CH[c['chan']] if self.cur else None)
            if not evs:
                continue
            self.insert(evs)
            if self.cur is not None:
                quick = any(e['type'] == 'task' and self.ep['tasks'][e['tid']]['kind'] == 'quick' for e in evs)
                urgent = quick or any(e['type'] in ('cancel', 'revise') for e in evs)
                long_thought = self.cur_item[2][0] == 'task' and self.ep['tasks'][self.cur_item[2][1]]['kind'] == 'math'
                if long_thought and urgent and self.p in ('router', 'tags'):
                    self.stack.append((self.cur_item, self.cur, self.plan))  # pause the train of thought, resume later
                    self.cur, self.plan = None, []
                elif self.cur['chan'] == 'think' and (self.p == 'orch' or urgent):
                    self.items.append(self.cur_item)  # abandon the thought, redo this item later
                    self.cur, self.plan = None, []
                elif quick and self.p in ('router', 'tags') and self.cur['chan'] != 'say':
                    self.stack.append((self.cur_item, self.cur, self.plan))  # suspend mid-line, answer, resume
                    self.cur, self.plan = None, []
                else:
                    self.cur['tagged'] = False  # tag formats must reopen the channel
        return self.stream

    def reorient(self):
        """Resume a paused math thought with a recap line, then redo the interrupted line from its start."""
        c, t = self.cur, self.cur_item[2][1]
        d = self.ep['tasks'][t]
        if d['kind'] != 'math':
            return
        q = ' '.join(POOLS['math'][d['mi']]['q'].split())
        ask = re.split(r'(?<=[.?!])\s+', q)[-1]
        toks, i = c['toks'], c['i']
        start = i
        while start > 0 and '\n' not in self.T.tk.decode([toks[start - 1]]):
            start -= 1
        recap = self.T.line(f"Back to T{t}. The question: {ask}")
        c['toks'] = ([self.T.nl] if i > start else []) + recap + toks[start:]
        c['i'] = 0

    def result(self):
        n_model = sum(1 for x in self.stream if x[2])
        return score(self.ep, self.env.lines, n_model, self.n_think, self.env.clock)
