"""sched.py - an in-order issue model for hand-written sm_60 SASS: it tracks, per register, when a fixed-latency result is ready and which
scoreboard (write barrier) a variable-latency load still owes, and sets the stall count of the previous instruction / the wait mask of
the consumer so the program is correct for ANY interleaving of other warps (stall counts are minimum delays between this warp's own
instructions; barrier waits only add delay).

Rules it enforces (all learnt on this chip, see sass/gen_hfma.py):
  * fixed-latency results (ALU, HFMA2, HMUL2, FFMA, ...): 6 cycles; the carry flag of IADD.CC: 12 cycles (8 slots was the floor);
  * a predicate used by a memory instruction, branch or any guarded instruction: 16 cycles after its ISETP;
  * a register a memory instruction still has to read (address / store data) may not be rewritten before its read barrier;
  * a register a load still has to write may not be read or rewritten before its write barrier.
Text format of an instruction is CuAssembler's: 'HFMA2 R12, R4.reuse, R5, R12'.
"""
import re

FIX_LAT = 6
PRED_LAT = 16
CC_LAT = 12
VAR_EST = {'LDS': 30, 'LDG': 400, 'S2R': 30, 'LDC': 30}   # estimated cycles for the model's clock after a barrier wait

LOADS = ('LDS', 'LDG', 'LD', 'LDC')
STORES = ('STS', 'STG', 'ST', 'RED', 'ATOM', 'ATOMS')
NOREG = ('BRA', 'SSY', 'SYNC', 'BAR', 'NOP', 'EXIT', 'DEPBAR', 'MEMBAR', 'PBK', 'BRK')


def width_regs(op):
    return 4 if '.128' in op else 2 if '.64' in op else 1


def split_ops(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch == '[': depth += 1
        if ch == ']': depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip()); cur = ''
        else:
            cur += ch
    if cur.strip(): out.append(cur.strip())
    return out


REG = re.compile(r'(?<![A-Za-z0-9_.])R(\d+)\b')


def regs_in(o):
    return [int(x) for x in REG.findall(o)]


def parse(text):
    """-> dict(op, base, pred (guard predicate name or None), dst [regs], src [regs], pdst [pred names], psrc [pred names], carry_w, carry_r)"""
    t = text.strip()
    pred = None
    m = re.match(r'^@(!?)(P\d)\s+(.*)$', t)
    if m:
        pred = m.group(2); t = m.group(3)
    parts = t.split(None, 1)
    op = parts[0]
    base = op.split('.')[0]
    ops = split_ops(parts[1]) if len(parts) > 1 else []
    d = dict(op=op, base=base, pred=pred, dst=[], src=[], pdst=[], psrc=[], carry_w=False, carry_r=False, ops=ops)
    if pred: d['psrc'].append(pred)
    if base in NOREG:
        for o in ops:
            d['src'] += regs_in(o)
        return d
    if base in LOADS:
        w = width_regs(op)
        r0 = regs_in(ops[0])
        if r0:
            d['dst'] = [r0[0] + k for k in range(w)]
        for o in ops[1:]:
            rs = regs_in(o)
            d['src'] += rs
            if rs and o.startswith('[') and ('.E' in op or base == 'LD'): d['src'].append(rs[0] + 1)
        return d
    if base in STORES:
        # [addr], data   (RED / ATOM: [addr], data)
        w = width_regs(op)
        for o in ops:
            rs = regs_in(o)
            if o.startswith('['):
                d['src'] += rs
                if rs and ('.E' in op or base == 'ST'): d['src'].append(rs[0] + 1)
            elif rs:
                d['src'] += [rs[0] + k for k in range(w)] if base in ('STS', 'STG', 'ST') else rs
        return d
    if base in ('ISETP', 'FSETP', 'PSETP', 'HSETP2', 'CSETP'):
        pr = [o for o in ops[:2] if re.match(r'^!?P[T\d]', o)]
        d['pdst'] = [p for p in pr if p != 'PT']
        for o in ops[2:]:
            d['src'] += regs_in(o)
            for pp in re.findall(r'!?(P\d)', o):
                d['psrc'].append(pp)
        return d
    # generic ALU
    if ops and re.match(r'^R\d+', ops[0]) and 'RZ' != ops[0].split('.')[0]:
        d['dst'] = [int(REG.findall(ops[0])[0])]
        rest = ops[1:]
    else:
        rest = ops[1:] if ops and ops[0].startswith('RZ') else ops
    for o in rest:
        d['src'] += regs_in(o)
        for pp in re.findall(r'!?(P\d)\b', o):
            d['psrc'].append(pp)
    mods = op.split('.')[1:]
    if ops and '.CC' in ops[0]:
        d['carry_w'] = True
    if 'X' in mods:
        d['carry_r'] = True
    return d


class Rec:
    __slots__ = ('texts', 'B', 'R', 'W', 'S', 'Y', 'pair', 't')

    def __init__(self):
        self.B = set(); self.R = None; self.W = None; self.S = 1; self.Y = False; self.texts = []; self.pair = False; self.t = 0


class Sched:
    def __init__(self):
        self.recs = []
        self.t = 0                  # issue time (model clock) of the last record
        self.ready = {}             # reg -> time its fixed-latency value is ready
        self.pw = {}                # reg -> write barrier still owed by a variable-latency op
        self.pr = {}                # reg -> read barrier of a memory op that still has to read it
        self.pready = {}            # predicate -> ready time
        self.cc_ready = 0
        self.bdone = {}             # barrier -> model time its last op completes (estimate)
        self.count = 0
        self.maxreg = 9
        self.labels = []
        self.stats = {}

    # ---- state transfer for loops -------------------------------------------------------------------------------------------
    def snapshot(self):
        return dict(pw=dict(self.pw), pr=dict(self.pr), bdone=dict(self.bdone))

    def restore(self, st):
        self.pw = dict(st['pw']); self.pr = dict(st['pr']); self.bdone = dict(st['bdone'])

    def settle(self):
        """everything fixed-latency is ready (back edge / label with padding): reset the clock reference"""
        self.ready = {}; self.pready = {}; self.cc_ready = 0

    # ---- emission ---------------------------------------------------------------------------------------------------------
    def label(self, name):
        self.recs.append(('label', name))

    def _note(self, d, text):
        for r in d['dst'] + d['src']:
            self.maxreg = max(self.maxreg, r)

    def _need(self, d, wait_extra=()):
        """earliest issue time and wait mask for an instruction parsed as d"""
        e = self.t + 1
        B = set(wait_extra)
        for r in d['src']:
            if r in self.ready: e = max(e, self.ready[r])
            if r in self.pw: B.add(self.pw[r])
        for r in d['dst']:
            if r in self.pw: B.add(self.pw[r])          # WAW with a load in flight
            if r in self.pr: B.add(self.pr[r])          # WAR with a memory op that has not read it yet
        for p in d['psrc']:
            if p in self.pready: e = max(e, self.pready[p])
        if d['carry_r']:
            e = max(e, self.cc_ready)
        for b in B:
            if b in self.bdone: e = max(e, self.bdone[b])
        return e, B

    def _commit(self, d, rec_t, B, wb, rb, kind_lat):
        for b in B:
            for r in [r for r, bb in self.pw.items() if bb == b]: del self.pw[r]
            for r in [r for r, bb in self.pr.items() if bb == b]: del self.pr[r]
        base = d['base']
        if base in LOADS:
            assert wb is not None, 'a load needs a write barrier: ' + d['op']
            for r in d['dst']:
                self.pw[r] = wb
                self.ready.pop(r, None)
            self.bdone[wb] = max(self.bdone.get(wb, 0), rec_t + VAR_EST.get(base, 30))
        else:
            for r in d['dst']:
                self.ready[r] = rec_t + (kind_lat or FIX_LAT)
        if d['carry_w']:
            self.cc_ready = rec_t + CC_LAT
        for p in d['pdst']:
            self.pready[p] = rec_t + PRED_LAT
        if rb is not None:
            for r in d['src']:
                self.pr[r] = rb
            self.bdone[rb] = max(self.bdone.get(rb, 0), rec_t + 10)

    def ins(self, text, wb=None, rb=None, lat=None, S=None, B=()):
        d = parse(text)
        self._note(d, text)
        e, Bm = self._need(d, B)
        rec = Rec()
        self._place(rec, e)
        rec.texts = [text]
        rec.B = set(Bm)
        rec.W = wb; rec.R = rb
        if S is not None: rec.S = S
        self._commit(d, rec.t, Bm, wb, rb, lat)
        self.recs.append(rec); self.count += 1
        base = d['base']
        self.stats[base] = self.stats.get(base, 0) + 1
        return rec

    def pair(self, text1, text2, wb=None, rb=None):
        """dual issue { text1 ; text2 } - independent instructions (checked: no register of one is touched by the other)"""
        d1, d2 = parse(text1), parse(text2)
        assert not (set(d1['dst']) & (set(d2['src']) | set(d2['dst']))) and not (set(d2['dst']) & set(d1['src'])), 'dependent pair'
        self._note(d1, text1); self._note(d2, text2)
        e1, B1 = self._need(d1); e2, B2 = self._need(d2)
        rec = Rec()
        self._place(rec, max(e1, e2))
        rec.texts = [text1, text2]; rec.pair = True
        rec.B = set(B1) | set(B2)
        rec.W = wb; rec.R = rb
        self._commit(d1, rec.t, B1, None, None, None)
        self._commit(d2, rec.t, B2, wb, rb, None)
        self.recs.append(rec); self.count += 2
        for d in (d1, d2):
            self.stats[d['base']] = self.stats.get(d['base'], 0) + 1
        return rec

    def _place(self, rec, e):
        """the previous record's stall count is raised so that this record issues at time >= e"""
        prev = None
        for r in reversed(self.recs):
            if isinstance(r, Rec): prev = r; break
            # a label between: no stall adjustment possible across a label (control may join here)
            break
        if prev is None:
            rec.t = max(e, self.t + 1) if self.recs == [] else max(e, self.t + 1)
            self.t = rec.t
            return
        want = e - prev.t
        if want > prev.S:
            while want > 15:
                # insert filler NOPs
                nop = Rec(); nop.texts = ['NOP']; nop.S = 15; nop.t = prev.t + prev.S
                # the previous record keeps its own S; NOP follows
                # (simplest: raise prev.S to 15 and add a NOP chain)
                prev.S = 15
                nop.t = prev.t + 15
                self.recs.append(nop); self.count += 1
                prev = nop
                want = e - prev.t
            prev.S = max(prev.S, want)
        rec.t = prev.t + max(prev.S, 1)
        self.t = rec.t

    def nop_to(self, cycles):
        """pad so that nothing fixed-latency is pending for `cycles` more cycles"""
        r = self.ins('NOP')
        r.S = 15
        return r

    # ---- rendering ------------------------------------------------------------------------------------------------------------
    def render(self):
        out = []
        for r in self.recs:
            if isinstance(r, tuple):
                out.append('  %s:' % r[1]); continue
            if r.pair:
                out.append('      [%s]         /*0000*/         {         %s ;' % (self.ctl(r, first=True), r.texts[0]))
                out.append('      [%s]         /*0000*/                   %s         }' % (self.ctl(r, first=False), r.texts[1]))
            else:
                out.append('      [%s]         /*0000*/                   %s ;' % (self.ctl(r, first=None), r.texts[0]))
        return out

    @staticmethod
    def ctl(r, first):
        if first is True:
            return 'B%s:R-:W-:-:S00' % ''.join(str(k) if k in r.B else '-' for k in range(6))
        if first is False:
            return 'B------:R%s:W%s:-:S%02d' % ('-' if r.R is None else str(r.R), '-' if r.W is None else str(r.W), min(max(r.S, 1), 15))
        S = min(max(r.S, 1), 15)
        B = ''.join(str(k) if k in r.B else '-' for k in range(6))
        return 'B%s:R%s:W%s:%s:S%02d' % (B, '-' if r.R is None else str(r.R), '-' if r.W is None else str(r.W), 'Y' if r.Y else '-', S)
