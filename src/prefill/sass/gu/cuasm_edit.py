#!/usr/bin/env python3
"""Small editing layer over CuAssembler's .cuasm text (my own code; it does not import CuAssembler).
A kernel's instruction lines look like
    [B0-2---:R1:W3:Y:S05]         /*0b70*/         {         HFMA2 R12, R4.reuse, R5, R12 ;
    [B------:R-:W-:-:S00]         /*0b78*/                   LDS.U.128 R8, [R2+0x20]         }
This module parses them, lets a script change control fields / opcode text, and writes them back with the same layout."""
import re

LINE = re.compile(r'^(?P<ind>\s*)\[(?P<B>B[-0-5]{6}):(?P<R>R[-0-5]):(?P<W>W[-0-5]):(?P<Y>[-Y]):S(?P<S>\d\d)\](?P<sp1>\s+)/\*(?P<addr>[0-9a-f]+)\*/(?P<sp2>\s+)(?P<body>.*)$')

class Ins:
    __slots__ = ('idx', 'ind', 'B', 'R', 'W', 'Y', 'S', 'sp1', 'addr', 'sp2', 'body', 'pair_first', 'pair_second', 'pred', 'op', 'ops', 'dirty')
    def text(self):
        return '%s[%s:%s:%s:%s:S%02d]%s/*%s*/%s%s' % (self.ind, self.B, self.R, self.W, self.Y, self.S, self.sp1, self.addr, self.sp2, self.body)

def parse_body(ins):
    b = ins.body
    ins.pair_first = b.lstrip().startswith('{')
    ins.pair_second = b.rstrip().endswith('}')
    t = b.strip()
    t = re.sub(r'^\{\s*', '', t)
    t = re.sub(r'\s*\}$', '', t)
    t = t.rstrip(';').strip()
    m = re.match(r'^(@!?U?P\w+)\s+(.*)$', t)
    ins.pred = None
    if m:
        ins.pred, t = m.group(1), m.group(2)
    parts = t.split(None, 1)
    ins.op = parts[0] if parts else ''
    ins.ops = [o.strip() for o in parts[1].split(',')] if len(parts) > 1 else []

def load(path):
    """returns (lines, instrs, sections) ; instrs: dict kernel -> list of Ins (in file order) ; Ins.idx = line index"""
    lines = open(path, encoding='utf-8').read().split('\n')
    instrs = {}
    cur = None
    for i, l in enumerate(lines):
        m = re.match(r'^\s*\.text\.(\w+):\s*$', l)
        if m:
            cur = m.group(1); instrs[cur] = []
            continue
        if cur is None: continue
        if re.match(r'^\s*\.section', l): cur = None; continue
        m = LINE.match(l)
        if not m: continue
        ins = Ins()
        ins.idx = i; ins.dirty = False
        ins.ind, ins.B, ins.R, ins.W, ins.Y, ins.S = m.group('ind'), m.group('B'), m.group('R'), m.group('W'), m.group('Y'), int(m.group('S'))
        ins.sp1, ins.addr, ins.sp2, ins.body = m.group('sp1'), m.group('addr'), m.group('sp2'), m.group('body')
        parse_body(ins)
        instrs[cur].append(ins)
    return lines, instrs

def save(path, lines, instrs):
    for k, lst in instrs.items():
        for ins in lst:
            lines[ins.idx] = ins.text()
    open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))

REG = re.compile(r'(?<![A-Za-z0-9_.])R(\d+)\b')

def regs_of(operand):
    """register numbers named in one operand string (ignores RZ, predicates, c[][], .reuse etc.)"""
    return [int(x) for x in REG.findall(operand)]

def dests_srcs(ins):
    """(dest regs, source regs) for the opcodes this project uses; vector widths via the opcode suffix."""
    op = ins.op.split('.')[0]
    full = ins.op
    ops = ins.ops
    def vec(r, n): return [r + k for k in range(n)]
    if op in ('HFMA2', 'HADD2', 'HMUL2', 'FADD', 'FMUL', 'FFMA', 'MOV', 'IADD', 'IADD32I', 'IADD3', 'SHL', 'SHR', 'LOP', 'LOP32I', 'LOP3', 'XMAD', 'ISCADD', 'LEA', 'BFI', 'SEL', 'PRMT', 'F2F', 'I2F', 'F2I', 'S2R', 'IMNMX', 'SHF', 'FSEL', 'BFE', 'FLO', 'POPC'):
        d = regs_of(ops[0]) if ops and re.match(r'^R\d+', ops[0]) else []
        s = []
        for o in ops[1:]:
            s += regs_of(o)
        return d, s
    if op == 'LDS' or op == 'LDG' or op == 'LD':
        w = 128 if '.128' in full else 64 if '.64' in full else 32
        n = w // 32
        d = vec(regs_of(ops[0])[0], n) if regs_of(ops[0]) else []
        s = []
        for o in ops[1:]: s += regs_of(o)
        return d, s
    if op == 'STS' or op == 'STG' or op == 'ST':
        w = 128 if '.128' in full else 64 if '.64' in full else 32
        n = w // 32
        a = regs_of(ops[0]); v = regs_of(ops[1])
        return [], a + (vec(v[0], n) if v else [])
    return [], [x for o in ops for x in regs_of(o)]
