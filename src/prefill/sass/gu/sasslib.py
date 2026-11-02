"""sasslib.py - emission and splicing helpers for hand-written sm_60 kernels (shared by the micro tests and the expert generators).
A kernel is a container (nvcc -cubin of an empty kernel with the real signature) disassembled by CuAssembler to .cuasm; `splice` replaces
each named kernel's instruction stream (from after the head's last S2R to the EXIT) with the generated text and patches the register count
and the EXIT offset, as sass/gen_hfma.py does."""
import re, sys
import cuasm_edit as ce

class Emit:
    def __init__(self):
        self.out = []
        self.count = 0
        self.maxreg = 9
    @staticmethod
    def ctl(B=(), Rb=None, Wb=None, Y=False, S=1):
        return 'B%s:R%s:W%s:%s:S%02d' % (''.join(str(k) if k in B else '-' for k in range(6)), '-' if Rb is None else str(Rb), '-' if Wb is None else str(Wb), 'Y' if Y else '-', S)
    def note(self, text):
        for m in re.finditer(r'(?<![A-Za-z0-9_.])R(\d+)\b', text):
            self.maxreg = max(self.maxreg, int(m.group(1)))
        # vector destinations / sources occupy consecutive registers
        m = re.search(r'\.(128|64)\s+(?:\[[^\]]*\]\s*,\s*)?R(\d+)', text)
        if m:
            self.maxreg = max(self.maxreg, int(m.group(2)) + (3 if m.group(1) == '128' else 1))
    def emit(self, text, **kw):
        self.out.append('      [%s]         /*0000*/                   %s ;' % (self.ctl(**kw), text)); self.count += 1
        self.note(text)
    def label(self, name):
        self.out.append('  %s:' % name)
    def pair(self, t1, kw1, t2, kw2):
        self.out.append('      [%s]         /*0000*/         {         %s ;' % (self.ctl(**kw1), t1))
        self.out.append('      [%s]         /*0000*/                   %s         }' % (self.ctl(**kw2), t2)); self.count += 2
        self.note(t1); self.note(t2)

def head_info(ins):
    i_s2r = max(k for k, x in enumerate(ins) if x.op.startswith('S2R'))
    info = dict(wait=set(), dests=set())
    for x in ins[:i_s2r + 1]:
        assert not re.match(r'(LDG|LDS|LDC|LD|ST|STG|STS|RED|ATOM)', x.op), 'the container head holds a memory op: %s %s' % (x.op, x.ops)
        if x.op.startswith('S2R'):
            r = int(x.ops[0][1:]); sr = x.ops[1].replace('SR_', '')
            info[sr] = r; info['dests'].add(r)
            if x.W != 'W-': info['wait'].add(int(x.W[1]))
    return i_s2r, info

def splice(inp, outp, builders, regs_override=None):
    """builders: {kernel name: function(head_info) -> Emit}"""
    lines, instrs = ce.load(inp)
    plans = []
    for name, fn in builders.items():
        ins = instrs[name]
        i_s2r, hd = head_info(ins)
        g = fn(hd, name)
        i_exit = next(k for k, x in enumerate(ins) if x.op == 'EXIT')
        plans.append((name, ins, i_s2r, i_exit, g))
    plans.sort(key=lambda p: -p[1][p[2]].idx)
    for name, ins, i_s2r, i_exit, g in plans:
        lines[ins[i_s2r].idx + 1:ins[i_exit].idx] = g.out
    txt = '\n'.join(lines)
    for name, ins, i_s2r, i_exit, g in plans:
        n_exit = (i_s2r + 1) + g.count
        exit_addr = 8 * (n_exit + n_exit // 3 + 1)
        regs = (regs_override or {}).get(name, g.maxreg + 1)
        txt, k1 = re.subn(r'(\.text\.%s,"ax",@progbits.*?\.sectioninfo\s+@"SHI_REGISTERS=)\d+' % re.escape(name), lambda m: m.group(1) + str(regs), txt, count=1, flags=re.S)
        txt, k2 = re.subn(r'(\.word\s+index@\(%s\)\s*\n\s*/\*[0-9a-f]+\*/\s*\.word\s+0x)[0-9a-f]+' % re.escape(name), lambda m: m.group(1) + '%08x' % regs, txt, count=1)
        m = re.search(r'\.section\s+\.nv\.info\.%s,.*?(?=\n// -{10,}|\n\s*\.section\s)' % re.escape(name), txt, flags=re.S)
        assert m, 'nv.info.' + name
        sec = m.group(0)
        sec2, k3 = re.subn(r'(EIATTR_EXIT_INSTR_OFFSETS.*?\.word\s+0x)[0-9a-f]+', lambda mm: mm.group(1) + '%08x' % exit_addr, sec, count=1, flags=re.S)
        txt = txt[:m.start()] + sec2 + txt[m.end():]
        assert k1 == 1 and k2 == 1 and k3 == 1, (name, k1, k2, k3)
        print('%-22s %5d generated instructions, registers up to R%d (SHI_REGISTERS=%d), EXIT at 0x%x' % (name, g.count, g.maxreg, regs, exit_addr))
    open(outp, 'w', encoding='utf-8', newline='\n').write(txt)
