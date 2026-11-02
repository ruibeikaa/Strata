#!/usr/bin/env python3
"""Generator of the hand-scheduled HFMA2 GEMM family for the Strata prompt read on GP100 (sm_60).

Successor of fp16-work/gemm_sass.py (same inner loop, same register layout, same flush), generalised in the places the
engine needs:
  * per-shape constants (K, N, strides) are immediates: one kernel per (N, K) shape, regenerated, not parameterised;
  * T (the rows) is a runtime value: the last row tile is SHIFTED up so it lies inside the matrix (rows T-128 .. T-1; needs
    T >= 128), computes some rows twice and stores only its own (rows >= the nominal tile start): every load is in bounds,
    nothing is predicated in the loop, and each output is the same bit pattern whichever tile computed it;
  * N need not be a multiple of 128 (only for hc down, N 320): the last column tile is shifted the same way, and a second
    epilogue stores only the columns that belong to it;
  * S = K/32 stages need not be a multiple of 4: a final shorter group is flushed after the loop (the C kernel's
    `if (S % kSteps != 0) flush`);
  * kind 'hc' (gemm_hfma_hc.cuh's gemm_sk_kernel): the result is multiplied by ysc (read from a device word the host
    filled), non-finite outputs are counted (RED.ADD to a counter) and, with split-K, the part's first stage, stage count
    and Y offset come from a table of 8 entries in the kernel parameters, indexed by blockIdx.z.
The arithmetic is the C kernels': the same products in the same order in FP16 pairs, flushed every 4 stages with
HADD2.F32 x2 + FADD, then FADD into the FP32 accumulator.
How it is built: build_hand.bat runs mk_containers.py (an empty kernel per spec with the real parameter list: the signature fixes the
parameter bank layout and the static shared memory), nvcc -cubin, CuAssembler's cuasm.py to disassemble, this script to replace each
kernel's instruction stream (from after the head's last S2R to the EXIT), cuasm.py to assemble, embed_cubin.py to make
gemm_sass_cubin.inc.  Parameters (constant bank 0 from 0x140): dense X, W, Y (8 B each), Tm128 = T - 128; hc X, W, Y, ysc pointer, bad
pointer, Tm128, and for the split kernel a table of 8 x (first stage in bytes, stages, Y offset in bytes) by blockIdx.z.

Rules this code follows because the hardware made them necessary (each cost an hour; the first two showed up only as stores that were
missing in the first wave of a cold launch, never in a repeat):
  * a predicate that guards a memory instruction wants well over 13 cycles after its ISETP (13 lost stores, 16 did not): every predicate
    here is computed >= 30 cycles ahead and not rewritten within a few groups of its last use (the loop's `@P2 LDG` keeps gemm_sass.py's
    ~12 cycles, which has always passed);
  * nothing of the container head may be in flight when the generated code starts (the compiler hoisted a load into the head once and its
    late write landed in a scratch register): the first instruction waits on every scoreboard and the head may not hold a memory op;
  * an IADD.X needs its IADD.CC at least ~6 cycles behind it (the carry flag has the ALU latency): the loop's pointer advances are 8 slots apart
    (a 4-slot spacing, gemm_sass.py's, lost the carry for an operand astride a 4 GiB boundary of the address space: a fault in the engine's
    buffers that no bench allocation ever hit);
  * CuAssembler can only encode instruction shapes it has seen (FSET with a predicate combine did not assemble; `@P IADD R, R, -R` and
    FSETP with a combine predicate did).
usage: gen_hfma.py containers.cuasm out.cuasm [name ...]   (kernels are the ones named in SPECS)"""
import re, struct, sys
import cuasm_edit as ce

ROWB = 72                      # shared-memory row stride (bytes)
TILEB = 128 * ROWB             # 9216
SMEM_B = 2 * TILEB             # B region start

def fbits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]

class Spec:
    def __init__(self, name, kind, N, K, split=False):
        assert kind in ('dense', 'hc')
        self.name, self.kind, self.N, self.K, self.split = name, kind, N, K, split
        self.S = K // 32
        self.KB = 2 * K          # bytes of one row of X and of W (X's stride == K)
        self.NB4 = 4 * N         # bytes of one row of Y (stride == N)
        assert K % 32 == 0 and self.S >= 2 and N % 16 == 0 and N >= 128
        assert not split or kind == 'hc'
        self.gx = (N + 127) // 128
        self.ntail = (N % 128) != 0
        self.dn = (128 - N % 128) if self.ntail else 0     # columns of the shifted last tile that are the previous tile's
        assert self.dn % 16 == 0
        assert 64 * self.KB < (1 << 22) and 128 * self.NB4 < (1 << 31)
        if kind == 'dense':
            self.P = dict(X=0x140, W=0x148, Y=0x150, TM=0x158)
        else:
            self.P = dict(X=0x140, W=0x148, Y=0x150, YSC=0x158, BAD=0x160, TM=0x168, TAB=0x16c)

SPECS = []
def _add(*a, **k):
    SPECS.append(Spec(*a, **k))
# dense: the engine's prompt projections of the Swift model (K = input width: 2560, 6144, 640)
_add('d_n10240_k2560', 'dense', 10240, 2560)     # attn_qkv
_add('d_n6144_k2560', 'dense', 6144, 2560)       # attn_gate
_add('d_n12288_k2560', 'dense', 12288, 2560)     # attn_q
_add('d_n2560_k6144', 'dense', 2560, 6144)       # ssm_out, attn_output
_add('d_n512_k2560', 'dense', 512, 2560)         # attn_k, attn_v
_add('d_n640_k2560', 'dense', 640, 2560)         # ffn_gate_shexp, ffn_up_shexp
_add('d_n2560_k640', 'dense', 2560, 640)         # ffn_down_shexp
# hyper-connection read (BF16 weights as FP16 copies; activation scaled by 2^-k)
_add('h_down_n320_k10240', 'hc', 320, 10240, split=True)
_add('h_up_n10240_k320', 'hc', 10240, 320)


class Gen:
    def __init__(self, spec, head):
        self.sp = spec
        self.head = head                       # {'TID.X': reg, 'CTAID.X': reg, ...}, 'wait': set of barrier ids, 'dests': set of regs
        self.out = []
        self.count = 0
        self.used = set(range(0, 10))
        sp = self.sp
        # ---- register allocation: the order is the validated gemm_sass.py's (the inner loop's banks depend on it) -----------
        self.GP = [self.aligned(4, 4) for _ in range(4)]
        self.FA = [[self.pair_reg(2 if i % 2 == 0 else 0) for i in range(8)] for p in range(2)]
        self.FB = [[self.pair_reg(2 if j % 2 == 0 else 0) for j in range(8)] for p in range(2)]
        self.C = [[self.bank((i & 1) if j % 2 == 0 else 2 + (i & 1)) for j in range(8)] for i in range(8)]
        self.D = [[self.one() for j in range(8)] for i in range(8)]
        self.pA, self.pB, self.YP = self.aligned(2, 2), self.aligned(2, 2), self.aligned(2, 2)
        (self.PA, self.PB, self.QA, self.QB, self.W, self.Dp, self.Dn, self.DQ, self.RS1, self.Rc64) = [self.one() for _ in range(10)]
        (self.TID, self.CXr, self.CYr, self.TXr, self.TYr, self.ROW0, self.QQ, self.X1, self.X2, self.X3, self.WB0) = [self.one() for _ in range(11)]
        # ---- new registers --------------------------------------------------------------------------------------------------
        self.DROW = self.one()                  # d = nominal row start - shifted row start (rows below d belong to the previous tile)
        if sp.kind == 'hc':
            self.CZr = self.one()
        if sp.split:
            self.RLIM = self.one()              # the part's stage count
        # prologue scratch: R4..R9 minus the head's S2R destinations
        # (the head's S2R destinations are copied by the first instructions; the container has no other pending operation)
        self.scr = [4, 5, 6, 7, 8, 9]
        self.maxreg = max(self.used)
        assert self.maxreg <= 254, self.maxreg

    # ---- register helpers -------------------------------------------------------------------------------------------------
    def bank(self, b, lo=10):
        for r in range(lo, 254):
            if r % 4 == b and r not in self.used:
                self.used.add(r); return r
        raise RuntimeError('out of registers (bank %d)' % b)
    def aligned(self, n, align, lo=10):
        base = lo + (-lo) % align
        while base + n <= 254:
            if all((base + k) not in self.used for k in range(n)):
                for k in range(n): self.used.add(base + k)
                return base
            base += align
        raise RuntimeError('out of registers (aligned)')
    def one(self, lo=10):
        for r in range(lo, 254):
            if r not in self.used:
                self.used.add(r); return r
        raise RuntimeError('out of registers')
    def pair_reg(self, mod4, lo=10):
        b = lo + (mod4 - lo) % 4
        while b + 2 <= 254:
            if b not in self.used and (b + 1) not in self.used:
                self.used.add(b); self.used.add(b + 1); return b
            b += 4
        raise RuntimeError('out of registers (pair)')

    # ---- emission ----------------------------------------------------------------------------------------------------------
    @staticmethod
    def ctl(B=(), Rb=None, Wb=None, Y=False, S=1):
        return 'B%s:R%s:W%s:%s:S%02d' % (''.join(str(k) if k in B else '-' for k in range(6)), '-' if Rb is None else str(Rb), '-' if Wb is None else str(Wb), 'Y' if Y else '-', S)
    def emit(self, text, **kw):
        self.out.append('      [%s]         /*0000*/                   %s ;' % (self.ctl(**kw), text)); self.count += 1
    def label(self, name):
        self.out.append('  %s:' % name)
    def emit_pair(self, t1, kw1, t2, kw2):
        self.out.append('      [%s]         /*0000*/         {         %s ;' % (self.ctl(**kw1), t1))
        self.out.append('      [%s]         /*0000*/                   %s         }' % (self.ctl(**kw2), t2)); self.count += 2
    def addr64(self, dst, lo_reg, base_off):
        self.emit('IADD R%d.CC, R%d, c[0x0][0x%x]' % (dst, lo_reg, base_off), S=6)
        self.emit('IADD.X R%d, RZ, c[0x0][0x%x]' % (dst + 1, base_off + 4), S=6)
    def mul_const(self, dst, src, c, tmp):
        """dst = src * c for an immediate c (shifts and ISCADDs: one instruction per set bit); dst != src, tmp unused for now"""
        bits = [b for b in range(31, -1, -1) if (c >> b) & 1]
        assert bits and dst != src
        self.emit('SHL R%d, R%d, 0x%x' % (dst, src, bits[0]), S=6)
        for b in bits[1:]:
            self.emit('ISCADD R%d, R%d, R%d, 0x%x' % (dst, src, dst, b), S=6)

    # ---- the kernel ----------------------------------------------------------------------------------------------------------
    def build(self):
        sp, hd = self.sp, self.head
        emit, GP, FA, FB, C, D = self.emit, self.GP, self.FA, self.FB, self.C, self.D
        PA, PB, QA, QB, W, Dp, Dn, DQ, RS1, Rc64 = self.PA, self.PB, self.QA, self.QB, self.W, self.Dp, self.Dn, self.DQ, self.RS1, self.Rc64
        TID, CXr, CYr, TXr, TYr, ROW0, QQ, X1, X2, X3, WB0 = self.TID, self.CXr, self.CYr, self.TXr, self.TYr, self.ROW0, self.QQ, self.X1, self.X2, self.X3, self.WB0
        pA, pB, YP, DROW = self.pA, self.pB, self.YP, self.DROW
        P = sp.P
        S = sp.S
        KB, NB4 = sp.KB, sp.NB4
        LDG_OFF = 64 * KB
        YROW64 = 16 * NB4
        waitB = (0, 1, 2, 3, 4, 5)              # whatever the container head still has in flight (S2R; a hoisted load wrote R6 once): wait for all of it
        X4, X5, X6, X7 = self.scr[0], self.scr[1], self.scr[2], self.scr[3]

        # ---- prologue ----------------------------------------------------------------------------------------------------
        emit('MOV R%d, R%d' % (TID, hd['TID.X']), B=waitB, S=6)
        emit('MOV R%d, R%d' % (CXr, hd['CTAID.X']), S=6)
        emit('MOV R%d, R%d' % (CYr, hd['CTAID.Y']), S=6)
        if sp.kind == 'hc':
            emit('MOV R%d, R%d' % (self.CZr, hd['CTAID.Z']), S=6)
        emit('LOP32I.AND R%d, R%d, 0xf' % (TXr, TID), S=6)
        emit('SHR.U32 R%d, R%d, 0x4' % (TYr, TID), S=6)
        emit('SHR.U32 R%d, R%d, 0x2' % (ROW0, TID), S=6)
        emit('LOP32I.AND R%d, R%d, 0x3' % (QQ, TID), S=6)
        emit('SHL R%d, R%d, 0x6' % (X1, TYr), S=6)
        emit('ISCADD R%d, R%d, R%d, 0x3' % (PA, TYr, X1), S=6)                 # ty*72
        emit('SHL R%d, R%d, 0x6' % (X1, TXr), S=6)
        emit('ISCADD R%d, R%d, R%d, 0x3' % (PB, TXr, X1), S=6)                 # tx*72
        emit('IADD32I R%d, R%d, 0x%x' % (PB, PB, SMEM_B), S=6)
        emit('IADD32I R%d, R%d, 0x%x' % (QA, PA, TILEB), S=6)
        emit('IADD32I R%d, R%d, 0x%x' % (QB, PB, TILEB), S=6)
        emit('MOV32I R%d, 0x%x' % (Dp, TILEB), S=6)
        emit('MOV32I R%d, 0x%x' % (Dn, (-TILEB) & 0xffffffff), S=6)
        emit('MOV R%d, RZ' % DQ, S=6)
        emit('MOV32I R%d, 0x40' % Rc64, S=6)
        emit('MOV32I R%d, 0x1' % RS1, S=6)
        emit('SHL R%d, R%d, 0x6' % (X1, ROW0), S=6)
        emit('ISCADD R%d, R%d, R%d, 0x3' % (X1, ROW0, X1), S=6)                # row0*72
        emit('ISCADD R%d, R%d, R%d, 0x4' % (WB0, QQ, X1), S=6)                 # + q*16  (write base, buffer 0)
        emit('IADD32I R%d, R%d, 0x%x' % (W, WB0, TILEB), S=6)                  # stage 0 of the loop writes buffer 1
        # split-K: this part's first stage (byte offset), stage count and Y offset from the table, by blockIdx.z
        S0B = YOFF = None
        if sp.split:
            S0B, YOFF = X6, X7
            emit('MOV R%d, c[0x0][0x%x]' % (S0B, P['TAB']), S=6)
            emit('MOV R%d, c[0x0][0x%x]' % (self.RLIM, P['TAB'] + 4), S=6)
            emit('MOV R%d, c[0x0][0x%x]' % (YOFF, P['TAB'] + 8), S=6)
            lp = ['P0', 'P3', 'P4', 'P5', 'P6', 'P1', 'P2']          # seven predicates, all computed before the first use (a predicate wants >= ~16 cycles before a memory-op consumer; 40 here)
            for z in range(1, 8):
                emit('ISETP.EQ.AND %s, PT, R%d, 0x%x, PT' % (lp[z - 1], self.CZr, z), S=6)
            for z in range(1, 8):
                emit('@%s MOV R%d, c[0x0][0x%x]' % (lp[z - 1], S0B, P['TAB'] + 12 * z), S=1)
                emit('@%s MOV R%d, c[0x0][0x%x]' % (lp[z - 1], self.RLIM, P['TAB'] + 12 * z + 4), S=1)
                emit('@%s MOV R%d, c[0x0][0x%x]' % (lp[z - 1], YOFF, P['TAB'] + 12 * z + 8), S=1)
            emit('NOP', S=6)
        # the tile origin, shifted into the matrix: X2 = m0' = min(cy*128, T-128); X1 = n0' = min(cx*128, N-128)
        emit('SHL R%d, R%d, 0x7' % (X1, CYr), S=6)
        emit('IMNMX R%d, R%d, c[0x0][0x%x], PT' % (X2, X1, P['TM']), S=6)
        emit('IADD R%d, R%d, -R%d' % (self.DROW, X1, X2), S=6)
        emit('SHL R%d, R%d, 0x7' % (X1, CXr), S=6)
        if sp.ntail:
            emit('IMNMX R%d, R%d, 0x%x, PT' % (X1, X1, sp.N - 128), S=6)
        # gA = X + (m0' + row0)*KB + q*16 (+ s0); gB = Wt + (n0' + row0)*KB + q*16 (+ s0)
        for (ptr, org, base) in ((pA, X2, P['X']), (pB, X1, P['W'])):
            emit('IADD R%d, R%d, R%d' % (X3, org, ROW0), S=6)
            self.mul_const(X4, X3, KB, X5)
            emit('ISCADD R%d, R%d, R%d, 0x4' % (X4, QQ, X4), S=6)
            if S0B is not None:
                emit('IADD R%d, R%d, R%d' % (X4, X4, S0B), S=6)
            self.addr64(ptr, X4, base)
        # YP = Y + ((m0' + ty)*N + n0' + tx)*4 (+ part offset)
        emit('IADD R%d, R%d, R%d' % (X3, X2, TYr), S=6)
        self.mul_const(X4, X3, NB4, X5)
        emit('IADD R%d, R%d, R%d' % (X3, X1, TXr), S=6)
        emit('ISCADD R%d, R%d, R%d, 0x2' % (X4, X3, X4), S=6)
        if YOFF is not None:
            emit('IADD R%d, R%d, R%d' % (X4, X4, YOFF), S=6)
        self.addr64(YP, X4, P['Y'])
        # zero C and D
        for i in range(8):
            for j in range(8):
                emit('MOV R%d, RZ' % C[i][j], S=1)
                emit('MOV R%d, RZ' % D[i][j], S=1)
        # first tile: global -> registers -> shared buffer 0
        def ldg(k, ptr, off, pred=''):
            emit('%sLDG.E.CI.128 R%d, [R%d%s]' % (pred, GP[k], ptr, '+0x%x' % off if off else ''), Rb=2, Wb=4)
        ldg(0, pA, 0); ldg(1, pA, LDG_OFF); ldg(2, pB, 0); ldg(3, pB, LDG_OFF)
        STS_OFFS = [(0, 0), (0, 8), (1, 4608), (1, 4616), (2, SMEM_B), (2, SMEM_B + 8), (3, SMEM_B + 4608), (3, SMEM_B + 4616)]
        def sts_all(base_reg, pred='', first_wait=True):
            for n, (k, off) in enumerate(STS_OFFS):
                half = 0 if n % 2 == 0 else 2
                emit('%sSTS.64 [R%d+0x%x], R%d' % (pred, base_reg, off, GP[k] + half), B=(4,) if (n == 0 and first_wait) else (), Rb=3)
        sts_all(WB0)
        emit('BAR.SYNC 0x0', B=(3,), S=15)                                      # also waits for the STS reads
        emit('IADD R%d.CC, R%d, R%d' % (pA, pA, Rc64), B=(2,), S=6)
        emit('IADD.X R%d, R%d, RZ' % (pA + 1, pA + 1), S=6)
        emit('IADD R%d.CC, R%d, R%d' % (pB, pB, Rc64), S=6)
        emit('IADD.X R%d, R%d, RZ' % (pB + 1, pB + 1), S=6)
        # step-0 fragments of stage 0 (buffer 0) into set 0
        for i in range(8):
            emit('LDS.U.64 R%d, [R%d+0x%x]' % (FA[0][i], PA, i * 1152), Wb=0)
        for j in range(8):
            emit('LDS.U.64 R%d, [R%d+0x%x]' % (FB[0][j], PB, j * 1152), Wb=0)

        # ---- main loop (gemm_sass.py's, unchanged except the stage-count compare) ----------------------------------------
        self.label('.L_%s_100' % sp.name)
        if sp.split:
            emit('ISETP.LT.AND P2, PT, R%d, R%d, PT' % (RS1, self.RLIM), S=2)
        else:
            emit('ISETP.LT.AND P2, PT, R%d, 0x%x, PT' % (RS1, S), S=2)
        LA = 2
        def cell(h):
            i = h // 8; jj = h % 8
            return i, (jj if i % 2 == 0 else 7 - jj)
        extras = {}
        def add(n, h, text, alu=False, **kw):
            extras.setdefault((n, h), []).append((text, kw, alu))
        for m in range(8):
            n = 2 * m
            pn = (m + 1) & 1
            c0 = 2 * (m + 1) if m < 7 else 0
            baseA, baseB = (PA, PB) if m < 7 else (QA, QB)
            for k in range(16):
                if k < 8:
                    add(n, 2 + 4 * k, 'LDS.U.64 R%d, [R%d+0x%x]' % (FA[pn][k], baseA, k * 1152 + 4 * c0), Wb=pn, Rb=5 if m == 6 else None)
                else:
                    j = k - 8
                    add(n, 2 + 4 * k, 'LDS.U.64 R%d, [R%d+0x%x]' % (FB[pn][j], baseB, j * 1152 + 4 * c0), Wb=pn, Rb=5 if m == 6 else None)
        STSN = 12
        BARN, BARH = 13, 40
        for t, (k, ptr, off) in enumerate(((0, pA, 0), (1, pA, LDG_OFF), (2, pB, 0), (3, pB, LDG_OFF))):
            add(0, 10 + 4 * t, '@P2 LDG.E.CI.128 R%d, [R%d%s]' % (GP[k], ptr, '+0x%x' % off if off else ''), Rb=2, Wb=4, B=(3,) if t == 0 else ())
        # the 64-bit pointer advances: each IADD.X reads the carry flag its IADD.CC wrote, which is not ready 4 cycles later (gemm_sass.py had them 4 slots
        # apart: a carry out of the low word - an operand astride a 4 GiB boundary of the address space - was lost, the pointer jumped back 4 GiB and the
        # load faulted); 8 slots apart now, and the second pair starts after the first pair's IADD.X has read the flag
        add(1, 6, 'IADD R%d.CC, R%d, R%d' % (pA, pA, Rc64), alu=True, B=(2,))
        add(1, 14, 'IADD.X R%d, R%d, RZ' % (pA + 1, pA + 1), alu=True)
        add(1, 18, 'IADD R%d.CC, R%d, R%d' % (pB, pB, Rc64), alu=True)
        add(1, 26, 'IADD.X R%d, R%d, RZ' % (pB + 1, pB + 1), alu=True)
        for n_, (k, off) in enumerate(STS_OFFS):
            half = 0 if n_ % 2 == 0 else 2
            add(STSN, 6 + 3 * n_, '@P2 STS.64 [R%d+0x%x], R%d' % (W, off, GP[k] + half), Rb=3, B=(4,) if n_ == 0 else ())
        add(1, 30, 'IADD R%d, R%d, R%d' % (QA, QA, DQ), alu=True)
        add(1, 34, 'IADD R%d, R%d, R%d' % (QB, QB, DQ), alu=True)
        add(BARN, BARH, 'BAR.SYNC 0x0', S=1)
        UPD_N, UPD_H = 13, 50
        add(UPD_N, UPD_H, 'IADD R%d, R%d, R%d' % (PA, PA, Dp), alu=True, B=(5,))
        add(UPD_N, UPD_H + 4, 'IADD R%d, R%d, R%d' % (PB, PB, Dp), alu=True)
        add(UPD_N, UPD_H + 8, 'IADD R%d, R%d, R%d' % (W, W, Dn), alu=True, B=(3,))
        add(15, 18, 'MOV R%d, R%d' % (DQ, Dn), alu=True)
        add(15, 22, 'LOP32I.XOR R%d, R%d, 0xfffff800' % (Dp, Dp), alu=True)
        add(15, 26, 'LOP32I.XOR R%d, R%d, 0xfffff800' % (Dn, Dn), alu=True)
        add(15, 30, 'LOP.AND.NZ P1, RZ, R%d, 0x3' % RS1, alu=True)
        add(15, 36, 'IADD32I R%d, R%d, 0x1' % (RS1, RS1), alu=True)
        PAIR_RE = r'(@P\d\s+)?(LDS|LDG|STS|BAR)'
        alu_after = {(n, h) for (n, h), lst in extras.items() for (t, kw, alu) in lst if alu}
        A_reg = lambda n, i: FA[(n >> 1) & 1][i] + (n & 1)
        B_reg = lambda n, j: FB[(n >> 1) & 1][j] + (n & 1)
        for n in range(16):
            seq = [cell(h) for h in range(64)]
            ra = [A_reg(n, i) for (i, j) in seq]; rb = [B_reg(n, j) for (i, j) in seq]
            fa = [False] * 64; fb = [False] * 64
            for k in range(64):
                for d in range(1, LA + 1):
                    if k + d < 64:
                        fa[k] = fa[k] or ra[k + d] == ra[k]
                        fb[k] = fb[k] or rb[k + d] == rb[k]
            for k in range(60, 64): fa[k] = fb[k] = False
            for h in range(64):
                if (n, h) in alu_after:
                    for kk in (h - 1, h):
                        if kk >= 0: fa[kk] = fb[kk] = False
            for h in range(64):
                i, j = seq[h]
                text = 'HFMA2 R%d, R%d%s, R%d%s, R%d' % (C[i][j], ra[h], '.reuse' if fa[h] else '', rb[h], '.reuse' if fb[h] else '', C[i][j])
                ex = list(extras.get((n, h), []))
                kw1 = dict(B=(((n >> 1) & 1),) if (h == 0 and n % 2 == 0) else (), S=0 if (ex and re.match(PAIR_RE, ex[0][0])) else 1)
                if ex and re.match(PAIR_RE, ex[0][0]):
                    t, kw, alu = ex.pop(0)
                    self.emit_pair(text, kw1, t, kw)
                else:
                    emit(text, **kw1)
                for (t, kw, alu) in ex:
                    emit(t, **kw)
        # ---- flush every 4th stage ------------------------------------------------------------------------------------
        TT = [GP[k // 4] + k % 4 for k in range(16)]
        def flush_block():
            for g in range(8):
                cells = [(g, j) for j in range(8)]
                for q, (i, j) in enumerate(cells):
                    emit('HADD2.F32 R%d, R%d.H0_H0, -RZ.H0_H0' % (TT[q], C[i][j]), S=1)
                    emit('HADD2.F32 R%d, R%d.H1_H1, -RZ.H0_H0' % (TT[q + 8], C[i][j]), S=1)
                for q, (i, j) in enumerate(cells):
                    emit('FADD R%d, R%d, R%d' % (TT[q], TT[q], TT[q + 8]), S=1)
                for q, (i, j) in enumerate(cells):
                    emit('FADD R%d, R%d, R%d' % (D[i][j], D[i][j], TT[q]), S=1)
                for q, (i, j) in enumerate(cells):
                    emit('MOV R%d, RZ' % C[i][j], S=1)
        emit('@P1 BRA `(.L_%s_101)' % sp.name, S=13)
        flush_block()
        self.label('.L_%s_101' % sp.name)
        emit('@P2 BRA `(.L_%s_100)' % sp.name, S=13)
        if S % 4 != 0:
            flush_block()                       # the last, shorter group (the C kernel's `if (S % kSteps != 0) flush`)

        # ---- epilogue: store D (x ysc, counting non-finite values, for hc) ----------------------------------------------
        # scratch from here: the dead C / FA / FB registers (B0 / B1 wait for the last prefetching LDS).  Rules learnt the hard way:
        # a predicate wants far more than 13 cycles between its ISETP and a memory instruction guarded by it (13 lost stores in the
        # first wave of a cold launch; 16 did not), so every predicate used by a store is computed >= 30 cycles ahead and no
        # predicate is rewritten within a few groups of its last use.
        PTRP = FA[0][0]                         # an even register: a 64-bit address pair
        YSCR, THR = FA[0][1], FA[1][0]
        CNTS = [C[5][k] for k in range(4)]      # partial counts of non-finite outputs (hc)
        RT = [C[7][k] for k in range(7)]        # DROW - 16 i, i = 1..7
        CT = [C[6][k] for k in range(8)]        # FSET results (hc)
        self.PTRP, self.CNTS = PTRP, CNTS
        if sp.kind == 'hc':
            emit('MOV R%d, c[0x0][0x%x]' % (PTRP, P['YSC']), B=(0, 1), S=6)
            emit('MOV R%d, c[0x0][0x%x]' % (PTRP + 1, P['YSC'] + 4), S=6)
            emit('LDG.E R%d, [R%d]' % (YSCR, PTRP), Rb=2, Wb=4)
            emit('MOV32I R%d, 0x%x' % (THR, fbits(3.0e38)), S=6)
            for c_ in CNTS: emit('MOV R%d, RZ' % c_, S=1)
        for i in range(8):
            rp = GP[i // 2] + 2 * (i % 2)
            emit('MOV32I R%d, 0x%x' % (X1, i * YROW64), B=(0, 1) if (i == 0 and sp.kind != 'hc') else (), S=6)
            emit('IADD R%d.CC, R%d, R%d' % (rp, YP, X1), S=6)
            emit('IADD.X R%d, R%d, RZ' % (rp + 1, YP + 1), S=6)
        if sp.kind == 'hc':
            emit('MOV R%d, R%d' % (X1, X1), B=(2, 4), S=6)          # ysc has arrived
        if sp.ntail:
            emit('ISETP.EQ.AND P0, PT, R%d, 0x%x, PT' % (CXr, sp.gx - 1), S=15)
            emit('NOP', S=15)
            emit('@P0 BRA `(.L_%s_tail)' % sp.name, S=13)
        RP = ['P0', 'P3', 'P4', 'P5', 'P6', 'P1', 'P2']     # the row predicates of groups 0..6; group 7 reuses P0 after group 3
        def store_block(jmin):
            js = list(range(jmin, 8))
            for i in range(1, 8):
                emit('IADD32I R%d, R%d, -0x%x' % (RT[i - 1], DROW, 16 * i), S=1)       # TYr + 16 i >= DROW  <=>  TYr >= DROW - 16 i
            emit('ISETP.GE.AND P0, PT, R%d, R%d, PT' % (TYr, DROW), S=6)
            for i in range(1, 7):
                emit('ISETP.GE.AND %s, PT, R%d, R%d, PT' % (RP[i], TYr, RT[i - 1]), S=6)
            for i in range(8):
                rp = GP[i // 2] + 2 * (i % 2)
                pr = RP[i] if i < 7 else 'P0'
                if i == 4:
                    emit('ISETP.GE.AND P0, PT, R%d, R%d, PT' % (TYr, RT[6]), S=6)   # group 7's predicate: group 0 was the last to use P0 (4 groups back), group 7 is 3 groups ahead
                if sp.kind == 'hc':
                    for j in js:
                        emit('FMUL R%d, R%d, R%d' % (D[i][j], D[i][j], YSCR), S=1)
                    emit('NOP', S=6)
                for j in js:
                    emit('@%s STG.E [R%d+0x%x], R%d' % (pr, rp, j * 64, D[i][j]), S=1)
                if sp.kind == 'hc':
                    for k, j in enumerate(js):
                        emit('FSET.GTU.AND R%d, |R%d|, R%d, PT' % (CT[k], D[i][j], THR), S=1)
                    for k, j in enumerate(js):
                        emit('@%s IADD R%d, R%d, -R%d' % (pr, CNTS[k % 4], CNTS[k % 4], CT[k]), S=2)
        def counted_flush():
            """RED.E.ADD of the thread's count of non-finite outputs (only when it is not 0)"""
            emit('IADD R%d, R%d, R%d' % (CNTS[0], CNTS[0], CNTS[1]), S=6)
            emit('IADD R%d, R%d, R%d' % (CNTS[2], CNTS[2], CNTS[3]), S=6)
            emit('IADD R%d, R%d, R%d' % (CNTS[0], CNTS[0], CNTS[2]), S=6)
            emit('ISETP.NE.AND P0, PT, R%d, RZ, PT' % CNTS[0], S=15)
            emit('MOV R%d, c[0x0][0x%x]' % (PTRP, P['BAD']), S=6)
            emit('MOV R%d, c[0x0][0x%x]' % (PTRP + 1, P['BAD'] + 4), S=6)
            emit('NOP', S=15)
            emit('@P0 RED.E.ADD [R%d], R%d' % (PTRP, CNTS[0]), S=1)
        store_block(0)
        if sp.kind == 'hc':
            counted_flush()
        if sp.ntail:
            emit('BRA `(.L_%s_end)' % sp.name, S=13)
            self.label('.L_%s_tail' % sp.name)
            store_block(sp.dn // 16)
            if sp.kind == 'hc':
                counted_flush()
            self.label('.L_%s_end' % sp.name)
        emit('NOP', S=15)


# ---- splice into the container cuasm ---------------------------------------------------------------------------------------
def head_info(lines, ins):
    """the S2R instructions of the container head: which register holds which special register, and which barriers to wait on"""
    i_s2r = max(k for k, x in enumerate(ins) if x.op.startswith('S2R'))
    info = dict(wait=set(), dests=set())
    for x in ins[:i_s2r + 1]:
        assert not re.match(r'(LDG|LDS|LDC|LD|ST|STG|STS|RED|ATOM)', x.op), 'the container head holds a memory op: %s %s' % (x.op, x.ops)
        if x.op.startswith('S2R'):
            r = int(x.ops[0][1:]); sr = x.ops[1].replace('SR_', '')
            info[sr] = r; info['dests'].add(r)
            if x.W != 'W-': info['wait'].add(int(x.W[1]))
    return i_s2r, info


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    want = set(sys.argv[3:])
    lines, instrs = ce.load(inp)
    # splice from the last kernel backwards so earlier line indices stay valid
    plans = []
    for sp in SPECS:
        if want and sp.name not in want:
            continue
        if sp.name not in instrs:
            continue
        ins = instrs[sp.name]
        i_s2r, hd = head_info(lines, ins)
        g = Gen(sp, hd)
        g.build()
        i_exit = next(k for k, x in enumerate(ins) if x.op == 'EXIT')
        plans.append((sp, ins, i_s2r, i_exit, g))
    plans.sort(key=lambda p: -p[1][p[2]].idx)
    for sp, ins, i_s2r, i_exit, g in plans:
        a = ins[i_s2r].idx + 1
        b = ins[i_exit].idx
        lines[a:b] = g.out
    txt = '\n'.join(lines)
    for sp, ins, i_s2r, i_exit, g in plans:
        name = sp.name
        n_exit = (i_s2r + 1) + g.count
        exit_addr = 8 * (n_exit + n_exit // 3 + 1)
        regs = g.maxreg + 1
        txt, k1 = re.subn(r'(\.text\.%s,"ax",@progbits.*?\.sectioninfo\s+@"SHI_REGISTERS=)\d+' % re.escape(name), lambda m: m.group(1) + str(regs), txt, count=1, flags=re.S)
        txt, k2 = re.subn(r'(\.word\s+index@\(%s\)\s*\n\s*/\*[0-9a-f]+\*/\s*\.word\s+0x)[0-9a-f]+' % re.escape(name), lambda m: m.group(1) + '%08x' % regs, txt, count=1)
        m = re.search(r'\.section\s+\.nv\.info\.%s,.*?(?=\n// -{10,}|\n\s*\.section\s)' % re.escape(name), txt, flags=re.S)
        assert m, 'nv.info.' + name
        sec = m.group(0)
        sec2, k3 = re.subn(r'(EIATTR_EXIT_INSTR_OFFSETS.*?\.word\s+0x)[0-9a-f]+', lambda mm: mm.group(1) + '%08x' % exit_addr, sec, count=1, flags=re.S)
        txt = txt[:m.start()] + sec2 + txt[m.end():]
        assert k1 == 1 and k2 == 1 and k3 == 1, (name, k1, k2, k3)
        nh = sum(1 for l in g.out if 'HFMA2' in l)
        print('%-22s %5d generated instructions (%d HFMA2), registers up to R%d (SHI_REGISTERS=%d), EXIT at 0x%x' % (name, g.count, nh, g.maxreg, regs, exit_addr))
    open(outp, 'w', encoding='utf-8', newline='\n').write(txt)


if __name__ == '__main__':
    main()
