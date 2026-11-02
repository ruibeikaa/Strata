#!/usr/bin/env python3
"""gen_gu.py - hand-scheduled SASS twin of moe_ws_hf.cuh's gu_kernel (the FP16 HFMA2 gate/up of a 4-bit expert layer, GP100 / sm_60).
NOT WIRED INTO THE ENGINE: bit for bit the compiler kernel's outputs, but no faster on the engine's real mix of passes (numbers below).

THE KERNEL.  One warp a CTA (grid x = 20 pieces of 64 weight rows, y = chunk), two weight rows a lane (rowA = piece*64 + lane, rowB = rowA + 32),
T tokens a pass (T = 32 or 16), the 4-bit codes decoded in registers through a 256-entry byte table (lut) and the item's scale through two
256-entry tables (hsA / hsB), all three built by a prep kernel per launch (the weights' shift k lives in them; see prep_kernel in gu_bench.cu)
and copied to shared memory.  Arithmetic = the compiler kernel's, product for product: per item 16 half2 products per token in index order,
hacc (half2) chains over the 4 items of a step, widened to FP32 at the step's end (HADD2.F32 x2 + FADD.FTZ), acc = FFMA.FTZ(fd, lo+hi, acc).
Only the order in which DIFFERENT outputs are computed differs.  (HADD2.F32 h.H0_H0, h.H1_H1 in ONE instruction is NOT the same bits as
the two conversions and a FADD: tried, differs.)  Quad-major: for each of the item's 4 weight quads, the groups of 4 tokens; the next
quad's weights are decoded inside group 0 of the current one (two weight buffers of one quad), the activations of the next 4-token block are
loaded (LDS.128, broadcast) one block ahead into two buffers, group-skip tests for tails per block.

WHAT WAS MEASURED (one PH402 die, tree HEAD dd69a7738, the engine's own header compiled as REF; gu_bench.cu; 29.5 M outputs compared bitwise per
run, 40 runs: 4 families x 2 tiers x ragged / full passes, inf / NaN inputs, 1-3 row tails, operands astride a 4 GiB boundary, memcheck 0 errors):
  tier 32, 2304 tokens x 10 over 512 experts (lognormal counts): REF 3078..3140 G MAC/s, hand 2962..2976 (x0.95..0.96)
  tier 32, every pass 32 tokens:                                 REF 3352..3415,        hand 3524..3579 (x1.03..1.06)
  tier 16:                                                       REF 2940..3202,        hand 2450..2800 (x0.84..0.90)
  The hand kernel has 12 % fewer issue slots per HFMA2 than the compiler's (1490 against 1700 an item per 1024 HFMA2), and the same share of its
  slot bound is lost to stalls; what decides it is the non-HFMA2 work - broadcast LDS.128 ~13 %, table decode ~14 %, FP32 flush ~7 %, group tests ~4 % -
  and, for a pass with fewer than 32 tokens, a staging latency (the next item's global loads) a short item cannot cover.  Ablation of the full
  kernel (ragged mix, ms): all 25.45; no decode 22.3; no flush 22.8; no activation LDS 22.75; no staging 25.3; none of the four 16.1 (4685 G MAC/s).
  Rows of the mix in a full 32-token pass: 65 % at 2304 tokens a launch, 80 % at 4096 (lognormal routing, sigma 0.8).

WHY THIS SHAPE (gu/mt.py, mrun.cu): a broadcast LDS.128 costs an issue slot and ~3 cycles of the shared pipe (an LDS.32 broadcast is free up to
4 per 16 HFMA2), and a lane that holds ONE weight row needs 1 LDS.128 per 4 HFMA2 (the compiler kernel's 61-65 % ceiling); two rows a lane halve
it.  Dependent HFMA2 latency is 6 cycles.  ALU ops, LDS.128, HMUL2, HADD2.F32, FADD, FFMA, MOV each cost one full issue slot and do not dual-issue
with HFMA2 (an ALU op paired with HFMA2 is an illegal instruction); LDS.32 / LDS.64 with distinct lane addresses pair for free.  One warp alone is
~7x slower than its issue bound (instruction fetch of an 11 KB loop), 8 warps hide it; registers (241) allow 8 warps an SM at T = 32.

REGISTER BANKS (reg % 4): row A's accumulators (half2 chains) in bank 0, row B's in bank 1, the 16 weight pairs of a row in banks 2/3, the
activation words come from LDS.128 into quads (bank = position in the quad = i % 4).  Each token's two products share the activation operand
(reuse cache); the product whose accumulator sits in the activation's bank goes second, so no instruction reads two operands from one bank.

build: build_gu.bat (needs CuAssembler, see sass/build_hand.bat) -> gu_hand.cubin;  bench: gu_bench.cu (nvcc -arch=sm_60 -use_fast_math).
"""
import os, re, struct, sys
ABL = set(filter(None, os.environ.get('GU_ABL', '').split(',')))
LDSPOS = [int(x) for x in os.environ.get('GU_LDSPOS', '4,8,12,16').split(',')]
import sched as sd
import sasslib as sl

# ---- smem layout (bytes) --------------------------------------------------------------------------------------------------------------
LUT, HSA, HSB, ACT0 = 0x000, 0x400, 0x800, 0x1000
ACT_TOG = 0x800      # T = 32; T = 16 uses 0x400 (see build)
ROWB, SCR, DR = 1280, 80, 20
KSC, KD = 1638400, 1740800
XROWB = 5120
GUR = 1280

# kernel parameter offsets (constant bank 0)
P_CH, P_W, P_XO, P_XH, P_SRC, P_DST, P_TAB, P_BAD = 0x140, 0x148, 0x150, 0x158, 0x160, 0x168, 0x170, 0x178

# barriers
SB_CODE = 3
SB_PRE = 4
SB_RD = 5
SB_LUT = 2


def fbits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


class Alloc:
    def __init__(self, T):
        self.T = T
        self.hA, self.hB, self.pool = [], [], []
        for t in range(T):
            q = 3 + t
            self.hA.append(4 * q); self.hB.append(4 * q + 1); self.pool += [4 * q + 2, 4 * q + 3]
        self.WVA = [[None] * 4 for _ in range(2)]
        self.WVB = [[None] * 4 for _ in range(2)]
        bankA = {0: 2, 1: 3, 2: 3, 3: 2}
        for b_ in range(2):
            for p in range(4):
                self.WVA[b_][p] = self.take(bankA[p])
                self.WVB[b_][p] = self.take(None)
        self.nxt = 4 * (3 + T)
        self.act = [[self.quad() for t in range(4)] for b in range(2)]
        self.pool_free = list(self.pool)
        self.pool = []

    def take(self, bank):
        for r in self.pool:
            if bank is None or r % 4 == bank:
                self.pool.remove(r); return r
        raise RuntimeError('pool empty (bank %s)' % bank)

    def quad(self):
        self.nxt += (-self.nxt) % 4
        r = self.nxt; self.nxt += 4; return r

    def pair(self):
        self.nxt += (-self.nxt) % 2
        r = self.nxt; self.nxt += 2; return r

    def one(self):
        if self.pool_free:
            return self.pool_free.pop(0)
        r = self.nxt; self.nxt += 1; return r


def build(hd, name, T=32, fam3=False):
    NG = T // 4
    NIT = 80
    ACT_TOG = 0x800 if T == 32 else 0x400
    s = sd.Sched()
    A = Alloc(T)
    hA, hB, ACT = A.hA, A.hB, A.act
    WVA, WVB = A.WVA, A.WVB
    accA = [A.one() for _ in range(T)]
    accB = [A.one() for _ in range(T)]
    qA, qB = A.quad(), A.quad()
    NAQ = 4 if T == 32 else 2
    apre = [A.quad() for _ in range(NAQ)]
    ch = apre[0]                                    # the chunk descriptor lives in apre[0] until the first item's activations arrive
    cp, ap, scp, dpp = [A.pair() for _ in range(4)]
    (lane, cx, nreg, slotb, wslot, nt, trow0, p0, rows, row0, cfk, scwA, scwB, dvA, dvB) = [A.one() for _ in range(15)]
    cnt = [A.one() for _ in range(2)]
    hs = [A.one() for _ in range(4)]               # hsA0, hsA1, hsB0, hsB1: alive through the item
    TMP = list(range(0, 12))
    # scalar aliases in R0..R11 (each alive in one phase only)
    lsA, lsB = 8, 9
    fdA, fdB = 8, 9
    sidx, thr = 10, 10
    cy = 11
    tp, off, dstp, slot = 2, 8, 4, 10
    idx = list(range(0, 8))                        # decode / flush temporaries
    NTOK_LDS = 4

    def I(text, **kw):
        return s.ins(text, **kw)

    def add64_cb(dst, offreg, cb):
        I('IADD R%d.CC, R%d, c[0x0][0x%x]' % (dst, offreg, cb))
        I('IADD.X R%d, RZ, c[0x0][0x%x]' % (dst + 1, cb + 4))

    def add64_rr(dst, offreg, base):
        I('IADD R%d.CC, R%d, R%d' % (dst, offreg, base))
        I('IADD.X R%d, R%d, RZ' % (dst + 1, base + 1))

    def inc64(pr, imm, pred='', tmp=11):
        I('MOV32I R%d, 0x%x' % (tmp, imm))
        I('%sIADD R%d.CC, R%d, R%d' % (pred, pr, pr, tmp))
        I('%sIADD.X R%d, R%d, RZ' % (pred, pr + 1, pr + 1))

    def dec64(pr, imm, tmp):
        I('MOV32I R%d, 0x%x' % (tmp, (-imm) & 0xffffffff))
        I('MOV32I R%d, 0xffffffff' % (tmp + 1))
        I('IADD R%d.CC, R%d, R%d' % (pr, pr, tmp))
        I('IADD.X R%d, R%d, R%d' % (pr + 1, pr + 1, tmp + 1))

    # ==============================================================================================================================
    # prologue
    # ==============================================================================================================================
    I('MOV R%d, R%d' % (lane, hd['TID.X']), B=(0, 1, 2, 3, 4, 5))
    I('MOV R%d, R%d' % (cx, hd['CTAID.X']))
    I('MOV R%d, R%d' % (cy, hd['CTAID.Y']))
    for r in hA + hB:
        I('MOV R%d, RZ' % r)
    t = TMP
    I('SHL R%d, R%d, 0x4' % (t[0], cy))
    add64_cb(t[2], t[0], P_CH)
    I('LDG.E.CI.128 R%d, [R%d]' % (ch, t[2]), wb=SB_CODE, rb=SB_RD)
    # tables -> shared memory: 192 uint4, lane + 32 k
    I('MOV R%d, c[0x0][0x%x]' % (t[4], P_TAB))
    I('MOV R%d, c[0x0][0x%x]' % (t[5], P_TAB + 4))
    I('SHL R%d, R%d, 0x4' % (t[6], lane))
    I('IADD R%d.CC, R%d, R%d' % (t[4], t[4], t[6]))
    I('IADD.X R%d, R%d, RZ' % (t[5], t[5]))
    st_regs = [ACT[0][0], ACT[0][1], ACT[0][2], ACT[0][3], ACT[1][0], ACT[1][1]]
    for k in range(6):
        I('LDG.E.CI.128 R%d, [R%d+0x%x]' % (st_regs[k], t[4], 512 * k), wb=SB_PRE, rb=SB_RD)
    for k in range(6):
        I('STS.128 [R%d+0x%x], R%d' % (t[6], 512 * k, st_regs[k]), rb=SB_RD)
    I('MOV R%d, c[0x0][0x%x]' % (t[8], P_TAB))
    I('MOV R%d, c[0x0][0x%x]' % (t[9], P_TAB + 4))
    I('LDG.E.CI R%d, [R%d+0x%x]' % (cfk, t[8], 3072), wb=SB_PRE, rb=SB_RD)
    # slot = w + xoff[j]
    I('SHL R%d, R%d, 0x3' % (t[0], ch), B=(SB_CODE,))
    add64_cb(t[2], t[0], P_XO)
    I('LDG.E.CI.64 R%d, [R%d]' % (off, t[2]), wb=SB_CODE, rb=SB_RD)
    I('IADD R%d, R%d, RZ' % (row0, ch + 1))
    I('IADD R%d, R%d, RZ' % (rows, ch + 2))
    # row offsets of this lane: rowA = cx*64 + lane
    I('ISCADD R%d, R%d, R%d, 0x6' % (t[1], cx, lane))          # rowA
    I('SHL R%d, R%d, 0xa' % (t[3], t[1]))
    I('ISCADD R%d, R%d, R%d, 0x8' % (t[3], t[1], t[3]))        # rowA*1280
    I('SHL R%d, R%d, 0x6' % (t[5], t[1]))
    I('ISCADD R%d, R%d, R%d, 0x4' % (t[5], t[1], t[5]))        # rowA*80
    I('IADD32I R%d, R%d, 0x%x' % (t[5], t[5], KSC))
    I('SHL R%d, R%d, 0x4' % (t[6], t[1]))
    I('ISCADD R%d, R%d, R%d, 0x2' % (t[6], t[1], t[6]))        # rowA*20
    I('IADD32I R%d, R%d, 0x%x' % (t[6], t[6], KD))
    I('IADD R%d.CC, R%d, c[0x0][0x%x]' % (slot, off, P_W), B=(SB_CODE,))
    I('IADD.X R%d, R%d, c[0x0][0x%x]' % (slot + 1, off + 1, P_W + 4))
    add64_rr(cp, t[3], slot)
    add64_rr(scp, t[5], slot)
    add64_rr(dpp, t[6], slot)
    I('MOV R%d, RZ' % p0)
    for c_ in cnt:
        I('MOV R%d, RZ' % c_)
    I('NOP', S=15)

    # ==============================================================================================================================
    # pass loop: setup
    # ==============================================================================================================================
    s.label('.L_%s_pass' % name)
    s.settle()
    t = TMP
    I('IADD R%d, R%d, -R%d' % (nt, rows, p0))
    I('IMNMX R%d, R%d, 0x%x, PT' % (nt, nt, T))
    I('IADD R%d, R%d, R%d' % (trow0, row0, p0))
    I('IADD32I R%d, R%d, -0x1' % (t[0], nt))
    if T == 32:
        I('IMNMX R%d, R%d, R%d, PT' % (t[0], t[0], lane))
    else:
        I('LOP32I.AND R%d, R%d, 0xf' % (t[1], lane))
        I('IMNMX R%d, R%d, R%d, PT' % (t[0], t[0], t[1]))
    I('IADD R%d, R%d, R%d' % (t[0], t[0], trow0))
    I('SHL R%d, R%d, 0x2' % (t[0], t[0]))
    add64_cb(tp, t[0], P_SRC)
    I('LDG.E.CI R%d, [R%d]' % (sidx, tp), wb=SB_PRE, rb=SB_RD)
    I('SHL R%d, R%d, 0xc' % (t[2], sidx), B=(SB_PRE,))
    I('ISCADD R%d, R%d, R%d, 0xa' % (t[2], sidx, t[2]))        # s * 5120
    if T == 16:
        # lane >> 4 selects which 2 of the 4 pieces: +32 bytes
        I('SHR.U32 R%d, R%d, 0x4' % (t[3], lane))
        I('ISCADD R%d, R%d, R%d, 0x5' % (t[2], t[3], t[2]))
    add64_cb(ap, t[2], P_XH)
    # smem addresses: rd slot, wr address (lane * 64 [+ half * 32])
    I('MOV32I R%d, 0x%x' % (slotb, ACT0))
    if T == 32:
        I('SHL R%d, R%d, 0x6' % (t[4], lane))
    else:
        I('LOP32I.AND R%d, R%d, 0xf' % (t[5], lane))
        I('SHL R%d, R%d, 0x6' % (t[4], t[5]))
        I('ISCADD R%d, R%d, R%d, 0x5' % (t[4], t[3], t[4]))      # + half * 32 (t[3] = lane >> 4 from above)
    I('IADD32I R%d, R%d, 0x%x' % (wslot, t[4], ACT0))
    # item 0's activations -> slot 0
    for k in range(NAQ):
        I('LDG.E.CI.128 R%d, [R%d+0x%x]' % (apre[k], ap, 16 * k), wb=SB_PRE, rb=SB_RD)
    # codes, scales, d of step 0
    I('LDG.E.CI.128 R%d, [R%d]' % (qA, cp), wb=SB_CODE, rb=SB_RD)
    I('LDG.E.CI.128 R%d, [R%d+0xa000]' % (qB, cp), wb=SB_CODE, rb=SB_RD)
    I('LDG.E.CI R%d, [R%d]' % (scwA, scp), wb=SB_CODE, rb=SB_RD)
    I('LDG.E.CI R%d, [R%d+0xa00]' % (scwB, scp), wb=SB_CODE, rb=SB_RD)
    I('LDG.E.CI.U16 R%d, [R%d]' % (dvA, dpp), wb=SB_PRE, rb=SB_RD)
    I('LDG.E.CI.U16 R%d, [R%d+0x280]' % (dvB, dpp), wb=SB_PRE, rb=SB_RD)
    for r in accA + accB:
        I('MOV R%d, RZ' % r)

    def stage_acts(first):
        """STS the prefetched activations (FAM3: pairs j and j+4 of each code byte are what the codes pair)"""
        for k in range(NAQ):
            src = apre[k]
            if fam3:
                t_ = TMP[0:4]
                a = [src + j for j in range(4)]
                I('PRMT R%d, R%d, 0x5410, R%d' % (t_[0], a[0], a[2]), B=(SB_PRE,))
                I('PRMT R%d, R%d, 0x7632, R%d' % (t_[1], a[0], a[2]))
                I('PRMT R%d, R%d, 0x5410, R%d' % (t_[2], a[1], a[3]))
                I('PRMT R%d, R%d, 0x7632, R%d' % (t_[3], a[1], a[3]))
                I('STS.128 [R%d+0x%x], R%d' % (wslot, 16 * k, t_[0]), rb=SB_RD)
            else:
                I('STS.128 [R%d+0x%x], R%d' % (wslot, 16 * k, src), rb=SB_RD, B=(SB_PRE,) if k == 0 else ())

    stage_acts(True)
    I('NOP', S=15, B=(SB_PRE,))
    I('IADD32I R%d, R%d, 0x%x' % (wslot, wslot, ACT_TOG))
    I('MOV R%d, RZ' % nreg)
    I('ISETP.LT.AND P3, PT, R%d, 0x%x, PT' % (nreg, NIT - 1))
    I('NOP', S=15)
    I('NOP', S=15)

    # ==============================================================================================================================
    # item loop
    # ==============================================================================================================================
    s.label('.L_%s_item' % name)
    s.settle()
    # ---- item head: the scale tables of this item's rows -----------------------------------------------------------------------
    I('LOP32I.AND R%d, R%d, 0xff' % (lsA, scwA), B=(SB_CODE,))
    I('LOP32I.AND R%d, R%d, 0xff' % (lsB, scwB))
    I('SHR.U32 R%d, R%d, 0x8' % (scwA, scwA))
    I('SHR.U32 R%d, R%d, 0x8' % (scwB, scwB))
    I('SHL R%d, R%d, 0x2' % (idx[0], lsA))
    I('SHL R%d, R%d, 0x2' % (idx[1], lsB))
    I('LDS.U.32 R%d, [R%d+0x%x]' % (hs[0], idx[0], HSA), wb=SB_LUT, rb=SB_RD)
    I('LDS.U.32 R%d, [R%d+0x%x]' % (hs[1], idx[0], HSB), wb=SB_LUT, rb=SB_RD)
    I('LDS.U.32 R%d, [R%d+0x%x]' % (hs[2], idx[1], HSA), wb=SB_LUT, rb=SB_RD)
    I('LDS.U.32 R%d, [R%d+0x%x]' % (hs[3], idx[1], HSB), wb=SB_LUT, rb=SB_RD)

    # ---- the decode of one quad (word m of both rows -> 4 weight pairs a row), in four stages ------------------------------------
    def decode_stages(m, buf):
        jobs = []
        for row in (0, 1):
            for j in range(4):
                jobs.append((row, j))
        tmps = list(range(0, 8))
        st = {'shl': [], 'and': [], 'lds': [], 'mul': []}
        for n_, (row, j) in enumerate(jobs):
            qreg = (qA if row == 0 else qB) + m
            wv = (WVA if row == 0 else WVB)[buf][j]
            h = hs[(0 if row == 0 else 2) + (0 if m < 2 else 1)]
            tmp = tmps[n_]
            st['shl'].append('SHL R%d, R%d, 0x2' % (tmp, qreg) if j == 0 else 'SHR.U32 R%d, R%d, 0x%x' % (tmp, qreg, 8 * j - 2))
            st['and'].append('LOP32I.AND R%d, R%d, 0x3fc' % (tmp, tmp))
            st['lds'].append('LDS.U.32 R%d, [R%d+0x%x]' % (wv, tmp, LUT))
            st['mul'].append('HMUL2 R%d, R%d, R%d' % (wv, wv, h))
        return st

    def decode_now(m, buf):
        if 'nodec' in ABL: return
        st = decode_stages(m, buf)
        for t_ in st['shl']: I(t_)
        for t_ in st['and']: I(t_)
        for t_ in st['lds']: I(t_, wb=SB_LUT, rb=SB_RD)
        for t_ in st['mul']: I(t_)

    decode_now(0, 0)
    # ---- prefetch of the next item's activations (the codes follow inside group 0 of quad 2) -----------------------------------------
    for k in range(0 if 'nostage' in ABL else NAQ):
        I('@P3 LDG.E.CI.128 R%d, [R%d+0x%x]' % (apre[k], ap, 64 + 16 * k), wb=SB_PRE, rb=SB_RD)

    # ---- the products: quad-major, groups of 4 tokens inside ------------------------------------------------------------------------
    def lds_block(buf, g, q, rb=None):
        for tk in range(4):
            I('LDS.U.128 R%d, [R%d+0x%x]' % (ACT[buf][tk], slotb, (4 * g + tk) * 64 + q * 16), wb=buf, rb=rb)

    lds_block(0, 0, 0)
    inserts = {}

    def add_insert(g, q, pos, fn):
        inserts.setdefault((g, q, pos), []).append(fn)

    # pointer maintenance inside group 0 of quad 3: two carry chains, 12+ cycles apart
    add_insert(0, 3, 3, lambda: I('IADD R%d.CC, R%d, 0x10' % (cp, cp)))
    add_insert(0, 3, 17, lambda: I('IADD.X R%d, R%d, RZ' % (cp + 1, cp + 1)))
    add_insert(0, 3, 19, lambda: I('IADD R%d.CC, R%d, 0x40' % (ap, ap)))
    add_insert(0, 3, 31, lambda: I('IADD.X R%d, R%d, RZ' % (ap + 1, ap + 1)))
    # the next quad's decode inside group 0 of quad q: shifts, masks, table loads, then the scale products at the end of the block
    for q in range(3):
        st = decode_stages(q + 1, (q + 1) % 2)
        if 'nodec' not in ABL:
            for n_ in range(8):
                add_insert(0, q, 1 + n_, (lambda t_=st['shl'][n_]: I(t_)))
                add_insert(0, q, 9 + n_, (lambda t_=st['and'][n_]: I(t_)))
                add_insert(0, q, 17 + n_, (lambda t_=st['lds'][n_]: I(t_, wb=SB_LUT, rb=SB_RD)))
                add_insert(0, q, 32 + 0, (lambda t_=st['mul'][n_]: I(t_)))
    # the next item's codes: after quad 3's words were read (the shifts of group 0 / quad 2)
    if 'nostage' not in ABL:
        add_insert(0, 2, 20, lambda: I('LDG.E.CI.128 R%d, [R%d+0x10]' % (qA, cp), wb=SB_CODE, rb=SB_RD))
        add_insert(0, 2, 21, lambda: I('LDG.E.CI.128 R%d, [R%d+0xa010]' % (qB, cp), wb=SB_CODE, rb=SB_RD))

    for q in range(int(os.environ.get('GU_NQ', '4'))):
        wb_ = q % 2
        for g in range(NG):
            if g > 0 and 'nobra' not in ABL:
                I('@P0 BRA `(.L_%s_qend%d)' % (name, q), S=13)
            buf = g % 2
            nxt_ = (g + 1, q) if g + 1 < NG else None
            pos = 0
            lds_done = 0
            for p in range(4):
                order = ('B', 'A') if p == 0 else ('A', 'B')
                for tk in range(4):
                    act = ACT[buf][tk] + p
                    for kk, row in enumerate(order):
                        cell = hA[4 * g + tk] if row == 'A' else hB[4 * g + tk]
                        wv = (WVA if row == 'A' else WVB)[wb_][p]
                        I('HFMA2 R%d, R%d%s, R%d, R%d' % (cell, act, '.reuse' if kk == 0 else '', wv, cell))
                        pos += 1
                        if pos == 3 and g + 1 < NG and 'nobra' not in ABL:
                            I('ISETP.LE.AND P0, PT, R%d, 0x%x, PT' % (nt, 4 * (g + 1)))
                        for fn in inserts.get((g, q, pos), []):
                            fn()
                        if nxt_ is not None and lds_done < 4 and pos == LDSPOS[lds_done]:
                            if 'nolds' not in ABL:
                                I('LDS.U.128 R%d, [R%d+0x%x]' % (ACT[1 - buf][lds_done], slotb, (4 * nxt_[0] + lds_done) * 64 + q * 16), wb=1 - buf)
                            lds_done += 1
        s.label('.L_%s_qend%d' % (name, q))
        s.settle()
        I('NOP', B=(0, 1))
        if q < 3:
            lds_block(0, 0, q + 1)
    # ---- item tail ---------------------------------------------------------------------------------------------------------------
    s.label('.L_%s_tail' % name)
    s.settle()
    I('NOP', B=(0, 1))
    I('IADD32I R%d, R%d, 0x1' % (nreg, nreg))
    I('ISETP.LT.AND P2, PT, R%d, 0x%x, PT' % (nreg, NIT))
    I('ISETP.LT.AND P3, PT, R%d, 0x%x, PT' % (nreg, NIT - 1))
    I('LOP32I.AND R%d, R%d, 0x3' % (idx[0], nreg))
    I('ISETP.NE.AND P1, PT, R%d, RZ, PT' % idx[0])
    I('LOP32I.AND R%d, R%d, 0x7' % (idx[1], nreg))
    I('ISETP.EQ.AND P5, PT, R%d, RZ, PT' % idx[1])
    stage_acts(False)
    I('LOP32I.XOR R%d, R%d, 0x%x' % (slotb, slotb, ACT_TOG))
    I('LOP32I.XOR R%d, R%d, 0x%x' % (wslot, wslot, ACT_TOG), B=(SB_RD,))
    # ---- flush every 4th item ----------------------------------------------------------------------------------------------------
    I('@P1 BRA `(.L_%s_noflush)' % name, S=13)
    # fd = f32(d) * cfk
    I('HADD2.F32 R%d, R%d.H0_H0, -RZ.H0_H0' % (fdA, dvA), B=(SB_CODE, SB_PRE))
    I('HADD2.F32 R%d, R%d.H0_H0, -RZ.H0_H0' % (fdB, dvB))
    I('FMUL.FTZ R%d, R%d, R%d' % (fdA, fdA, cfk))
    I('FMUL.FTZ R%d, R%d, R%d' % (fdB, fdB, cfk))
    # next step's scale words and d (the pointers move at once; the loads are consumed one item later)
    inc64(scp, 4)
    inc64(dpp, 2, '@P5 ')
    I('LDG.E.CI R%d, [R%d]' % (scwA, scp), wb=SB_CODE, rb=SB_RD)
    I('LDG.E.CI R%d, [R%d+0xa00]' % (scwB, scp), wb=SB_CODE, rb=SB_RD)
    I('LDG.E.CI.U16 R%d, [R%d]' % (dvA, dpp), wb=SB_PRE, rb=SB_RD)
    I('LDG.E.CI.U16 R%d, [R%d+0x280]' % (dvB, dpp), wb=SB_PRE, rb=SB_RD)
    FL1 = 'fl3' not in ABL      # one-instruction widening: HADD2.F32 sum, h.H0_H0 + h.H1_H1
    for g in range(0 if 'noflush' in ABL else NG):
        if g > 0:
            I('ISETP.LE.AND P0, PT, R%d, 0x%x, PT' % (nt, 4 * g))
            I('@P0 BRA `(.L_%s_fl_end)' % name, S=13)
        cells = [(hA[4 * g + tk], accA[4 * g + tk], fdA) for tk in range(4)] + [(hB[4 * g + tk], accB[4 * g + tk], fdB) for tk in range(4)]
        n_c = len(cells)
        for i in range(n_c + 6):
            if i < n_c:
                h, ac, fd = cells[i]
                I('HADD2.F32 R%d, R%d.H0_H0, -RZ.H0_H0' % (i % 8, h))
                I('HADD2.F32 R%d, R%d.H1_H1, -RZ.H0_H0' % (h, h))
            if 3 <= i < n_c + 3:
                h, ac, fd = cells[i - 3]
                I('FADD.FTZ R%d, R%d, R%d' % ((i - 3) % 8, (i - 3) % 8, h))
            if 6 <= i:
                h, ac, fd = cells[i - 6]
                I('FFMA.FTZ R%d, R%d, R%d, R%d' % (ac, fd, (i - 6) % 8, ac))
                I('MOV R%d, RZ' % h)
    s.label('.L_%s_fl_end' % name)
    s.label('.L_%s_noflush' % name)
    s.settle()
    I('@P2 BRA `(.L_%s_item)' % name, S=13)

    # ==============================================================================================================================
    # pass end: pointers back, stores
    # ==============================================================================================================================
    t = TMP
    I('NOP', B=(0, 1, SB_CODE, SB_PRE), S=6)
    dec64(cp, ROWB, t[0])
    dec64(scp, SCR, t[0])
    dec64(dpp, DR, t[0])
    I('SHL R%d, R%d, 0xa' % (t[1], trow0))
    I('ISCADD R%d, R%d, R%d, 0x8' % (t[1], trow0, t[1]))        # trow0 * 1280
    I('ISCADD R%d, R%d, R%d, 0x6' % (t[2], cx, lane))           # cx*64 + lane
    I('IADD R%d, R%d, R%d' % (t[1], t[1], t[2]))
    I('SHL R%d, R%d, 0x2' % (t[1], t[1]))
    add64_cb(dstp, t[1], P_DST)
    I('MOV32I R%d, 0x%x' % (thr, fbits(3.0e38)))
    for g in range(NG):
        if g > 0:
            I('ISETP.LE.AND P0, PT, R%d, 0x%x, PT' % (nt, 4 * g))
            I('@P0 BRA `(.L_%s_st_end)' % name, S=13)
        for tk in range(1, 4):
            I('ISETP.GT.AND P%d, PT, R%d, 0x%x, PT' % (3 + tk, nt, 4 * g + tk))
        for tk in range(4):
            tok = 4 * g + tk
            pr = '' if tk == 0 else '@P%d ' % (3 + tk)
            for (arow, o) in ((accA, 0), (accB, 128)):
                I('%sSTG.E [R%d+0x%x], R%d' % (pr, dstp, tok * XROWB + o, arow[tok]), rb=SB_RD)
                ct = (0, 1, 2, 3, 6, 7, 8, 9)[(tk * 2 + (o >> 7)) % 8]
                I('FSET.GTU.AND R%d, |R%d|, R%d, PT' % (ct, arow[tok], thr))
                I('%sIADD R%d, R%d, -R%d' % (pr, cnt[(tk * 2 + (o >> 7)) % 2], cnt[(tk * 2 + (o >> 7)) % 2], ct))
    s.label('.L_%s_st_end' % name)
    s.settle()
    I('IADD32I R%d, R%d, 0x%x' % (p0, p0, T))
    I('ISETP.LT.AND P2, PT, R%d, R%d, PT' % (p0, rows))
    I('NOP', S=15)
    I('@P2 BRA `(.L_%s_pass)' % name, S=13)
    # the non-finite count
    I('IADD R%d, R%d, R%d' % (cnt[0], cnt[0], cnt[1]))
    I('ISETP.NE.AND P0, PT, R%d, RZ, PT' % cnt[0])
    I('MOV R%d, c[0x0][0x%x]' % (t[0], P_BAD))
    I('MOV R%d, c[0x0][0x%x]' % (t[1], P_BAD + 4))
    I('NOP', S=15)
    I('@P0 RED.E.ADD [R%d], R%d' % (t[0], cnt[0]), rb=SB_RD)
    I('NOP', S=15)
    s.out = s.render()
    return s


KERNELS = [('gu_t32', 32, False), ('gu_t32p', 32, True), ('gu_t16', 16, False), ('gu_t16p', 16, True)]


def write_containers(path):
    out = ['// containers_gu.cu - generated by gen_gu.py: the parameter layout of the hand-scheduled gate/up kernels (bodies replaced)',
           '#include <cstdint>']
    for name, T, f3 in KERNELS:
        out.append('extern "C" __global__ void __launch_bounds__(32, 8) %s(const int4* __restrict__ chunks, const char* __restrict__ w, const int64_t* __restrict__ xoff, '
                   'const uint4* __restrict__ xh, const int32_t* __restrict__ src, float* __restrict__ dst, const uint32_t* __restrict__ tab, unsigned* __restrict__ bad) {' % name)
        out.append('    __shared__ __align__(16) uint32_t pool[%d];' % (2048 if T == 32 else 1536))
        out.append('    uint32_t tid, cx, cy;')
        out.append('    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));')
        out.append('    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(cx));')
        out.append('    asm volatile("mov.u32 %0, %%ctaid.y;" : "=r"(cy));')
        out.append('    pool[tid] = cx + cy; __syncthreads();')
        out.append('    dst[tid] = (float) pool[(tid * 7) %% %d]' % (2048 if T == 32 else 1536) + ' + (float) ((uintptr_t) chunks + (uintptr_t) w + (uintptr_t) xoff + (uintptr_t) xh + (uintptr_t) src + (uintptr_t) tab); atomicAdd(bad, 1u);')
        out.append('}')
    open(path, 'w').write('\n'.join(out) + '\n')


def main():
    cmd = sys.argv[1]
    if cmd == 'containers':
        write_containers('containers_gu.cu')
        print('containers_gu.cu:', len(KERNELS), 'kernels')
    elif cmd == 'gen':
        want = set(sys.argv[4:])
        builders = {}
        for name, T, f3 in KERNELS:
            if want and name not in want: continue
            builders[name] = (lambda hd, nm, T=T, f3=f3: build(hd, nm, T, f3))
        sl.splice(sys.argv[2], sys.argv[3], builders)


if __name__ == '__main__':
    main()
