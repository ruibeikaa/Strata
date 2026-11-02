#!/usr/bin/env python3
"""mt.py - micro tests of sm_60 issue behaviour with HFMA2 (chains, stalls, dual-issue of LDS / ALU, LDS.128 broadcast cost).
usage: mt.py <suite>   writes mt_containers.cu, builds (mt_build.bat), prints the run line list for mrun"""
import sys, subprocess, os
import sasslib as sl

K = {}    # name -> (builder, smem_bytes, hfma2_per_iter)

def prologue(e, hd, regs_init):
    e.emit('MOV R4, R%d' % hd['TID.X'], B=(0, 1, 2, 3, 4, 5), S=6)
    for r, v in regs_init:
        e.emit('MOV32I R%d, 0x%x' % (r, v), S=1)
    e.emit('MOV R9, c[0x0][0x148]', S=6)

def epilogue(e, hd, regs):
    # fold the registers so the work is observable
    acc = regs[0]
    for r in regs[1:]:
        e.emit('FADD R%d, R%d, R%d' % (acc, acc, r), S=1)
    e.emit('NOP', S=15)
    e.emit('SHL R4, R4, 0x2', S=6)
    e.emit('IADD R6.CC, R4, c[0x0][0x140]', S=6)
    e.emit('IADD.X R7, RZ, c[0x0][0x144]', S=6)
    e.emit('STG.E [R6], R%d' % acc, S=1)
    e.emit('NOP', S=15)

def loop_end(e, label, pred_early=None):
    e.emit('IADD32I R9, R9, -0x1', S=6)
    e.emit('ISETP.NE.AND P0, PT, R9, RZ, PT', S=15)
    e.emit('@P0 BRA `(%s)' % label, S=13)

def reg_in_bank(bank, n, lo=12):
    out = []
    r = lo
    while len(out) < n:
        if r % 4 == bank: out.append(r)
        r += 1
    return out

# ---- T1: R independent chains, stall S, plain HFMA2 -------------------------------------------------------------------
def t_chain(R, S, nb=256, pairs_alu=0):
    def build(hd, name):
        e = sl.Emit()
        A, B = 10, 11               # banks 2, 3
        banks01 = reg_in_bank(0, 128) + reg_in_bank(1, 128)
        banks01.sort()
        C = banks01[:R]
        prologue(e, hd, [(A, 0x38003800), (B, 0x38003800)] + [(c, 0) for c in C])
        e.label('.L_%s_loop' % name)
        for k in range(nb):
            c = C[k % R]
            e.emit('HFMA2 R%d, R%d, R%d, R%d' % (c, A, B, c), S=S)
        loop_end(e, '.L_%s_loop' % name)
        epilogue(e, hd, C)
        return e
    return build, 4096, nb

for R in (1, 2, 4, 6, 8, 12, 16):
    for S in (1, 2, 4, 6):
        K['c_R%d_S%d' % (R, S)] = t_chain(R, S)

def t_val(S, n=512):
    def build(hd, name):
        e = sl.Emit()
        A, B = 10, 11
        prologue(e, hd, [(A, 0x3c003c00), (B, 0x3c003c00), (12, 0)])
        e.label('.L_%s_loop' % name)
        for k in range(n):
            e.emit('HFMA2 R12, R%d, R%d, R12' % (A, B), S=S)
        loop_end(e, '.L_%s_loop' % name)
        e.emit('NOP', S=15)
        e.emit('SHL R4, R4, 0x2', S=6)
        e.emit('IADD R6.CC, R4, c[0x0][0x140]', S=6)
        e.emit('IADD.X R7, RZ, c[0x0][0x144]', S=6)
        e.emit('STG.E [R6], R12', S=1)
        e.emit('NOP', S=15)
        return e
    return build, 4096, n

for S in (1, 2, 3, 4, 5, 6, 7, 8):
    K['v_S%d' % S] = t_val(S)

# ---- T2: LDS of width W (bits), K of them per 16 HFMA2, paired with HFMA2 or alone; T3: ALU ops --------------------------------------
def t_mix(kind, k, paired, units=16, R=4):
    """kind: 'l128' | 'l64' | 'l32' | 'alu' | 'hmul' ; k ops per unit of 16 HFMA2"""
    def build(hd, name):
        e = sl.Emit()
        A, B = 10, 11
        banks01 = sorted(reg_in_bank(0, 64) + reg_in_bank(1, 64))
        C = banks01[:R]
        D = [80 + 4 * i for i in range(8)]       # LDS destinations (4-aligned)
        prologue(e, hd, [(A, 0x38003800), (B, 0x38003800)] + [(c, 0) for c in C] + [(70, 1), (71, 3)])
        wsh = {'u128': 4, 'u64': 3, 'u32': 2}.get(kind)
        if wsh is not None:
            e.emit('SHL R72, R4, 0x%x' % wsh, S=6)
        if kind.startswith('g'):    # g<pattern><n>_<width>: lanes read <n> distinct addresses; pattern m: lane % n ; b: lane // (32/n)
            pat, n_, wd = kind[1], int(kind[2:].split('w')[0]), int(kind.split('w')[1])
            e.emit('LOP32I.AND R72, R4, 0x1f', S=6)
            if pat == 'm': e.emit('LOP32I.AND R72, R72, 0x%x' % (n_ - 1), S=6)
            else: e.emit('SHR.U32 R72, R72, 0x%x' % (5 - (n_.bit_length() - 1)), S=6)
            e.emit('SHL R72, R72, 0x%x' % {32: 2, 64: 3, 128: 4}[wd], S=6)
        e.emit('MOV R73, RZ', S=6)
        e.emit('MOV R74, 0x40', S=6)
        e.label('.L_%s_loop' % name)
        n = 0
        for u in range(units):
            # place k ops among the 16 HFMA2 at evenly spaced positions
            pos = set((j * 16) // k for j in range(k)) if k else set()
            for j in range(16):
                c = C[j % R]
                h = 'HFMA2 R%d, R%d, R%d, R%d' % (c, A, B, c)
                if j in pos:
                    q = (u * k + len(pos & set(range(j)))) % 8
                    if kind == 'l128': op = 'LDS.U.128 R%d, [RZ+0x%x]' % (D[q], 16 * (q & 7))
                    elif kind == 'l64': op = 'LDS.U.64 R%d, [RZ+0x%x]' % (D[q], 8 * (q & 7))
                    elif kind == 'l32': op = 'LDS.U.32 R%d, [RZ+0x%x]' % (D[q], 4 * (q & 7))
                    elif kind.startswith('g'):
                        wd = int(kind.split('w')[1]); ins_ = {32: 'LDS.U.32', 64: 'LDS.U.64', 128: 'LDS.U.128'}[wd]
                        op = '%s R%d, [R72+0x%x]' % (ins_, D[q], (wd // 8) * 32 * (q & 3))
                    elif kind == 'b128': op = 'LDS.U.128 R%d, [R73+0x%x]' % (D[q], 16 * (q & 7))
                    elif kind == 'b64': op = 'LDS.U.64 R%d, [R73+0x%x]' % (D[q], 8 * (q & 7))
                    elif kind == 'b32': op = 'LDS.U.32 R%d, [R73+0x%x]' % (D[q], 4 * (q & 7))
                    elif kind == 'u128': op = 'LDS.U.128 R%d, [R72+0x%x]' % (D[q], 16 * 32 * (q & 3))
                    elif kind == 'u64': op = 'LDS.U.64 R%d, [R72+0x%x]' % (D[q], 8 * 32 * (q & 3))
                    elif kind == 'u32': op = 'LDS.U.32 R%d, [R72+0x%x]' % (D[q], 4 * 32 * (q & 7))
                    elif kind == 'alu': op = 'SHL R%d, R70, 0x%x' % (D[q], 1 + (q & 3))
                    elif kind == 'cvt': op = 'HADD2.F32 R%d, R%d.H0_H0, -RZ.H0_H0' % (D[q] + 1, D[(q + 3) % 8])
                    elif kind == 'fadd': op = 'FADD.FTZ R%d, R%d, R%d' % (D[q] + 1, D[(q + 3) % 8], D[(q + 5) % 8] + 2)
                    elif kind == 'ffma': op = 'FFMA.FTZ R%d, R%d, R%d, R%d' % (D[q] + 1, D[(q + 3) % 8], D[(q + 5) % 8] + 2, D[(q + 6) % 8] + 3)
                    elif kind == 'mov': op = 'MOV R%d, RZ' % (D[q] + 1)
                    elif kind == 'shr': op = 'SHR.U32 R%d, R70, 0x%x' % (D[q] + 1, 1 + (q & 3))
                    elif kind == 'isetp': op = 'ISETP.LT.AND P%d, PT, R70, R71, PT' % (1 + (q & 3))
                    elif kind == 'alu2': op = 'LOP32I.AND R%d, R70, 0x%x' % (D[q], 0x3fc)
                    elif kind == 'hmul': op = 'HMUL2 R%d, R%d, R%d' % (D[q], A, B)
                    if paired:
                        e.pair(h, dict(S=0), op, dict(S=1))
                    else:
                        e.emit(h, S=1); e.emit(op, S=1)
                else:
                    e.emit(h, S=1)
        loop_end(e, '.L_%s_loop' % name)
        epilogue(e, hd, C)
        return e
    return build, 8192, 16 * units

for kind in ('l128', 'l64', 'l32'):
    for k in (2, 4, 8, 16):
        for p in (1, 0):
            K['m_%s_k%d_%s' % (kind, k, 'p' if p else 's')] = t_mix(kind, k, p)
for kind in ('alu', 'alu2', 'hmul', 'cvt', 'fadd', 'ffma', 'mov', 'shr'):
    for k in (4, 8, 16):
        for p in (0,):
            K['m_%s_k%d_%s' % (kind, k, 'p' if p else 's')] = t_mix(kind, k, p)
K['m_base'] = t_mix('alu', 0, 0)
for pat in ('m', 'b'):
    for n_ in (1, 2, 4, 8, 16, 32):
        for wd in (128, 64):
            for k in (2, 4):
                K['g_g%s%dw%d_k%d' % (pat, n_, wd, k)] = t_mix('g%s%dw%d' % (pat, n_, wd), k, 1)
for kind in ('b128', 'b64', 'b32'):
    for k in (1, 2, 4):
        for p in (1,):
            K['p_%s_k%d_%s' % (kind, k, 'p' if p else 's')] = t_mix(kind, k, p)
for kind in ('u128', 'u64', 'u32'):
    for k in (1, 2, 4, 8):
        for p in (1, 0):
            K['n_%s_k%d_%s' % (kind, k, 'p' if p else 's')] = t_mix(kind, k, p)
for kind in ('l128', 'l64'):
    for k in (1, 3):
        for p in (1, 0):
            K['n_%s_k%d_%s' % (kind, k, 'p' if p else 's')] = t_mix(kind, k, p)

# ---- T4: per unit of 16 HFMA2: NL broadcast LDS.128, NA ALU ops, NH HMUL2; pairing mode: 'sep' (all separate), 'la' (LDS paired with ALU),
#          'lh' (LDS paired with HMUL2), 'lf' (LDS paired with HFMA2) ---------------------------------------------------------------
def t_comb(NL, NA, NH, mode, lw=128, units=16, R=4):
    def build(hd, name):
        e = sl.Emit()
        A, B = 10, 11
        banks01 = sorted(reg_in_bank(0, 64) + reg_in_bank(1, 64))
        C = banks01[:R]
        D = [80 + 4 * i for i in range(8)]
        prologue(e, hd, [(A, 0x38003800), (B, 0x38003800)] + [(c, 0) for c in C] + [(70, 1), (71, 3)])
        e.emit('MOV R73, RZ', S=6)
        e.label('.L_%s_loop' % name)
        ldsop = {128: 'LDS.U.128', 64: 'LDS.U.64', 32: 'LDS.U.32'}[lw]
        for u in range(units):
            # build the unit's instruction list: 16 HFMA2 slots, with the other ops at even positions
            others = []
            for j in range(NL): others.append(('L', j))
            for j in range(NA): others.append(('A', j))
            for j in range(NH): others.append(('H', j))
            # interleave: spread evenly through the unit
            n_o = len(others)
            pos = {}
            for idx, o in enumerate(others):
                pos.setdefault((idx * 16) // max(n_o, 1), []).append(o)
            qd = 0
            alu_q = []
            hmul_q = []
            for j in range(16):
                c = C[j % R]
                h = 'HFMA2 R%d, R%d, R%d, R%d' % (c, A, B, c)
                ops = pos.get(j, [])
                texts = []
                for (kind, idx) in ops:
                    if kind == 'L':
                        q = (u * NL + idx) % 8
                        texts.append(('L', '%s R%d, [R73+0x%x]' % (ldsop, D[q], (lw // 8) * (q & 7))))
                    elif kind == 'A':
                        texts.append(('A', 'SHL R%d, R70, 0x%x' % (D[(idx + 4) % 8] + 1, 1 + (idx & 3))))
                    else:
                        texts.append(('H', 'HMUL2 R%d, R%d, R%d' % (D[(idx + 2) % 8] + 2, A, B)))
                Ls = [t for (k, t) in texts if k == 'L']
                As = [t for (k, t) in texts if k == 'A']
                Hs = [t for (k, t) in texts if k == 'H']
                if mode == 'lf' and Ls:
                    e.pair(h, dict(S=0), Ls.pop(0), dict(S=1))
                else:
                    e.emit(h, S=1)
                if mode == 'la':
                    while Ls and As:
                        e.pair(As.pop(0), dict(S=0), Ls.pop(0), dict(S=1))
                elif mode == 'lh':
                    while Ls and Hs:
                        e.pair(Hs.pop(0), dict(S=0), Ls.pop(0), dict(S=1))
                for t in Ls + As + Hs:
                    e.emit(t, S=1)
        loop_end(e, '.L_%s_loop' % name)
        epilogue(e, hd, C)
        return e
    return build, 8192, 16 * units

for NL, NA, NH in ((2, 0, 0), (2, 8, 0), (2, 8, 4), (4, 8, 4), (2, 6, 2), (1, 6, 2)):
    for mode in ('sep', 'la', 'lh', 'lf'):
        K['q_L%dA%dH%d_%s' % (NL, NA, NH, mode)] = t_comb(NL, NA, NH, mode)

def write_containers(path):
    out = ['// mt_containers.cu - generated by mt.py', '#include <cstdint>']
    for name, (b, smem, nh) in K.items():
        smem = int(os.environ.get('MT_SMEM', smem))
        out.append('extern "C" __global__ void __launch_bounds__(128, 1) %s(float* __restrict__ out, int iters) {' % name)
        out.append('    __shared__ __align__(16) uint32_t pool[%d];' % (smem // 4))
        out.append('    uint32_t tid; asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));')
        out.append('    pool[tid] = tid; __syncthreads();')
        out.append('    out[tid] = (float) pool[(tid * 7) %% %d] + (float) iters;' % (smem // 4))
        out.append('}')
    open(path, 'w').write('\n'.join(out) + '\n')

def main():
    cmd = sys.argv[1]
    if cmd == 'containers':
        write_containers('mt_containers.cu')
        print('containers for', len(K), 'kernels')
    elif cmd == 'gen':
        suite = sys.argv[2] if len(sys.argv) > 2 else ''
        names = [n for n in K if n.startswith(suite)]
        sl.splice('mt_containers.cuasm', 'mt_hand.cuasm', {n: K[n][0] for n in names})
        with open('mt_runlist.txt', 'w') as f:
            for n in names:
                f.write('%s:%d:%d:%d:%d' % (n, int(os.environ.get('CPS', '3')), int(os.environ.get('THR', '128')), int(os.environ.get('ITERS', '2000')), K[n][2]) + chr(10))
        print('runlist:', len(names))

if __name__ == '__main__':
    main()
