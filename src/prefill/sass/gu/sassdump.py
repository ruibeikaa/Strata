#!/usr/bin/env python3
"""sassdump.py listing.sass [lo hi]: print instructions with control info (stall yield wr rd wait reuse) from a cuobjdump -sass listing"""
import re, sys
def parse(path):
    L = open(path, encoding='utf-8-sig').read().splitlines()
    ins = []   # (addr, text, hexword)
    words = []
    for l in L:
        m = re.match(r'\s*/\*([0-9a-f]{4})\*/\s+(.*?)\s*;\s*/\*\s*0x([0-9a-f]{16})\s*\*/', l)
        if m:
            ins.append((int(m.group(1), 16), m.group(2).strip(), int(m.group(3), 16)))
        else:
            m2 = re.match(r'\s*/\*\s*0x([0-9a-f]{16})\s*\*/', l)
            if m2: words.append(int(m2.group(1), 16))
    return ins, words
def ctrl(words, addr):
    b = addr // 32
    w = words[b]
    k = (addr % 32) // 8 - 1
    f = (w >> (21 * k)) & 0x1fffff
    return dict(stall=f & 15, yld=(f >> 4) & 1, wr=(f >> 5) & 7, rd=(f >> 8) & 7, wait=(f >> 11) & 63, reuse=(f >> 17) & 15)
if __name__ == '__main__':
    ins, words = parse(sys.argv[1])
    lo = int(sys.argv[2], 16) if len(sys.argv) > 2 else 0
    hi = int(sys.argv[3], 16) if len(sys.argv) > 3 else 1 << 30
    for a, t, h in ins:
        if not (lo <= a <= hi): continue
        c = ctrl(words, a)
        wr = '-' if c['wr'] == 7 else str(c['wr']); rd = '-' if c['rd'] == 7 else str(c['rd'])
        wt = ''.join(str(i) if (c['wait'] >> i) & 1 else '-' for i in range(6))
        print('%04x S%-2d %s W%s R%s B%s r%x  %s' % (a, c['stall'], 'Y' if c['yld'] else '-', wr, rd, wt, c['reuse'], t))
