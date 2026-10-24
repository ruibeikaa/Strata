// src/kernels/cuda/q8_sm60_f.cuh - STRATA_Q8F=1 (opt-in, Pascal GP100 decode, campaign MROW): the Q8_0 GEMV of
// q8sm60 / q8sm60p for 2-8 verify rows with the int8 dots done as exact fp32 FMAs, so an extra row costs about one FFMA
// per weight instead of one VMAD (GP100 has no dp4a; STRATA_DP4A is four VMAD.S8, a half-rate instruction).
//
// What stays the same as q8sm60::kernel / q8sm60p::kernel, per output row and column (bit-exactness):
//   * the integer dot s of every 16-byte chunk (16 int8 weights x 16 int8 activations).  Here it is summed in fp32:
//     every product and every partial sum is an integer of magnitude <= 16 * 128 * 127 = 260096 < 2^24, so each FMA is
//     exact and the result equals float(s) exactly (and is never -0: the sum starts at +0);
//   * the fp32 chain of a row: lane L (of TPR = 32) adds chunks c = L, L + 32, ... in that order with the same
//     expression acc += dw * sd * float(s), then the same xor tree over the 32 lanes;
//   * which values go in: the packed planes (int8 row-major, fp16 scale per 32) and the q8_1 blocks (int8 + the fp32
//     of their fp16 d), as q8sm60 stages them.
// What changes: WHO computes what.  A workgroup holds the window's activations in registers as floats, one 16-value
// chunk per thread (thread c = 32 i + L owns chunk c, so the NIN / 16 threads cover a row); for each row it converts the
// row's chunk to floats once (byte ^ 0x80 under 0x4B000000 by PRMT, minus 2^23 + 128: exact) and runs the NCOLS dots.
// The per-chunk sums go to shared memory; one warp per (row, column) then runs that pair's chain exactly as q8sm60's
// lane L would.  Rows are taken RB at a time (one barrier per RB rows, two buffers).
#pragma once

namespace q8f {

// a byte b of v as the float of the int8 it holds: 2^23 + (b ^ 0x80) - (2^23 + 128), all exact
__device__ __forceinline__ void w16(const int4 v, float (&w)[16]) {
    const uint32_t a[4] = {uint32_t(v.x) ^ 0x80808080u, uint32_t(v.y) ^ 0x80808080u, uint32_t(v.z) ^ 0x80808080u,
                           uint32_t(v.w) ^ 0x80808080u};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
#pragma unroll
        for (int b = 0; b < 4; ++b)
            w[4 * q + b] = __int_as_float(int(__byte_perm(a[q], 0x4B000000u, 0x7540u | unsigned(b)))) - 8388736.0f;
    }
}

template<int NIN> constexpr int threads() { return NIN / 16; }

// RB rows a pass; PF: the next pass's weights are loaded before this pass's chains run
template<int NCOLS, int NIN, int RB, bool PF>
__launch_bounds__(NIN / 16)
__global__ void kernel(const q8sm60p::Segs S, const void* __restrict__ xin) {
    constexpr int NCH = NIN / 16;      // 16-byte chunks a row = threads a block
    constexpr int CPL = NCH / 32;      // chunks a lane of the chain (q8sm60's TPR = 32)
    constexpr int NB = NIN / 32;
    static_assert(NCH % 32 == 0, "whole warps");
    constexpr int NBUF = 2 * RB * NCOLS * NCH * 8 <= 46 * 1024 ? 2 : 1;   // one buffer (and a second barrier) when two do not fit
    __shared__ float2 sb[NBUF][RB][NCOLS][NCH];   // (dw * sd, float(s)) of every chunk
    const Q81Block* __restrict__ x = static_cast<const Q81Block*>(xin);
    const int c = int(threadIdx.x), lane = c & 31, warp = c >> 5;

    float xf[NCOLS][16];
    float xs[NCOLS];   // this thread's chunk's activation scale per column
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        const Q81Block* xb = x + std::size_t(j) * NB + (c >> 1);
        const int* src = reinterpret_cast<const int*>(xb->qs) + (c & 1) * 4;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int v = src[q];
#pragma unroll
            for (int b = 0; b < 4; ++b) xf[j][4 * q + b] = float(int(int8_t(v >> (8 * b))));
        }
        xs[j] = __low2float(xb->ds);
    }

    // the segment of group g (as q8sm60p::kernel's locate, RB rows a group)
    auto locate = [&](int g, const int8_t*& qs, const half*& dp, float*& y, int& n_out, int& r0) {
        int gs = 0;
        q8sm60p::Seg sg = S.s[0];
        if (g >= S.s[0].g_end) {
            gs = S.s[0].g_end;
            sg = S.s[1];
            if (g >= S.s[1].g_end) {
                gs = S.s[1].g_end;
                sg = S.s[2];
            }
        }
        qs = sg.qs;
        dp = sg.d;
        y = sg.y;
        n_out = sg.n_out;
        r0 = (g - gs) * RB;
    };
    auto load = [&](int g, int4 (&wv)[RB], half (&dh)[RB]) {
        if (g >= S.groups) return;
        const int8_t* qs; const half* dp; float* y; int n_out, r0;
        locate(g, qs, dp, y, n_out, r0);
#pragma unroll
        for (int rr = 0; rr < RB; ++rr) {
            const int row = r0 + rr;
            wv[rr] = row < n_out ? __ldg(reinterpret_cast<const int4*>(qs + std::size_t(row) * NIN) + c) : make_int4(0, 0, 0, 0);
            dh[rr] = row < n_out ? __ldg(dp + std::size_t(row) * NB + (c >> 1)) : __float2half(0.0f);
        }
    };

    int buf = 0;
    int4 wv[RB];
    half dh[RB];
    int g = int(blockIdx.x);
    if (PF) load(g, wv, dh);
    for (; g < S.groups; g += int(gridDim.x)) {
        const int8_t* qs; const half* dp; float* y; int n_out, r0;
        locate(g, qs, dp, y, n_out, r0);
        if (!PF) load(g, wv, dh);
#pragma unroll
        for (int rr = 0; rr < RB; ++rr) {
            float w[16];
            w16(wv[rr], w);
            const float dw = __half2float(dh[rr]);
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                float s = 0.0f;
#pragma unroll
                for (int e = 0; e < 16; ++e) s = fmaf(w[e], xf[j][e], s);
                sb[buf][rr][j][c] = make_float2(dw * xs[j], s);   // q8sm60's dw * sd, rounded the same
            }
        }
        if (PF) load(g + int(gridDim.x), wv, dh);
        __syncthreads();
        for (int p = warp; p < RB * NCOLS; p += CPL) {   // warp-uniform
            const int rr = p / NCOLS, j = p - rr * NCOLS, row = r0 + rr;
            if (row >= n_out) continue;
            float acc = 0.0f;
#pragma unroll
            for (int i = 0; i < CPL; ++i) {
                const float2 ps = sb[buf][rr][j][lane + 32 * i];
                acc += ps.x * ps.y;   // fma(dw * sd, float(s), acc): q8sm60's acc += dw * sd * float(s)
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o, 32);
            if (lane == 0) y[std::size_t(j) * n_out + row] = acc;
        }
        if (NBUF == 2) buf ^= 1;
        else __syncthreads();   // the next pass writes the one buffer
    }
}

}  // namespace q8f
