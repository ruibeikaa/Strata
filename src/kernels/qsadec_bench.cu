// src/kernels/qsadec_bench.cu - campaign batch 2 item C: the decode window's QSA select + attention on one die.
//
// One QSA layer of a decode window, as verify.cpp records it (window graphs past 65536 cells take the long select:
// qsa_block_scores_long + qsa_block_topk_long through the multi-CTA scratch; below, qsa_block_scores + qsa_block_topk;
// then qsa_decode_attn_batch), capacity 1M cells (--max-context 1048576), int8 K/V, T = 1..4 queries.  Every timing
// is a CUDA graph of `inner` layers, each behind an 8 MiB L2 flush (the experts between two QSA layers evict L2 in
// the engine); the flush graph's time is subtracted.  Rounds interleave the variants; medians are printed.
//
//   qsadec_bench [what=parity,sel,attn,abl] [ctxs=32768,115000,300000] [Ts=1,2,3,4] [rounds=5] [dump]
//
// parity: the production outputs against every proto variant, memcmp: the score rows [0, n_bid] and ids [0, width)
// of each query (normal keys, 7-prototype keys = heavy ties, half-zero keys), twice in a row (the fused histogram row
// must reset itself), and the attention output [T, 24, 256] (+ the chunk partials).
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsadec_proto.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace k = strata::kernels;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)

namespace {

__device__ __forceinline__ uint32_t hash32(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return (uint32_t) x;
}
__device__ __forceinline__ float unif(uint64_t x) { return (float) (hash32(x) >> 8) * (1.0f / 16777216.0f); }
__device__ __forceinline__ float gauss(uint64_t x) {
    const float u1 = fmaxf(unif(2 * x), 1e-7f), u2 = unif(2 * x + 1);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

// mode 0: 0.5 a_b dir + noise; 1: one of 7 prototype rows (ties); 2: every other row zero, the rest as 0
__global__ void gen_pooled(float* p, int64_t rows, int mode, uint32_t seed) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 128) return;
    const int64_t b = i / 128;
    const int d = (int) (i % 128);
    const float dir = gauss(((uint64_t) seed << 40) ^ (0xD1Cull << 20) ^ (uint64_t) d);
    if (mode == 1) {
        const int proto = (int) (hash32(((uint64_t) seed << 40) ^ (uint64_t) b) % 7);
        p[i] = 0.5f * (float) (proto - 3) * dir + 0.3f * gauss(((uint64_t) seed << 40) ^ (0xA00ull << 24) ^ (uint64_t) (proto * 128 + d));
        return;
    }
    if (mode == 2 && (b & 1)) { p[i] = 0.0f; return; }
    const float a = gauss(((uint64_t) seed << 40) ^ (0xBEEull << 28) ^ (uint64_t) b);
    p[i] = 0.5f * a * dir + gauss(((uint64_t) seed << 40) ^ (uint64_t) i);
}
__global__ void gen_float(float* p, int64_t n, float scale, uint32_t seed) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = scale * gauss(((uint64_t) seed << 40) ^ (uint64_t) i);
}
__global__ void gen_i8(int8_t* p, int64_t n, uint32_t seed) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = (int8_t) ((int) (hash32(((uint64_t) seed << 40) ^ (uint64_t) i) % 255) - 127);
}
__global__ void gen_scale(uint16_t* p, int64_t n, uint32_t seed) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = __half_as_ushort(__float2half(0.002f + 0.02f * unif(((uint64_t) seed << 40) ^ (uint64_t) i)));
}
__global__ void gen_iota(int32_t* p, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = (int32_t) i;
}
__global__ void flush_kernel(float4* buf, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        float4 v = buf[i];
        v.x += 1.0f;
        buf[i] = v;
    }
}
inline unsigned nblk(int64_t n) { return (unsigned) ((n + 255) / 256); }

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

std::vector<int64_t> parse_list(const char* s) {
    std::vector<int64_t> v;
    std::string x(s);
    size_t a = 0;
    while (a <= x.size()) {
        size_t b = x.find(',', a);
        if (b == std::string::npos) b = x.size();
        if (b > a) v.push_back(std::atoll(x.substr(a, b - a).c_str()));
        a = b + 1;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string what = argc > 1 ? argv[1] : "parity,sel,attn,abl";
    const std::vector<int64_t> ctxs = parse_list(argc > 2 ? argv[2] : "32768,115000,300000");
    const std::vector<int64_t> Ts = parse_list(argc > 3 ? argv[3] : "1,2,3,4");
    const int rounds = argc > 4 ? std::atoi(argv[4]) : 5;
    const char* dump = argc > 5 ? argv[5] : nullptr;
    const auto want = [&](const char* w) { return what.find(w) != std::string::npos; };

    int dev = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, dev));
    std::printf("device %d: %s sm_%d%d, %d SMs, %d MHz, pci %02x:%02x\n", dev, prop.name, prop.major, prop.minor,
                prop.multiProcessorCount, prop.clockRate / 1000, prop.pciBusID, prop.pciDeviceID);

    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t capacity = 1048576, max_blocks = capacity / 4 + 2;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const int MAXT = 4;
    const int64_t kv_cells = 310000, kv_pages = (kv_cells + 3) / 4, kv_rows = kv_pages * s.n_head_kv * s.page_size;
    std::printf("capacity %lld cells, max_blocks %lld, cap %lld, KV %lld cells int8\n", (long long) capacity,
                (long long) max_blocks, (long long) cap, (long long) kv_cells);
    {
        char buf[2048];
        k::qsadec_bench_attrs(buf, sizeof buf);
        std::printf("%s", buf);
    }

    float *pooled, *dead, *qidx, *qattn, *sc_a, *sc_b, *as_a, *as_b, *at_a, *at_b;
    int32_t *steps, *ids_a, *ids_b, *scr_a, *scr_b, *page_table;
    int8_t *k_q, *v_q;
    uint16_t *k_s, *v_s;
    const int64_t scr_ints = std::max<int64_t>(k::qsa_topk_multi_scratch_ints(MAXT), k::qsadec_select_scratch_ints(MAXT));
    const int64_t as_floats = (int64_t) k::qsa_decode_attn_scratch_floats(cap, s);
    CK(cudaMalloc(&pooled, (size_t) max_blocks * 128 * 4));
    CK(cudaMalloc(&dead, 128 * 4));
    CK(cudaMalloc(&qidx, (size_t) MAXT * 512 * 4));
    CK(cudaMalloc(&qattn, (size_t) MAXT * 24 * 256 * 4));
    CK(cudaMalloc(&sc_a, (size_t) MAXT * max_blocks * 4));
    CK(cudaMalloc(&sc_b, (size_t) MAXT * max_blocks * 4));
    CK(cudaMalloc(&as_a, (size_t) MAXT * as_floats * 4));
    CK(cudaMalloc(&as_b, (size_t) MAXT * as_floats * 4));
    CK(cudaMalloc(&at_a, (size_t) MAXT * 6144 * 4));
    CK(cudaMalloc(&at_b, (size_t) MAXT * 6144 * 4));
    CK(cudaMalloc(&steps, (size_t) MAXT * k::kStepCount * 4));
    CK(cudaMalloc(&ids_a, (size_t) MAXT * cap * 4));
    CK(cudaMalloc(&ids_b, (size_t) MAXT * cap * 4));
    CK(cudaMalloc(&scr_a, (size_t) scr_ints * 4));
    CK(cudaMalloc(&scr_b, (size_t) scr_ints * 4));
    CK(cudaMemset(scr_a, 0, (size_t) scr_ints * 4));
    CK(cudaMemset(scr_b, 0, (size_t) scr_ints * 4));
    CK(cudaMalloc(&page_table, (size_t) (capacity / 4) * 4));
    CK(cudaMalloc(&k_q, (size_t) kv_rows * 256));
    CK(cudaMalloc(&v_q, (size_t) kv_rows * 256));
    CK(cudaMalloc(&k_s, (size_t) kv_rows * 4 * 2));
    CK(cudaMalloc(&v_s, (size_t) kv_rows * 4 * 2));
    float4* flushbuf;
    const int64_t flush_f4 = (8ll << 20) / 16;
    CK(cudaMalloc(&flushbuf, (size_t) flush_f4 * 16));
    CK(cudaMemset(flushbuf, 0, (size_t) flush_f4 * 16));

    auto gen_keys = [&](int mode, uint32_t seed) {
        gen_pooled<<<nblk(max_blocks * 128), 256>>>(pooled, max_blocks, mode, seed);
        gen_float<<<1, 128>>>(dead, 128, 1.0f, seed + 99);
        gen_float<<<nblk(MAXT * 512), 256>>>(qidx, MAXT * 512, 0.15f, seed + 7);
        CK(cudaGetLastError());
    };
    gen_keys(0, 1);
    gen_float<<<nblk(MAXT * 6144), 256>>>(qattn, MAXT * 6144, 1.0f, 3);
    gen_i8<<<nblk(kv_rows * 256), 256>>>(k_q, kv_rows * 256, 4);
    gen_i8<<<nblk(kv_rows * 256), 256>>>(v_q, kv_rows * 256, 5);
    gen_scale<<<nblk(kv_rows * 4), 256>>>(k_s, kv_rows * 4, 6);
    gen_scale<<<nblk(kv_rows * 4), 256>>>(v_s, kv_rows * 4, 7);
    gen_iota<<<nblk(capacity / 4), 256>>>(page_table, capacity / 4);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    k::QsaAttnPools pools{};
    pools.k_q = k_q; pools.v_q = v_q; pools.k_scale = k_s; pools.v_scale = v_s; pools.page_table = page_table;

    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    std::vector<int32_t> hsteps;
    auto set_steps = [&](int64_t ctx, int T) {
        hsteps.assign((size_t) T * k::kStepCount, 0);
        for (int i = 0; i < T; ++i) {
            int32_t* sp = hsteps.data() + i * k::kStepCount;
            const int64_t pos = ctx - T + i;
            sp[k::kStepPos] = (int32_t) pos;
            sp[k::kStepNKv] = (int32_t) (pos + 1);
            sp[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
            sp[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
        }
        CK(cudaMemcpy(steps, hsteps.data(), hsteps.size() * 4, cudaMemcpyHostToDevice));
    };
    const int64_t long_at = 65536;
    // production: verify.cpp's idx_select (sel_long_ past 65536 cells) and the attention
    auto prod_select = [&](int64_t ctx, int T, float* sc, int32_t* ids, int32_t* scr) {
        if (ctx >= long_at) {
            k::qsa_block_scores_long(pooled, dead, qidx, steps, T, max_blocks, s, sc, st, -1);
            k::qsa_block_topk_long(sc, steps, T, max_blocks, cap, s, ids, st, -1, scr);
        } else {
            k::qsa_block_scores(pooled, dead, qidx, steps, T, max_blocks, s, sc, st);
            k::qsa_block_topk(sc, steps, T, max_blocks, cap, s, ids, st);
        }
    };
    auto prod_attn = [&](int T, const int32_t* ids, float* as, float* at) {
        k::qsa_decode_attn_batch(qattn, pools, ids, steps, cap, s, as, at, T, st);
    };

    // ---- timing: a graph of `inner` (flush + body); minus a graph of `inner` flushes
    const int inner = 8, launches = 6;
    auto graph_us = [&](const std::function<void()>& body, bool with_body) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < inner; ++i) {
            flush_kernel<<<96, 256, 0, st>>>(flushbuf, flush_f4);
            if (with_body) body();
        }
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        CK(cudaGraphLaunch(ge, st));
        CK(cudaStreamSynchronize(st));
        CK(cudaEventRecord(e0, st));
        for (int r = 0; r < launches; ++r) CK(cudaGraphLaunch(ge, st));
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        CK(cudaGraphExecDestroy(ge));
        CK(cudaGraphDestroy(g));
        return 1000.0 * ms / (launches * inner);
    };
    struct Arm { std::string name; std::function<void()> body; std::vector<double> t; };
    auto run_arms = [&](std::vector<Arm>& arms, const char* title, const std::function<void()>& before) {
        for (Arm& a : arms) {   // uncaptured once: lazy statics (occupancy queries) outside any capture
            if (before) before();
            a.body();
            CK(cudaStreamSynchronize(st));
        }
        for (int r = 0; r < rounds; ++r) {
            const double fl = graph_us([] {}, false);
            for (Arm& a : arms) {
                if (before) before();
                a.t.push_back(graph_us(a.body, true) - fl);
            }
        }
        const double base = median(arms[0].t);
        std::printf("%s\n", title);
        for (Arm& a : arms) {
            const double m = median(a.t);
            auto mm = std::minmax_element(a.t.begin(), a.t.end());
            std::printf("  %-34s %8.1f us  (x%.3f vs first; min %.1f max %.1f)\n", a.name.c_str(), m, base / m, *mm.first,
                        *mm.second);
        }
        std::fflush(stdout);
    };

    // ---------------------------------------------------------------- parity
    int64_t bad = 0, checked = 0;
    // QSADEC_NEG=1: the negative control - one bit of one input flipped between the production run and the variants
    const bool neg = std::getenv("QSADEC_NEG") != nullptr;
    auto flip = [&](float* d, int64_t i) {
        float v;
        CK(cudaMemcpy(&v, d + i, 4, cudaMemcpyDeviceToHost));
        uint32_t u;
        std::memcpy(&u, &v, 4);
        u ^= 1u;
        std::memcpy(&v, &u, 4);
        CK(cudaMemcpy(d + i, &v, 4, cudaMemcpyHostToDevice));
    };
    if (neg) std::printf("NEGATIVE CONTROL: one input bit flipped for the variants (every comparison should differ)\n");
    if (want("parity")) {
        std::vector<float> ha, hb;
        std::vector<int32_t> ia, ib;
        const int sel_variants[] = {0, 1, 2, 3, 4, 5, 10, 20, 21, 22, 23, 24};
        for (int mode = 0; mode < 3; ++mode) {
            gen_keys(mode, 11 + mode);
            CK(cudaDeviceSynchronize());
            for (int64_t ctx : {(int64_t) 70001, (int64_t) 115000, (int64_t) 300000, (int64_t) 300003, (int64_t) 1000000}) {
                for (int64_t T : Ts) {
                    set_steps(ctx, (int) T);
                    CK(cudaMemsetAsync(sc_a, 0, (size_t) MAXT * max_blocks * 4, st));
                    prod_select(ctx, (int) T, sc_a, ids_a, scr_a);
                    CK(cudaStreamSynchronize(st));
                    ha.resize((size_t) T * max_blocks);
                    ia.resize((size_t) T * cap);
                    CK(cudaMemcpy(ha.data(), sc_a, ha.size() * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(ia.data(), ids_a, ia.size() * 4, cudaMemcpyDeviceToHost));
                    if (neg) flip(qidx, 5);
                    for (int v : sel_variants) {
                        CK(cudaMemsetAsync(scr_b, 0, (size_t) scr_ints * 4, st));
                        for (int rep = 0; rep < 2; ++rep) {
                            CK(cudaMemsetAsync(sc_b, 0x7f, (size_t) MAXT * max_blocks * 4, st));
                            CK(cudaMemsetAsync(ids_b, 0x7f, (size_t) MAXT * cap * 4, st));
                            if (!k::qsadec_select(pooled, dead, qidx, steps, T, max_blocks, cap, s, sc_b, ids_b, scr_b, st, v)) {
                                std::printf("  select variant %d not applicable\n", v);
                                break;
                            }
                            CK(cudaStreamSynchronize(st));
                            hb.resize(ha.size());
                            ib.resize(ia.size());
                            CK(cudaMemcpy(hb.data(), sc_b, hb.size() * 4, cudaMemcpyDeviceToHost));
                            CK(cudaMemcpy(ib.data(), ids_b, ib.size() * 4, cudaMemcpyDeviceToHost));
                            for (int i = 0; i < T; ++i) {
                                const int32_t* sp = hsteps.data() + i * k::kStepCount;
                                const int64_t nb = sp[k::kStepNBid], w = sp[k::kStepWidth];
                                const size_t rows = (size_t) std::min<int64_t>(nb + 1, max_blocks);
                                const bool s_ok = std::memcmp(ha.data() + i * max_blocks, hb.data() + i * max_blocks, rows * 4) == 0;
                                const bool i_ok = std::memcmp(ia.data() + i * cap, ib.data() + i * cap, (size_t) w * 4) == 0;
                                checked += 2;
                                if (!s_ok || !i_ok) {
                                    bad += !s_ok + !i_ok;
                                    int64_t f = 0;
                                    while (f < (int64_t) rows && std::memcmp(&ha[(size_t) (i * max_blocks + f)], &hb[(size_t) (i * max_blocks + f)], 4) == 0) ++f;
                                    std::printf("MISMATCH select v%d mode %d ctx %lld T %lld q %d rep %d: scores %s (first %lld/%zu), ids %s\n",
                                                v, mode, (long long) ctx, (long long) T, i, rep, s_ok ? "same" : "DIFFER",
                                                (long long) f, rows, i_ok ? "same" : "DIFFER");
                                }
                            }
                        }
                    }
                    if (neg) flip(qidx, 5);
                    // the attention on the production ids
                    if (mode == 0 && ctx <= kv_cells) {
                        CK(cudaMemsetAsync(at_a, 0, (size_t) MAXT * 6144 * 4, st));
                        prod_attn((int) T, ids_a, as_a, at_a);
                        CK(cudaStreamSynchronize(st));
                        std::vector<float> oa((size_t) T * 6144), ob(oa.size()), pa((size_t) T * as_floats), pb(pa.size());
                        CK(cudaMemcpy(oa.data(), at_a, oa.size() * 4, cudaMemcpyDeviceToHost));
                        CK(cudaMemcpy(pa.data(), as_a, pa.size() * 4, cudaMemcpyDeviceToHost));
                        if (neg) flip(qattn, 5);
                        for (int v = 0; v <= 8; ++v) {
                            CK(cudaMemsetAsync(at_b, 0x7f, (size_t) MAXT * 6144 * 4, st));
                            CK(cudaMemsetAsync(as_b, 0, (size_t) MAXT * as_floats * 4, st));
                            if (!k::qsadec_attn(qattn, pools, ids_a, steps, cap, s, as_b, at_b, T, st, v)) {
                                std::printf("  attn variant %d not applicable\n", v);
                                continue;
                            }
                            CK(cudaStreamSynchronize(st));
                            CK(cudaMemcpy(ob.data(), at_b, ob.size() * 4, cudaMemcpyDeviceToHost));
                            CK(cudaMemcpy(pb.data(), as_b, pb.size() * 4, cudaMemcpyDeviceToHost));
                            const bool o_ok = std::memcmp(oa.data(), ob.data(), oa.size() * 4) == 0;
                            if (neg && v == 8) flip(qattn, 5);
                            // partials: acc, m, l of the chunks (the +64 spare per query is never written)
                            bool p_ok = true;
                            for (int i = 0; i < T; ++i)
                                p_ok = p_ok && std::memcmp(pa.data() + i * as_floats, pb.data() + i * as_floats, (size_t) (as_floats - 64) * 4) == 0;
                            checked += 2;
                            if (!o_ok || !p_ok) {
                                bad += !o_ok + !p_ok;
                                std::printf("MISMATCH attn v%d ctx %lld T %lld: output %s, partials %s\n", v, (long long) ctx,
                                            (long long) T, o_ok ? "same" : "DIFFER", p_ok ? "same" : "DIFFER");
                            }
                        }
                    }
                }
            }
            std::printf("parity keys mode %d done: %lld bad of %lld so far\n", mode, (long long) bad, (long long) checked);
            std::fflush(stdout);
        }
        gen_keys(0, 1);
        CK(cudaDeviceSynchronize());
        std::printf("PARITY %s: %lld of %lld comparisons differ\n", bad ? "FAIL" : "PASS", (long long) bad, (long long) checked);
    }

    // ---------------------------------------------------------------- timing
    for (int64_t ctx : ctxs) {
        for (int64_t T64 : Ts) {
            const int T = (int) T64;
            set_steps(ctx, T);
            prod_select(ctx, T, sc_a, ids_a, scr_a);   // ids for the attention arms
            CK(cudaStreamSynchronize(st));
            const bool lng = ctx >= long_at;
            char title[160];
            if (want("sel")) {
                std::vector<Arm> arms;
                arms.push_back({"prod select", [&] { prod_select(ctx, T, sc_a, ids_a, scr_a); }, {}});
                if (lng) {
                    arms.push_back({"copy select (dx+tail+6)", [&] { k::qsadec_select(pooled, dead, qidx, steps, T, max_blocks, cap, s, sc_b, ids_b, scr_b, st, 10); }, {}});
                    arms.push_back({"R unfused (R+tail+6)", [&] { k::qsadec_select(pooled, dead, qidx, steps, T, max_blocks, cap, s, sc_b, ids_b, scr_b, st, 1); }, {}});
                }
                arms.push_back({"PROTO DEFAULT (dxs fused auto)", [&] { k::qsadec_select(pooled, dead, qidx, steps, T, max_blocks, cap, s, sc_b, ids_b, scr_b, st, 0); }, {}});
                arms.push_back({"R fused 4w (+5)", [&] { k::qsadec_select(pooled, dead, qidx, steps, T, max_blocks, cap, s, sc_b, ids_b, scr_b, st, 5); }, {}});
                if (!lng) arms.push_back({"prod long path (SELECT_LONG_AT low)", [&] { k::qsa_block_scores_long(pooled, dead, qidx, steps, T, max_blocks, s, sc_b, st, -1); k::qsa_block_topk_long(sc_b, steps, T, max_blocks, cap, s, ids_b, st, -1, scr_b); }, {}});
                arms.push_back({"dxs2 unfused (+tail+6)", [&] { k::qsadec_select(pooled, dead, qidx, steps, T, max_blocks, cap, s, sc_b, ids_b, scr_b, st, 20); }, {}});
                arms.push_back({"dxs4 unfused (+tail+6)", [&] { k::qsadec_select(pooled, dead, qidx, steps, T, max_blocks, cap, s, sc_b, ids_b, scr_b, st, 21); }, {}});
                arms.push_back({"dxs1 fused (+5)", [&] { k::qsadec_select(pooled, dead, qidx, steps, T, max_blocks, cap, s, sc_b, ids_b, scr_b, st, 24); }, {}});
                arms.push_back({"dxs2 fused (+5)", [&] { k::qsadec_select(pooled, dead, qidx, steps, T, max_blocks, cap, s, sc_b, ids_b, scr_b, st, 22); }, {}});
                arms.push_back({"dxs4 fused (+5)", [&] { k::qsadec_select(pooled, dead, qidx, steps, T, max_blocks, cap, s, sc_b, ids_b, scr_b, st, 23); }, {}});
                arms.push_back({"proto v2 (8 warps/CTA)", [&] { k::qsadec_select(pooled, dead, qidx, steps, T, max_blocks, cap, s, sc_b, ids_b, scr_b, st, 2); }, {}});
                arms.push_back({"proto v3 (2 CTA/SM)", [&] { k::qsadec_select(pooled, dead, qidx, steps, T, max_blocks, cap, s, sc_b, ids_b, scr_b, st, 3); }, {}});
                arms.push_back({"proto v4 (3 CTA/SM)", [&] { k::qsadec_select(pooled, dead, qidx, steps, T, max_blocks, cap, s, sc_b, ids_b, scr_b, st, 4); }, {}});
                if (lng) {   // stages of the copies (each alone, on the scores in place)
                    auto stg = [&](int stage, int abl) { return [&, stage, abl] { k::qsadec_bench_select_stage(stage, abl, pooled, dead, qidx, steps, T, max_blocks, cap, sc_b, ids_b, scr_b, st); }; };
                    arms.push_back({"  dx scorer copy", stg(0, 0), {}});
                    arms.push_back({"  dx abl: no key loads", stg(0, 1), {}});
                    arms.push_back({"  dx abl: no tree", stg(0, 2), {}});
                    arms.push_back({"  tail", stg(7, 0), {}});
                    arms.push_back({"  hist pass 0", stg(1, 0), {}});
                    arms.push_back({"  hist pass 1", stg(2, 0), {}});
                    arms.push_back({"  hist pass 2", stg(3, 0), {}});
                    arms.push_back({"  hist pass 3", stg(4, 0), {}});
                    arms.push_back({"  count", stg(5, 0), {}});
                    arms.push_back({"  emit", stg(6, 0), {}});
                    arms.push_back({"  R scorer alone", stg(8, 0), {}});
                    arms.push_back({"  R scorer fused alone", stg(9, 0), {}});
                }
                std::snprintf(title, sizeof title, "== select ctx %lld T %d (%s path)", (long long) ctx, T, lng ? "long" : "short");
                // before every arm: a full copy run leaves valid top-k state (a stage alone then reads sane
                // thresholds: emit never writes past its row), then the fused histogram row zeroed
                const int64_t qints = k::qsadec_select_scratch_ints(1);
                run_arms(arms, title, [&] {
                    if (lng) k::qsadec_select(pooled, dead, qidx, steps, T, max_blocks, cap, s, sc_b, ids_b, scr_b, st, 10);
                    CK(cudaStreamSynchronize(st));
                    for (int i = 0; i < T; ++i) CK(cudaMemsetAsync(scr_b + i * qints, 0, 256 * 4, st));
                });
            }
            if (want("attn")) {
                std::vector<Arm> arms;
                arms.push_back({"prod attn (pre75 + merge v1)", [&] { prod_attn(T, ids_a, as_a, at_a); }, {}});
                arms.push_back({"copy chunk + merge v2", [&] { k::qsadec_attn(qattn, pools, ids_a, steps, cap, s, as_b, at_b, T, st, 3); }, {}});
                arms.push_back({"early loads + merge v1", [&] { k::qsadec_attn(qattn, pools, ids_a, steps, cap, s, as_b, at_b, T, st, 4); }, {}});
                arms.push_back({"early loads + merge v2", [&] { k::qsadec_attn(qattn, pools, ids_a, steps, cap, s, as_b, at_b, T, st, 2); }, {}});
                arms.push_back({"V staged + merge v1", [&] { k::qsadec_attn(qattn, pools, ids_a, steps, cap, s, as_b, at_b, T, st, 1); }, {}});
                arms.push_back({"V staged 33K + merge v2", [&] { k::qsadec_attn(qattn, pools, ids_a, steps, cap, s, as_b, at_b, T, st, 8); }, {}});
                arms.push_back({"PROTO DEFAULT (a2 auto + v2)", [&] { k::qsadec_attn(qattn, pools, ids_a, steps, cap, s, as_b, at_b, T, st, 0); }, {}});
                arms.push_back({"a2 3/SM + merge v1", [&] { k::qsadec_attn(qattn, pools, ids_a, steps, cap, s, as_b, at_b, T, st, 7); }, {}});
                arms.push_back({"a2 3/SM + merge v2", [&] { k::qsadec_attn(qattn, pools, ids_a, steps, cap, s, as_b, at_b, T, st, 5); }, {}});
                arms.push_back({"a2 2/SM + merge v2", [&] { k::qsadec_attn(qattn, pools, ids_a, steps, cap, s, as_b, at_b, T, st, 6); }, {}});
                std::snprintf(title, sizeof title, "== attention ctx %lld T %d", (long long) ctx, T);
                run_arms(arms, title, nullptr);
            }
            if (want("both") && lng) {
                std::vector<Arm> arms;
                arms.push_back({"prod select + attn", [&] { prod_select(ctx, T, sc_a, ids_a, scr_a); prod_attn(T, ids_a, as_a, at_a); }, {}});
                arms.push_back({"prod select + attn, merge v2", [&] { prod_select(ctx, T, sc_a, ids_a, scr_a); k::qsadec_attn(qattn, pools, ids_a, steps, cap, s, as_a, at_a, T, st, 3); }, {}});
                arms.push_back({"PROTO select + attn", [&] { k::qsadec_select(pooled, dead, qidx, steps, T, max_blocks, cap, s, sc_b, ids_b, scr_b, st, 0); k::qsadec_attn(qattn, pools, ids_b, steps, cap, s, as_b, at_b, T, st, 0); }, {}});
                std::snprintf(title, sizeof title, "== QSA decode layer (select + attention) ctx %lld T %d", (long long) ctx, T);
                run_arms(arms, title, [&] {
                    CK(cudaStreamSynchronize(st));
                    const int64_t qints = k::qsadec_select_scratch_ints(1);
                    for (int i = 0; i < T; ++i) CK(cudaMemsetAsync(scr_b + i * qints, 0, 256 * 4, st));
                });
            }
            if (want("abl")) {
                std::vector<Arm> arms;
                auto ab = [&](int abl, int merge) { return [&, abl, merge] { k::qsadec_bench_attn_abl(abl, merge, qattn, pools, ids_a, steps, cap, s, as_b, at_b, T, st); }; };
                arms.push_back({"copy chunk + merge v1", ab(0, 0), {}});
                arms.push_back({"copy chunk alone", ab(0, -1), {}});
                arms.push_back({"merge v1 alone", ab(-1, 0), {}});
                arms.push_back({"merge v2 alone", ab(-1, 1), {}});
                arms.push_back({"chunk: q not read", ab(1, -1), {}});
                arms.push_back({"chunk: K not read", ab(2, -1), {}});
                arms.push_back({"chunk: V not read", ab(4, -1), {}});
                arms.push_back({"chunk: K, V not read", ab(6, -1), {}});
                arms.push_back({"chunk: no value phase", ab(8, -1), {}});
                arms.push_back({"chunk: no score phase", ab(16, -1), {}});
                arms.push_back({"chunk: no ids/page reads", ab(32, -1), {}});
                arms.push_back({"chunk: no global reads (39)", ab(39, -1), {}});
                std::snprintf(title, sizeof title, "== attention ablations ctx %lld T %d (chunk + merge parts)", (long long) ctx, T);
                run_arms(arms, title, nullptr);
            }
        }
    }

    // ---------------------------------------------------------------- the recorded selections (exp\E3 dump)
    if (dump && want("dump")) {
        std::FILE* f = std::fopen(dump, "rb");
        if (!f) { std::perror(dump); return 1; }
        _fseeki64(f, 0, SEEK_END);
        const long long size = _ftelli64(f);
        long long off = 0;
        int used = 0;
        while (off + 16 <= size && used < 2) {
            _fseeki64(f, off, SEEK_SET);
            int32_t h[4];
            if (std::fread(h, 4, 4, f) != 4 || h[3] != cap || h[2] <= 0 || h[2] > 8192) break;
            const long long next = off + 16 + (long long) h[2] * h[3] * 4;
            if (h[1] > 100000 && h[2] >= 4 && ((used == 0 && h[0] == 3) || (used == 1 && h[0] == 9))) {
                for (int64_t T64 : Ts) {
                    const int T = (int) T64;
                    std::vector<int32_t> ids((size_t) T * cap);
                    _fseeki64(f, off + 16 + (long long) (h[2] - T) * cap * 4, SEEK_SET);
                    if (std::fread(ids.data(), 4, ids.size(), f) != ids.size()) break;
                    // steps of the rows' positions: row r of the record is position p0 + r
                    const int64_t ctx = (int64_t) h[1] + h[2];
                    set_steps(ctx, T);
                    CK(cudaMemcpy(ids_a, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
                    // parity on the recorded ids
                    prod_attn(T, ids_a, as_a, at_a);
                    k::qsadec_attn(qattn, pools, ids_a, steps, cap, s, as_b, at_b, T, st, 0);
                    CK(cudaStreamSynchronize(st));
                    std::vector<float> oa((size_t) T * 6144), ob(oa.size());
                    CK(cudaMemcpy(oa.data(), at_a, oa.size() * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(ob.data(), at_b, ob.size() * 4, cudaMemcpyDeviceToHost));
                    const bool ok = std::memcmp(oa.data(), ob.data(), oa.size() * 4) == 0;
                    std::printf("dump layer %d positions %lld..%lld: attention output %s\n", h[0], (long long) (ctx - T),
                                (long long) (ctx - 1), ok ? "same" : "DIFFERS");
                    ++checked;
                    bad += !ok;
                    std::vector<Arm> arms;
                    arms.push_back({"prod attn", [&] { prod_attn(T, ids_a, as_a, at_a); }, {}});
                    arms.push_back({"proto v0", [&] { k::qsadec_attn(qattn, pools, ids_a, steps, cap, s, as_b, at_b, T, st, 0); }, {}});
                    char title[160];
                    std::snprintf(title, sizeof title, "== attention on recorded ids (layer %d, ctx %lld) T %d", h[0], (long long) ctx, T);
                    run_arms(arms, title, nullptr);
                }
                ++used;
            }
            off = next;
        }
        std::fclose(f);
    }
    std::printf("DONE: %lld of %lld comparisons differ\n", (long long) bad, (long long) checked);
    return bad ? 1 : 0;
}
