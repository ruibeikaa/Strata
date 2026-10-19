// src/prefill/moe_ws_nib_parity.cu - P1 part D (K1), M4: the prompt path's expert products of a layer on 4-bit slots
// against the production WS kernels and the in-place MMQ (next13), bit for bit (D_DESIGN.md section 8(b)).
//
//     build/moe_ws_nib_parity <shard.gguf> [<shard.gguf> ...] [--fmt 22,18,21] [--layers 1,17]
//                             [--t 1,7,128,256,512,1792,8192] [--sigma 0.5,1.0] [--negative]
//
// Per model (any shard; the others are opened beside it), one layer of each converted format (or `--layers`), all 512
// experts read from the GGUF tensors into the blob the pack holds (gate rows | up rows | down rows), the blobs packed
// at their exact size; the 4-bit slots made from them by the engine's own transcoder (nibble::transcode, the down rows
// copied behind kDownOff), the last slot `hashed` bytes long so compute-sanitizer sees any read past it.  Per sigma
// and T (rows = 10 T, lognormal counts over 512 experts, each expert at most T rows: the ws2 count model; u = rows per
// routed expert runs 1 .. 160 over the list, so every TPP tier and the chunk tails are reached), three arms on the same
// q8_1 rows (mmq::quantize of random activations):
//   W  production WS on the GGUF blobs: ws::gate_up, one mmq::swiglu_quant over all rows, ws::down (prefill.cpp's
//      ws_layer path; where gate_up declines, its per-group in-place MMQ, as prefill.cpp does);
//   N  the same calls on the 4-bit slots (ws::Layer::gu_nib, down_off = kDownOff); gate_up must not decline;
//   M  the in-place MMQ of next13 on the GGUF blobs (STRATA_MMQ_WS=0's path): per 16-expert group a gate/up product,
//      swiglu_quant of the group's rows, a down product (prefill.cpp's `compute` with `ip`).
// GU ([rows][1280] floats) and Dm ([rows][2560]) are compared row by row (N vs W, M vs W); H as the q8_1 rows of
// ONE swiglu_quant over all rows (N's and W's as the WS path makes them; M's GU through the same call, its per-group
// Hq feeds its own down and so shows in its Dm).  W's GU and Dm are pre-filled with 0xFF and must have every word
// written and finite.  `--negative`: one code byte per slot changed after the transcode; N's GU must then differ.
// Exit 0 when everything is equal, 1 otherwise, 2 on a usage or setup error.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_nibble.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/moe_ws.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

namespace nib = strata::core::nibble;
namespace ws = strata::prefill::ws;
namespace mmq = strata::prefill::mmq;
namespace cpu = strata::kernels::cpu;

#define CK(x)                                                                                                       \
    do {                                                                                                            \
        cudaError_t e_ = (x);                                                                                       \
        if (e_ != cudaSuccess) {                                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));                 \
            std::exit(2);                                                                                           \
        }                                                                                                           \
    } while (0)

constexpr int64_t kEmbd = nib::kNEmbd, kFf = nib::kNFf, kK = 10, kNE = 512;
constexpr int kGroup = 16;   // prefill.cpp's MMQ_GROUP

// rows of `width` 32-bit words: the rows that differ, the words that differ, the first differing row
__global__ void cmp_rows(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b, long long rows, int width,
                         unsigned long long* bad_words, unsigned long long* bad_rows, unsigned long long* first_row) {
    const long long r = blockIdx.x;
    if (r >= rows) return;
    __shared__ int any;
    if (threadIdx.x == 0) any = 0;
    __syncthreads();
    int n = 0;
    for (int c = threadIdx.x; c < width; c += blockDim.x) n += a[r * width + c] != b[r * width + c];
    if (n) {
        atomicAdd(bad_words, (unsigned long long) n);
        atomicOr(&any, 1);
    }
    __syncthreads();
    if (threadIdx.x == 0 && any) {
        atomicAdd(bad_rows, 1ull);
        atomicMin(first_row, (unsigned long long) r);
    }
}
// words still at the 0xFF pre-fill, and non-finite floats
__global__ void sanity(const uint32_t* __restrict__ a, long long n, unsigned long long* unwritten, unsigned long long* nonfin) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (a[i] == 0xFFFFFFFFu) atomicAdd(unwritten, 1ull);
    else if (((a[i] >> 23) & 0xFF) == 0xFF) atomicAdd(nonfin, 1ull);
}
__global__ void fill_x(float* x, long long n, unsigned long long seed) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long h = seed ^ (unsigned long long) i * 0x9E3779B97F4A7C15ull;
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDull; h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ull; h ^= h >> 33;
    // a residual-stream-like spread: mostly small, a few large values per row
    const float u = (float) (h & 0xFFFFFF) / 16777216.0f - 0.5f;
    x[i] = ((h >> 40) % 97 == 0) ? 24.0f * u : 2.0f * u;
}

unsigned long long *g_bw, *g_br, *g_first, *g_un, *g_nf;

struct Cmp {
    unsigned long long rows = 0, words = 0, first = 0;
};
Cmp compare(const void* a, const void* b, long long rows, int width) {
    CK(cudaMemset(g_bw, 0, 8));
    CK(cudaMemset(g_br, 0, 8));
    CK(cudaMemset(g_first, 0x7F, 8));
    if (rows > 0) cmp_rows<<<(unsigned) rows, 128>>>((const uint32_t*) a, (const uint32_t*) b, rows, width, g_bw, g_br, g_first);
    CK(cudaGetLastError());
    Cmp c;
    CK(cudaMemcpy(&c.words, g_bw, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&c.rows, g_br, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&c.first, g_first, 8, cudaMemcpyDeviceToHost));
    return c;
}
std::pair<unsigned long long, unsigned long long> check_ref(const void* a, long long words) {
    CK(cudaMemset(g_un, 0, 8));
    CK(cudaMemset(g_nf, 0, 8));
    sanity<<<(unsigned) ((words + 255) / 256), 256>>>((const uint32_t*) a, words, g_un, g_nf);
    CK(cudaGetLastError());
    unsigned long long un = 0, nf = 0;
    CK(cudaMemcpy(&un, g_un, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&nf, g_nf, 8, cudaMemcpyDeviceToHost));
    return { un, nf };
}

// One layer's 512 experts: GGUF blobs and converted slots on the device.
struct Layer {
    std::string tag;
    int layer = -1;
    cpu::NativeFmt f;
    nib::LayerVram V;
    uint8_t* raw = nullptr;     // kNE blobs, f.bytes apart, exact size
    uint8_t* slots = nullptr;   // kNE slots, V.slot_bytes apart, the last `hashed` long
};

bool load_layer(const strata::GgufModel& m, int l, bool negative, Layer& Y, long long& fails) {
    const char* roles[3] = { "gate", "up", "down" };
    const strata::TensorInfo* t[3] = {};
    size_t sh[3] = {};
    for (int r = 0; r < 3; ++r) t[r] = m.find("blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight", &sh[r]);
    if (!t[0] || !t[1] || !t[2]) return false;
    std::string err;
    if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, kEmbd, kFf, Y.f, err)) {
        std::printf("layer %d: %s\n", l, err.c_str());
        return false;
    }
    Y.layer = l;
    const size_t gu_role = Y.f.up_off, dn_role = Y.f.bytes - Y.f.down_off;
    std::vector<uint8_t> host((size_t) kNE * Y.f.bytes);
    const uint8_t* G = m.shard(sh[0]).tensor_data(*t[0]);
    const uint8_t* U = m.shard(sh[1]).tensor_data(*t[1]);
    const uint8_t* D = m.shard(sh[2]).tensor_data(*t[2]);
    for (int64_t e = 0; e < kNE; ++e) {
        uint8_t* b = host.data() + (size_t) e * Y.f.bytes;
        std::memcpy(b, G + (size_t) e * gu_role, gu_role);
        std::memcpy(b + Y.f.up_off, U + (size_t) e * gu_role, gu_role);
        std::memcpy(b + Y.f.down_off, D + (size_t) e * dn_role, dn_role);
    }
    nib::Config c;
    c.on = true;
    Y.V = nib::plan_layer(l, Y.f, c);
    if (!Y.V.on) {
        std::printf("layer %d: plan_layer does not convert it\n", l);
        return false;
    }
    CK(cudaMalloc(&Y.raw, host.size()));
    CK(cudaMemcpy(Y.raw, host.data(), host.size(), cudaMemcpyHostToDevice));
    const size_t sb = (size_t) (kNE - 1) * Y.V.slot_bytes + Y.V.hashed;
    CK(cudaMalloc(&Y.slots, sb));
    CK(cudaMemset(Y.slots, 0xA5, sb));
    std::vector<unsigned long long> hp(kNE);
    for (int64_t e = 0; e < kNE; ++e) hp[e] = (unsigned long long) (Y.slots + (size_t) e * Y.V.slot_bytes);
    unsigned long long *dp = nullptr, *dbad = nullptr;
    CK(cudaMalloc(&dp, kNE * 8));
    CK(cudaMalloc(&dbad, 8));
    CK(cudaMemcpy(dp, hp.data(), kNE * 8, cudaMemcpyHostToDevice));
    CK(cudaMemset(dbad, 0, 8));
    for (int64_t e0 = 0; e0 < kNE; e0 += 64)
        if (nib::transcode(Y.V, Y.raw + (size_t) e0 * Y.V.raw_bytes, 64, dp + e0, dbad, nullptr) != 0) std::exit(2);
    for (int64_t e = 0; e < kNE; ++e)
        CK(cudaMemcpyAsync(Y.slots + (size_t) e * Y.V.slot_bytes + nib::kDownOff,
                           Y.raw + (size_t) e * Y.V.raw_bytes + Y.V.raw_down_off, Y.V.down_bytes, cudaMemcpyDeviceToDevice));
    CK(cudaDeviceSynchronize());
    unsigned long long bad = 0;
    CK(cudaMemcpy(&bad, dbad, 8, cudaMemcpyDeviceToHost));
    std::printf("\n##### %s layer %d %s / down %d: 512 experts, blob %zu B, slot %llu B; transcode: %llu items do not round-trip\n",
                Y.tag.c_str(), l, nib::fmt_name(Y.f.gu_type), Y.f.d_type, Y.f.bytes, (unsigned long long) Y.V.slot_bytes, bad);
    fails += (long long) bad;
    if (negative) {
        for (int64_t e = 0; e < kNE; ++e) {
            uint8_t* at = Y.slots + (size_t) e * Y.V.slot_bytes + nib::code_off(5, 3);
            uint8_t v = 0;
            CK(cudaMemcpy(&v, at, 1, cudaMemcpyDeviceToHost));
            v ^= 0x11;
            CK(cudaMemcpy(at, &v, 1, cudaMemcpyHostToDevice));
        }
        std::printf("NEGATIVE CONTROL: one code byte changed per slot - N's GU must differ below\n");
    }
    CK(cudaFree(dp));
    CK(cudaFree(dbad));
    return true;
}

std::vector<double> parse_list(const char* s) {
    std::vector<double> v;
    std::string t(s);
    size_t p = 0;
    while (p < t.size()) {
        size_t q = t.find(',', p);
        if (q == std::string::npos) q = t.size();
        v.push_back(std::atof(t.substr(p, q - p).c_str()));
        p = q + 1;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> shards;
    std::vector<double> ts = { 1, 7, 128, 256, 512, 1792, 8192 }, sigmas = { 0.5, 1.0 }, fmts = { 22, 18, 21 }, only;
    bool negative = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--t") ts = parse_list(next());
        else if (a == "--sigma") sigmas = parse_list(next());
        else if (a == "--fmt") fmts = parse_list(next());
        else if (a == "--layers") only = parse_list(next());
        else if (a == "--negative") negative = true;
        else if (a.size() > 5 && a.substr(a.size() - 5) == ".gguf") shards.push_back(a);
        else {
            std::fprintf(stderr, "usage: moe_ws_nib_parity <shard.gguf> [...] [--fmt 22,18,21] [--layers L,..] "
                                 "[--t 1,7,128,256,512,1792,8192] [--sigma 0.5,1.0] [--negative]\n");
            return 2;
        }
    }
    if (shards.empty()) {
        std::fprintf(stderr, "moe_ws_nib_parity: give at least one model shard (.gguf)\n");
        return 2;
    }
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, 0));
    char bus[64] = {};
    CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
    std::printf("device 0: %s, bus %s, cc %d.%d, %d SMs; ws::mode() %d; mmq::built() %d\n", p.name, bus, p.major, p.minor,
                p.multiProcessorCount, ws::mode(), (int) mmq::built());
    if (ws::mode() != 1 || !mmq::built()) {
        std::fprintf(stderr, "moe_ws_nib_parity: needs the WS kernels (sm_60, STRATA_MMQ_WS unset or 1) and MMQ\n");
        return 2;
    }
    CK(cudaMalloc(&g_bw, 8));
    CK(cudaMalloc(&g_br, 8));
    CK(cudaMalloc(&g_first, 8));
    CK(cudaMalloc(&g_un, 8));
    CK(cudaMalloc(&g_nf, 8));
    mmq::Context ctx;
    long long fails = 0, cmps = 0, rows_cmp = 0;
    for (size_t si = 0; si < shards.size(); ++si) {
        std::unique_ptr<strata::GgufModel> model;
        try {
            model = std::make_unique<strata::GgufModel>(strata::GgufModel::open(shards[si]));
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "moe_ws_nib_parity: %s\n", ex.what());
            return 2;
        }
        std::printf("\n#### model %c: %s\n", (char) ('A' + si), shards[si].c_str());
        std::vector<int> pick;
        std::vector<int> seen;
        for (int l = 0; l < 256; ++l) {
            const auto* g = model->find("blk." + std::to_string(l) + ".ffn_gate_exps.weight");
            if (g == nullptr) break;
            const int gt = (int) g->type;
            if (!nib::fmt_built(gt) || std::find(fmts.begin(), fmts.end(), (double) gt) == fmts.end()) continue;
            if (only.empty() ? std::find(seen.begin(), seen.end(), gt) == seen.end()
                             : std::find(only.begin(), only.end(), (double) l) != only.end()) {
                pick.push_back(l);
                seen.push_back(gt);
            }
        }
        for (int l : pick) {
            Layer Y;
            Y.tag = std::string(1, (char) ('A' + si));
            if (!load_layer(*model, l, negative, Y, fails)) {
                ++fails;
                continue;
            }
            const int gt = Y.f.gu_type, dt = Y.f.d_type;
            const bool ip = mmq::inplace(gt, 1280, kEmbd) && mmq::inplace(dt, kEmbd, kFf);
            std::printf("ws::supported %d, mmq::inplace %d, swiglu_quant_ok %d\n", (int) ws::supported(gt, dt), (int) ip,
                        (int) mmq::swiglu_quant_ok(dt));
            if (!ws::supported(gt, dt) || !ip || !mmq::swiglu_quant_ok(dt)) {
                ++fails;
                continue;
            }
            int64_t *dxp = nullptr, *dxn = nullptr;
            CK(cudaMalloc(&dxp, kNE * 8));
            CK(cudaMalloc(&dxn, kNE * 8));
            for (double sigma : sigmas)
                for (double tt : ts) {
                    const int T = (int) tt;
                    const int rows = T * (int) kK;
                    // lognormal counts over the 512 experts, each at most T, summing to rows (the ws2 model)
                    std::mt19937_64 rng(1000003ull * (uint64_t) T + (uint64_t) (sigma * 1000) + 7 + (uint64_t) l);
                    std::vector<int32_t> cnt(kNE, 0);
                    {
                        std::normal_distribution<double> nd(0.0, 1.0);
                        std::vector<double> wgt(kNE);
                        for (auto& x : wgt) x = std::exp(sigma * nd(rng));
                        const double S = std::accumulate(wgt.begin(), wgt.end(), 0.0);
                        std::vector<std::pair<double, int>> fr;
                        int sum = 0;
                        for (int e = 0; e < kNE; ++e) {
                            const double cc = rows * wgt[e] / S;
                            cnt[e] = std::min((int) std::floor(cc), T);
                            sum += cnt[e];
                            fr.push_back({ cc - std::floor(cc), e });
                        }
                        std::sort(fr.begin(), fr.end(), [](auto& a, auto& b) { return a.first > b.first; });
                        for (size_t i = 0; sum < rows && i < 4 * (size_t) kNE; ++i) {
                            const int e = fr[i % kNE].second;
                            if (cnt[e] < T) { ++cnt[e]; ++sum; }
                        }
                    }
                    std::vector<int32_t> order;
                    int R = 0;
                    for (int e = 0; e < kNE; ++e)
                        if (cnt[e] > 0) { order.push_back(e); R += cnt[e]; }
                    // the resident-sort order (STRATA_MMQ_RESIDENT_SORT_NE=1 sorts by count; ws::chunks re-sorts anyway)
                    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return cnt[a] != cnt[b] ? cnt[a] < cnt[b] : a < b; });
                    const int n = (int) order.size();
                    std::vector<int32_t> bounds(n + 1, 0);
                    for (int j = 0; j < n; ++j) bounds[j + 1] = bounds[j] + cnt[order[j]];
                    std::vector<ws::Chunk> ch(ws::max_chunks(R, n));
                    const size_t nch = ws::chunks(order.data(), (size_t) n, cnt.data(), bounds.data(), ch.data());
                    size_t tails = 0;
                    for (size_t i = 0; i < nch; ++i) tails += ch[i].rows < ws::kChunkRows;
                    // device tables: bounds (absolute n+1, then per group relative), offsets, chunks, identity ids
                    const int ng = (n + kGroup - 1) / kGroup;
                    std::vector<int32_t> bh(n + 1 + (size_t) ng * (kGroup + 1));
                    std::copy(bounds.begin(), bounds.end(), bh.begin());
                    for (int g = 0; g < ng; ++g)
                        for (int i = 0; i <= kGroup; ++i)
                            bh[n + 1 + (size_t) g * (kGroup + 1) + i] = bounds[std::min(n, g * kGroup + i)] - bounds[g * kGroup];
                    int32_t *dbounds = nullptr, *dids = nullptr;
                    CK(cudaMalloc(&dbounds, bh.size() * 4));
                    CK(cudaMemcpy(dbounds, bh.data(), bh.size() * 4, cudaMemcpyHostToDevice));
                    CK(cudaMalloc(&dids, (size_t) R * 4));
                    mmq::iota(dids, R, nullptr);
                    std::vector<int64_t> hxp(n), hxn(n);
                    for (int j = 0; j < n; ++j) {
                        hxp[j] = (int64_t) order[j] * (int64_t) Y.f.bytes;
                        hxn[j] = (int64_t) order[j] * (int64_t) Y.V.slot_bytes;
                    }
                    CK(cudaMemcpy(dxp, hxp.data(), (size_t) n * 8, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(dxn, hxn.data(), (size_t) n * 8, cudaMemcpyHostToDevice));
                    ws::Chunk* dch = nullptr;
                    CK(cudaMalloc(&dch, nch * sizeof(ws::Chunk)));
                    CK(cudaMemcpy(dch, ch.data(), nch * sizeof(ws::Chunk), cudaMemcpyHostToDevice));
                    // the activations: random floats, quantized as quantize_scatter would (each row once)
                    float* X = nullptr;
                    void* xq = nullptr;
                    CK(cudaMalloc(&X, (size_t) R * kEmbd * 4));
                    CK(cudaMalloc(&xq, mmq::q8_bytes(R, kEmbd)));
                    fill_x<<<(unsigned) (((long long) R * kEmbd + 255) / 256), 256>>>(X, (long long) R * kEmbd,
                                                                                      0xA11CEull + (uint64_t) R + (uint64_t) gt);
                    CK(cudaGetLastError());
                    mmq::quantize(X, nullptr, xq, gt, kEmbd, kEmbd, R, nullptr);
                    CK(cudaDeviceSynchronize());
                    CK(cudaFree(X));
                    // three arms' GU, Hq and Dm
                    const size_t gub = (size_t) R * 1280 * 4, dmb = (size_t) R * kEmbd * 4, hqb = mmq::q8_bytes(R, kFf);
                    float *GU[3], *DM[3];
                    void* HQ[3];
                    void* HQg = nullptr;
                    for (int a = 0; a < 3; ++a) {
                        CK(cudaMalloc(&GU[a], gub));
                        CK(cudaMalloc(&DM[a], dmb));
                        CK(cudaMalloc(&HQ[a], hqb));
                        CK(cudaMemset(GU[a], a == 0 ? 0xFF : 0x00, gub));
                        CK(cudaMemset(DM[a], a == 0 ? 0xFF : 0x00, dmb));
                        CK(cudaMemset(HQ[a], 0xFF, hqb));   // the same fill: the q8 rows' padding is not written
                    }
                    CK(cudaMalloc(&HQg, mmq::q8_bytes(R, kFf)));
                    const double u = (double) R / n;
                    std::printf("\n=== %s%d %s sigma %.1f T %d: rows %d, routed %d (u = %.2f), chunks %zu (%zu short), groups %d; "
                                "faster(gu) %d faster(down) %d\n",
                                Y.tag.c_str(), l, nib::fmt_name(gt), sigma, T, R, n, u, nch, tails, ng,
                                (int) ws::faster(gt, u), (int) ws::faster(dt, u));
                    // the per-group in-place MMQ gate/up of groups [j0, j1) into gu (prefill.cpp's fallback and arm M)
                    auto mmq_gu = [&](int j0, int j1, float* gu) {
                        int64_t maxr = 0;
                        for (int i = j0; i < j1; ++i) maxr = std::max<int64_t>(maxr, cnt[order[i]]);
                        mmq::Product pr;
                        pr.w = Y.raw; pr.type = gt; pr.w_rows = 1280; pr.w_cols = kEmbd; pr.expert_bytes = Y.f.up_off * 2;
                        pr.n = j1 - j0; pr.xq = xq; pr.bounds = dbounds + j0; pr.ids = dids; pr.total_rows = R;
                        pr.max_rows = maxr; pr.dst = gu; pr.ld_dst = 1280; pr.xoff = dxp + j0;
                        ctx.run(pr, nullptr);
                    };
                    // arms W and N: prefill.cpp's ws_layer path
                    bool declined[2] = { false, false };
                    for (int a = 0; a < 2; ++a) {
                        ws::Layer L;
                        L.gu_type = gt; L.d_type = dt; L.w = a == 0 ? (const void*) Y.raw : (const void*) Y.slots;
                        L.xoff = a == 0 ? dxp : dxn;
                        L.down_off = (int64_t) (a == 0 ? Y.f.down_off : nib::kDownOff);
                        L.chunks = dch; L.nchunks = (int64_t) nch; L.rows = R; L.n = n; L.xq = xq;
                        L.gu = GU[a]; L.hq = HQ[a]; L.dm = DM[a]; L.gu_nib = a == 1;
                        if (!ws::gate_up(L, nullptr)) {
                            declined[a] = true;
                            if (a == 0)
                                for (int j0 = 0; j0 < n; j0 += kGroup) mmq_gu(j0, std::min(n, j0 + kGroup), GU[0]);
                        }
                        mmq::swiglu_quant(GU[a], HQ[a], R, false, nullptr);
                        ws::down(L, nullptr);
                    }
                    // arm M: per 16-expert group, gate/up, the group's H, down (in place)
                    for (int g = 0; g < ng; ++g) {
                        const int j0 = g * kGroup, j1 = std::min(n, j0 + kGroup);
                        mmq_gu(j0, j1, GU[2]);
                        const int64_t r0 = bounds[j0], nr = bounds[j1] - r0;
                        int64_t maxr = 0;
                        for (int i = j0; i < j1; ++i) maxr = std::max<int64_t>(maxr, cnt[order[i]]);
                        mmq::swiglu_quant(GU[2] + r0 * 1280, HQg, nr, false, nullptr);
                        mmq::Product dn;
                        dn.w = Y.raw; dn.type = dt; dn.w_rows = kEmbd; dn.w_cols = kFf; dn.expert_bytes = Y.f.bytes - Y.f.down_off;
                        dn.n = j1 - j0; dn.xq = HQg; dn.bounds = dbounds + n + 1 + (size_t) g * (kGroup + 1); dn.ids = dids;
                        dn.total_rows = nr; dn.max_rows = maxr; dn.dst = DM[2] + r0 * kEmbd; dn.ld_dst = kEmbd;
                        dn.xoff = dxp + j0; dn.xadd = (int64_t) Y.f.down_off;
                        ctx.run(dn, nullptr);
                    }
                    mmq::swiglu_quant(GU[2], HQ[2], R, false, nullptr);   // M's H as one pass makes it, for the H check
                    CK(cudaGetLastError());
                    CK(cudaDeviceSynchronize());
                    if (declined[0]) std::printf("note: ws::gate_up declined the GGUF layer (W took the per-group MMQ)\n");
                    if (declined[1]) {
                        std::printf("FAIL ws::gate_up declined the 4-bit layer (prefill.cpp would stop: nibble::fatal)\n");
                        ++fails;
                    }
                    const auto s_gu = check_ref(GU[0], (long long) R * 1280), s_dm = check_ref(DM[0], (long long) R * kEmbd);
                    std::printf("REF   W        GU: unwritten %llu, non-finite %llu | Dm: unwritten %llu, non-finite %llu %s\n",
                                s_gu.first, s_gu.second, s_dm.first, s_dm.second,
                                s_gu.first || s_gu.second || s_dm.first || s_dm.second ? "BAD" : "ok");
                    if (s_gu.first || s_gu.second || s_dm.first || s_dm.second) ++fails;
                    const char* an[3] = { "W (WS GGUF)", "N (WS 4-bit)", "M (MMQ in place)" };
                    for (int a = 1; a < 3; ++a) {
                        const Cmp g = compare(GU[0], GU[a], R, 1280);
                        const Cmp h = compare(HQ[0], HQ[a], 1, (int) (hqb / 4));
                        const Cmp d = compare(DM[0], DM[a], R, (int) kEmbd);
                        cmps += 3;
                        rows_cmp += 2LL * R;
                        const bool bad = g.rows || h.rows || d.rows;
                        std::printf("CHECK %-16s vs W: GU %d rows, %llu differ (%llu words) | H %s | Dm %d rows, %llu differ (%llu words)%s\n",
                                    an[a], R, g.rows, g.words, h.rows ? "DIFFERS" : "equal", R, d.rows, d.words,
                                    bad ? " DIFF" : "");
                        if (bad) {
                            ++fails;
                            if (g.rows) std::printf("      first differing GU row %llu\n", g.first);
                        }
                    }
                    for (int a = 0; a < 3; ++a) {
                        CK(cudaFree(GU[a]));
                        CK(cudaFree(DM[a]));
                        CK(cudaFree(HQ[a]));
                    }
                    CK(cudaFree(HQg));
                    CK(cudaFree(xq));
                    CK(cudaFree(dch));
                    CK(cudaFree(dbounds));
                    CK(cudaFree(dids));
                }
            CK(cudaFree(dxp));
            CK(cudaFree(dxn));
            CK(cudaFree(Y.raw));
            CK(cudaFree(Y.slots));
        }
    }
    std::printf("\n%s: %lld comparisons, %lld rows compared, %lld failures\n", fails ? "FAIL" : "ALL CHECKS EQUAL", cmps,
                rows_cmp, fails);
    return fails ? 1 : 0;
}
