// src/prefill/gemm_sass.cuh - STRATA_PF_HFMA_SASS: the hand-scheduled SASS twins of gemm_hfma.cuh's gemm_kernel and gemm_hfma_hc.cuh's
// gemm_sk_kernel, loaded through the CUDA driver API from a cubin embedded in the binary (sass/gemm_sass_cubin.inc), launched on the
// caller's stream.  GP100 (sm_60) only.
//
// What they are.  The same arithmetic in the same order as the compiler's kernels - the products of a lane's 64 values in FP16 pairs by
// HFMA2, widened into the FP32 accumulator every 4 stages with HADD2.F32 x2 + FADD - so the outputs are bit for bit the compiler
// kernels' (fp16-work/run-3.txt; sass-work/run_*.txt: every output of every shape compared, tails, inf and NaN included).  What
// differs is only the instruction schedule: register-bank-alternating HFMA2 order with operand reuse, no scoreboards on HFMA2, memory
// instructions dual-issued with HFMA2, LDS.64 fragments - 81% of the HFMA2 peak against the compiler's 64% in the loop.  The cubin is
// made by sass/gen_hfma.py (see its head), one kernel per (N, K) shape: the strides are immediates.
//
// What they cover (everything else keeps the compiler kernel, counted by reason and printed with the read's report):
//   dense  K 2560: N 10240 (attn_qkv), 6144 (attn_gate), 12288 (attn_q), 512 (attn_k, attn_v), 640 (ffn_gate/up_shexp)
//          K 6144: N 2560 (ssm_out, attn_output);  K 640: N 2560 (ffn_down_shexp)
//   hc     down N 320 K 10240 (split-K, the compiler path's own partition table: the parts' sums are the same bits),
//          up N 10240 K 320 (never split: the compiler path's Z = 3 gives each part one 4-stage group and a reduce kernel that sums
//          the three in order - the same FP32 additions as one pass, and 0.5 ms of memory traffic for a 1.0 ms product)
//   T >= 128 rows (the last row tile is shifted up so every load is in bounds; a slice below 128 rows keeps the compiler kernel),
//   X row stride == K, Y row stride == N, X 16-byte aligned.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace strata::prefill::sass {

#include "sass/gemm_sass_cubin.inc"   // static const unsigned char kCubin[]; static const size_t kCubinSize

enum Kern { kD10240x2560, kD6144x2560, kD12288x2560, kD512x2560, kD640x2560, kD2560x6144, kD2560x640, kHcDown, kHcUp, kKernCount };
struct KernInfo { const char* sym; int N, K; };
constexpr KernInfo kKern[kKernCount] = {
    {"d_n10240_k2560", 10240, 2560}, {"d_n6144_k2560", 6144, 2560}, {"d_n12288_k2560", 12288, 2560}, {"d_n512_k2560", 512, 2560},
    {"d_n640_k2560", 640, 2560},     {"d_n2560_k6144", 2560, 6144}, {"d_n2560_k640", 2560, 640},
    {"h_down_n320_k10240", 320, 10240}, {"h_up_n10240_k320", 10240, 320}};

namespace detail {

struct Api {
    bool ok = false;
    decltype(&cuModuleLoadData) load = nullptr;
    decltype(&cuModuleGetFunction) getfn = nullptr;
    decltype(&cuLaunchKernel) launch = nullptr;
    decltype(&cuCtxGetCurrent) cur = nullptr;
};
template <class F> inline bool resolve(const char* name, F& f) {
    cudaDriverEntryPointQueryResult q{};
    void* p = nullptr;
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) { (void) cudaGetLastError(); return false; }
    f = (F) p;
    return true;
}
inline const Api& api() {
    static const Api a = [] {
        Api x;
        x.ok = resolve("cuModuleLoadData", x.load) && resolve("cuModuleGetFunction", x.getfn) && resolve("cuLaunchKernel", x.launch) &&
               resolve("cuCtxGetCurrent", x.cur);
        return x;
    }();
    return a;
}
struct Dev {
    bool tried = false, ok = false;
    CUmodule mod = nullptr;
    CUfunction fn[kKernCount] = {};
};
inline std::mutex g_mu;
inline Dev g_dev[64];

// the device's module, loaded once into the device's primary context (the runtime's: the calling thread is on this device)
inline const Dev* dev_module(int dev) {
    std::lock_guard<std::mutex> lk(g_mu);
    Dev& d = g_dev[dev];
    if (d.tried) return d.ok ? &d : nullptr;
    d.tried = true;
    const Api& a = api();
    if (!a.ok) { std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_SASS: the driver entry points are missing: the compiler kernels stay\n"); return nullptr; }
    cudaFree(nullptr);   // the primary context exists and is current on this thread
    CUcontext ctx = nullptr;
    if (a.cur(&ctx) != CUDA_SUCCESS || ctx == nullptr) { std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_SASS: GPU %d: no current context: the compiler kernels stay\n", dev); return nullptr; }
    CUresult r = a.load(&d.mod, kCubin);
    if (r != CUDA_SUCCESS) { std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_SASS: GPU %d: cuModuleLoadData failed (%d): the compiler kernels stay\n", dev, (int) r); return nullptr; }
    for (int k = 0; k < kKernCount; ++k) {
        r = a.getfn(&d.fn[k], d.mod, kKern[k].sym);
        if (r != CUDA_SUCCESS) { std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_SASS: GPU %d: kernel %s missing (%d): the compiler kernels stay\n", dev, kKern[k].sym, (int) r); return nullptr; }
    }
    d.ok = true;
    std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_SASS: GPU %d loaded %d hand-scheduled HFMA2 kernels (%zu bytes of cubin)\n", dev, (int) kKernCount, (size_t) kCubinSize);
    return &d;
}

#pragma pack(push, 4)
struct ArgsDense { CUdeviceptr X, W, Y; int Tm128; };
struct ArgsHc { CUdeviceptr X, W, Y, ysc, bad; int Tm128; };
struct ArgsHcz { CUdeviceptr X, W, Y, ysc, bad; int Tm128; unsigned tab[24]; };
#pragma pack(pop)
static_assert(sizeof(ArgsDense) == 28 && sizeof(ArgsHc) == 44 && sizeof(ArgsHcz) == 140, "the kernels' parameter blocks");

template <class A>
inline CUresult launch(CUfunction f, unsigned gx, unsigned gy, unsigned gz, A& a, cudaStream_t st) {
    size_t sz = sizeof(A);
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &a, CU_LAUNCH_PARAM_BUFFER_SIZE, &sz, CU_LAUNCH_PARAM_END};
    return api().launch(f, gx, gy, gz, 256, 1, 1, 0, (CUstream) st, nullptr, extra);
}
// STRATA_PF_HFMA_SASS_SYNC=1 (diagnostic): synchronize after every hand-scheduled launch and name the call that faulted
inline bool sync_debug() {
    static const bool on = [] { const char* e = std::getenv("STRATA_PF_HFMA_SASS_SYNC"); return e != nullptr && std::atoi(e) != 0; }();
    return on;
}
inline void check(const char* what, int k, int64_t T, int64_t N, int64_t K, const void* a, const void* b, const void* c, cudaStream_t st, int dev) {
    if (!sync_debug()) return;
    const cudaError_t e = cudaStreamSynchronize(st);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_SASS_SYNC: GPU %d: %s %s T=%lld N=%lld K=%lld (X %p W %p Y %p) failed: %s\n", dev, what,
                     k >= 0 ? kKern[k].sym : "?", (long long) T, (long long) N, (long long) K, a, b, c, cudaGetErrorString(e));
        std::exit(1);
    }
}
// STRATA_PF_HFMA_SASS_ONLY=d_n10240,h_up (diagnostic): only the kernels whose symbol contains one of these comma-separated pieces
inline bool allowed(int k) {
    static const std::string only = [] { const char* e = std::getenv("STRATA_PF_HFMA_SASS_ONLY"); return std::string(e != nullptr ? e : ""); }();
    if (only.empty()) return true;
    size_t b = 0;
    while (b <= only.size()) {
        size_t e = only.find(',', b);
        if (e == std::string::npos) e = only.size();
        if (e > b && std::string(kKern[k].sym).find(only.substr(b, e - b)) != std::string::npos) return true;
        b = e + 1;
    }
    return false;
}
inline int find(int N, int K, bool hc) {
    for (int k = 0; k < kKernCount; ++k)
        if (kKern[k].N == N && kKern[k].K == K && ((k >= kHcDown) == hc)) return allowed(k) ? k : -2;
    return -1;
}

}  // namespace detail

// dense Y[T, N] (FP32, row stride ldy) = X[T, K] . W[N, K]^T, as gemm_hfma.cuh's gemm().  nullptr: launched on `st`; else why not (the caller
// runs the compiler kernel).  `kern` gets the kernel's index when launched.
inline const char* dense(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, cudaStream_t st,
                         int dev, int* kern) {
    const int k = detail::find((int) N, (int) K, false);
    if (k == -2) return "filtered out (STRATA_PF_HFMA_SASS_ONLY)";
    if (k < 0) return "shape not in the generated set";
    if (T < 128) return "T below 128 rows";
    if (ldx != K) return "X row stride is not K";
    if (ldy != N) return "Y row stride is not N";
    if (((uintptr_t) X & 15) != 0 || ((uintptr_t) W & 15) != 0 || ((uintptr_t) Y & 3) != 0) return "misaligned operand";
    if ((T + 128) * 2 * K >= (int64_t) 1 << 32 || (T + 128) * 4 * N >= (int64_t) 1 << 32) return "offsets past 2^32 bytes";
    if (dev < 0 || dev >= 64) return "device index";
    const detail::Dev* d = detail::dev_module(dev);
    if (d == nullptr) return "module not loaded";
    detail::ArgsDense a{(CUdeviceptr) X, (CUdeviceptr) W, (CUdeviceptr) Y, (int) (T - 128)};
    if (detail::launch(d->fn[k], (unsigned) ((N + 127) / 128), (unsigned) ((T + 127) / 128), 1, a, st) != CUDA_SUCCESS) return "launch failed";
    detail::check("dense", k, T, N, K, X, W, Y, st, dev);
    *kern = k;
    return nullptr;
}

// the hyper-connection slice's product: rows x K FP16 image `x16` (rows of K, contiguous) times the weight's FP16 copy `w16`, times the scale in
// `ysc` (a device float, written by ysc_kernel after the slice's cvt_kernel), into `y` (Z == 1; row stride N) or the Z split-K parts `part` + z * rows * N (Z > 1;
// reduce_kernel sums them).  `bad`: where a part's count of non-finite outputs goes (Z == 1: g_hc.bad_y; Z > 1: a sink).  down: Z as the compiler path's
// partition table (s0/R by the same integer formula); up: Z must be 1.  nullptr: launched; else why not.
inline const char* hc_slice(bool down, const uint16_t* x16, const uint16_t* w16, float* y, float* part, int64_t rows, int64_t N, int64_t K, int64_t ldy, int Z,
                            float* ysc, unsigned* bad, cudaStream_t st, int dev) {
    const int k = detail::find((int) N, (int) K, true);
    if (k == -2) return "filtered out (STRATA_PF_HFMA_SASS_ONLY)";
    if (k < 0 || (k == kHcDown) != down) return "shape not in the generated set";
    if (rows < 128) return "slice below 128 rows";
    if (ldy != N) return "Y row stride is not N";
    if (!down && Z != 1) return "up split";
    if (Z < 1 || Z > 8 || (Z > 1 && part == nullptr)) return "split";
    if (((uintptr_t) x16 & 15) != 0 || ((uintptr_t) w16 & 15) != 0 || ((uintptr_t) y & 3) != 0) return "misaligned operand";
    if (dev < 0 || dev >= 64) return "device index";
    const detail::Dev* d = detail::dev_module(dev);
    if (d == nullptr) return "module not loaded";
    const unsigned gx = (unsigned) ((N + 127) / 128), gy = (unsigned) ((rows + 127) / 128);
    if (!down) {
        detail::ArgsHc a{(CUdeviceptr) x16, (CUdeviceptr) w16, (CUdeviceptr) y, (CUdeviceptr) ysc, (CUdeviceptr) bad, (int) (rows - 128)};
        if (detail::launch(d->fn[k], gx, gy, 1, a, st) != CUDA_SUCCESS) return "launch failed";
        detail::check("hc up", k, rows, N, K, x16, w16, y, st, dev);
        return nullptr;
    }
    detail::ArgsHcz a{};
    a.X = (CUdeviceptr) x16; a.W = (CUdeviceptr) w16; a.Y = (CUdeviceptr) (Z > 1 ? (void*) part : (void*) y);
    a.ysc = (CUdeviceptr) ysc; a.bad = (CUdeviceptr) bad; a.Tm128 = (int) (rows - 128);
    const int S = (int) (K / 32), G = (S + 3) / 4;
    for (int z = 0; z < Z; ++z) {
        const int s0 = (int) ((int64_t) G * z / Z) * 4;
        const int s1 = S < (int) ((int64_t) G * (z + 1) / Z) * 4 ? S : (int) ((int64_t) G * (z + 1) / Z) * 4;
        a.tab[3 * z] = (unsigned) s0 * 64u;                                  // first stage, in bytes along the rows (64 B a stage)
        a.tab[3 * z + 1] = (unsigned) (s1 - s0);                             // the part's stages
        a.tab[3 * z + 2] = (unsigned) ((int64_t) z * rows * N * 4);          // the part's Y offset (floats * 4)
    }
    if (detail::launch(d->fn[k], gx, gy, (unsigned) Z, a, st) != CUDA_SUCCESS) return "launch failed";
    detail::check("hc down", k, rows, N, K, x16, w16, Z > 1 ? (void*) part : (void*) y, st, dev);
    return nullptr;
}

}  // namespace strata::prefill::sass
