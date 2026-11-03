// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
#include "strata/prefill/moe_mmq.hpp"
#endif

#ifdef STRATA_USE_HIP
#include "wmma_gemm.h"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
// The HIP compatibility shim maps CUDA shuffle spellings to Strata helpers.
// hipBLASLt's public headers declare native HIP shuffle functions, so keep
// those declarations from being macro-expanded in this translation unit.
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#endif

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
#include "gemm_hfma.cuh"   // STRATA_PF_HFMA
#include "gemm_hfma_hc.cuh"   // STRATA_PF_HFMA_HC
#include "gemm_sass.cuh"   // STRATA_PF_HFMA_SASS
#include "strata/prefill/pf_small.hpp"   // STRATA_PF_HFMA_SMALL
#include "strata/prefill/pf_small2.hpp"   // STRATA_PF_ROUTER_V2
#include "strata/kernels/ktime.hpp"   // STRATA_PREFILL_TIMING=2: the hyper-connection products' kernels
#include <cctype>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#endif

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
#include "hipblaslt_tuning.hpp"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <set>
#include <tuple>
#endif
#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
#include "rocblas_tuning.hpp"
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <filesystem>
#include <map>
#include <string>
#include <system_error>
#include <tuple>
#endif

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#endif

namespace strata::prefill {
namespace {
#if defined(__HIPCC__)
// STRATA_F16_RANGE (gemm_set_f16_range): the largest |value| and the count beyond 65504 / non-finite, for the BF16
// weights converted to FP16 (slot 0, bf16_to_f16_rows) and the FP16 GEMM outputs widened (slot 1, widen_rows_f16 -
// an output rocBLAS wrote as FP16 is Inf there, not saturated).  As the activation counter in kernels.cu.
__device__ int g_f16_range_on_g = 0;
__device__ unsigned g_f16_range_max_g[2] = {0, 0};
__device__ unsigned long long g_f16_range_over_g[2] = {0, 0};
__device__ __forceinline__ void f16_range_note_g(int k, float f) {
    const float a = fabsf(f);
    const unsigned bits = __float_as_uint(a);
    if (bits > g_f16_range_max_g[k]) atomicMax(&g_f16_range_max_g[k], bits);
    if (a > 65504.0f || isnan(f)) atomicAdd(&g_f16_range_over_g[k], 1ull);
}
// Y's rows, written by an FP16-out GEMM as FP16 at the start of each FP32 row (ldc = 2 ldy halves), widened in place.
// Float c overwrites halves 2c and 2c+1, so a row is walked from its end in blocks: a block's halves are read into
// registers, the block syncs, then writes its floats - which only cover halves of blocks already read.
// STRATA_DBG_NAN: the FP16 outputs that are not finite (an FP32-accumulated sum past 65504 becomes inf in FP16).
__device__ unsigned long long g_f16_nonfinite = 0;
__global__ void widen_rows_f16(float* __restrict__ Y, int64_t n, int64_t ldy, int count_nonfinite) {
    float* y = Y + (int64_t) blockIdx.x * ldy;
    const __half* h = reinterpret_cast<const __half*>(y);
    const int64_t nb = (n + blockDim.x - 1) / blockDim.x;
    for (int64_t b = nb - 1; b >= 0; --b) {
        const int64_t c = b * blockDim.x + threadIdx.x;
        const float v = c < n ? __half2float(h[c]) : 0.0f;
        if (count_nonfinite && c < n && !isfinite(v)) atomicAdd(&g_f16_nonfinite, 1ull);
        if (c < n && g_f16_range_on_g) f16_range_note_g(1, v);
        __syncthreads();
        if (c < n) y[c] = v;
        __syncthreads();
    }
}
// BF16 weight rows -> FP16, saturated (bench/results/2026-10-04-rdna2-fp16-prompt: none leaves FP16's range)
__global__ void bf16_to_f16_rows(const uint16_t* __restrict__ s, __half* __restrict__ d, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float f = __uint_as_float((uint32_t) s[i] << 16);
        if (g_f16_range_on_g) f16_range_note_g(0, f);
        d[i] = __float2half(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f));   // as hf_sat: a NaN stays NaN
    }
}
#endif
}  // namespace

void gemm_set_f16_range(bool on) {
#if defined(__HIPCC__)
    const int v = on ? 1 : 0;
    if (cudaMemcpyToSymbol(g_f16_range_on_g, &v, sizeof v) != cudaSuccess) {
        std::fprintf(stderr, "prefill gemm: setting the FP16 range check failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
#else
    (void) on;
#endif
}
bool gemm_f16_range_read(float max_abs[2], unsigned long long over[2], bool reset) {
#if defined(__HIPCC__)
    unsigned bits[2] = {0, 0}; unsigned long long o[2] = {0, 0};
    if (cudaMemcpyFromSymbol(bits, g_f16_range_max_g, sizeof bits) != cudaSuccess ||
        cudaMemcpyFromSymbol(o, g_f16_range_over_g, sizeof o) != cudaSuccess) { cudaGetLastError(); return false; }
    for (int k = 0; k < 2; ++k) { union { unsigned u; float f; } c; c.u = bits[k]; max_abs[k] = c.f; over[k] = o[k]; }
    if (reset) {
        const unsigned z[2] = {0, 0}; const unsigned long long zz[2] = {0, 0};
        cudaMemcpyToSymbol(g_f16_range_max_g, z, sizeof z); cudaMemcpyToSymbol(g_f16_range_over_g, zz, sizeof zz);
    }
    return true;
#else
    (void) max_abs; (void) over; (void) reset;
    return false;
#endif
}

namespace {

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

// #247/#325: on Windows (seen on gfx1201), hipBLAS can return success with the correct BF16/FP16 product for some
// shapes (hc up once T >= 96, the router) and still leave hipErrorInvalidValue set, which the next kernel's error
// check turns into an exit. The multiply has finished, so that one stale error is cleared after a GEMM that succeeded;
// any other error still stops the engine. Windows only: on Linux a stale hipErrorInvalidValue is a real error from
// an earlier call and keeps being reported. A no-op everywhere else (CUDA compiles none of it).
#if defined(__HIPCC__) && defined(_WIN32)
void absorb_hipblas_sticky(const char* what) {
    const hipError_t sticky = hipGetLastError();
    if (sticky == hipSuccess || sticky == hipErrorInvalidValue) return;
    std::fprintf(stderr, "prefill gemm: %s left %s\n", what, hipGetErrorString(sticky));
    std::exit(1);
}
#define STRATA_ABSORB_HIPBLAS_STICKY(what) absorb_hipblas_sticky(what)
#else
#define STRATA_ABSORB_HIPBLAS_STICKY(what) ((void) 0)
#endif

// A setup call whose failure the engine survives (the handle keeps its defaults), as before #240 - but said.
void note(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d (continuing)\n", what, (int) s);
}

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
struct HipLtCallKey {
    strata::prefill::hipblaslt::InputType type;
    int t;
    int n;
    int k;
    int ldy;
    uint32_t beta_bits;

    bool operator<(const HipLtCallKey& other) const {
        return std::tie(type, n, k, ldy, t, beta_bits) <
               std::tie(other.type, other.n, other.k, other.ldy, other.t, other.beta_bits);
    }
};

struct HipLtCachedAlgo {
    bool supported = false;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
};

struct HipLtState {
    hipblasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    strata::prefill::hipblaslt::TuningTable table;
    std::map<HipLtCallKey, HipLtCachedAlgo> cache;
    uint64_t lt_launches = 0;
    uint64_t fallbacks = 0;
    std::set<std::tuple<strata::prefill::hipblaslt::InputType, int, int, int, int>> fallback_shapes;

    ~HipLtState() {
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: hipBLASLt summary launches=%llu fallbacks=%llu unique_fallback_shapes=%zu\n",
                         (unsigned long long) lt_launches, (unsigned long long) fallbacks, fallback_shapes.size());
            for (const auto& shape : fallback_shapes) {
                const auto type = std::get<0>(shape);
                std::fprintf(stderr, "prefill gemm: fallback shape dtype=%s T=%d N=%d K=%d ldy=%d\n",
                             type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16",
                             std::get<1>(shape), std::get<2>(shape), std::get<3>(shape), std::get<4>(shape));
            }
        }
        if (handle) hipblasLtDestroy(handle);
    }
};

struct HipLtDescriptors {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;

    ~HipLtDescriptors() {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }

    bool init(hipDataType type, int t, int n, int k, int ldy, int ldb = 0) {
        const hipblasOperation_t trans_a = HIPBLAS_OP_T;
        const hipblasOperation_t trans_b = HIPBLAS_OP_N;
        if (hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&a, type, k, n, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&b, type, k, t, ldb > k ? ldb : k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, n, t, ldy) != HIPBLAS_STATUS_SUCCESS) {
            return false;
        }
        return true;
    }
};

// The engine package's own tuning folder (<engine folder>/tuning), for a start that did not come through the launcher (a hand-made
// config, a CLI run): without a table the prompt GEMMs fall back to plain hipBLAS and a Strix Halo reads prompts at half speed
// (340 against 690 tok/s, IQ3_S 4K).  The launcher sets STRATA_HIPBLASLT_TUNING itself; this is the same table found by the engine.
std::string own_tuning_dir() {
#if defined(_WIN32)
    char p[4096] = {};
    const DWORD n = GetModuleFileNameA(nullptr, p, (DWORD) sizeof(p));
    std::string s(p, n);
    const size_t k = s.find_last_of("\\/");
    return k == std::string::npos ? std::string() : s.substr(0, k + 1) + "tuning/";
#else
    char p[4096] = {};
    const ssize_t n = readlink("/proc/self/exe", p, sizeof(p) - 1);
    std::string s(p, n > 0 ? (size_t) n : 0);
    const size_t k = s.find_last_of('/');
    return k == std::string::npos ? std::string() : s.substr(0, k + 1) + "tuning/";
#endif
}

std::unique_ptr<HipLtState> create_hipblaslt_state(void* workspace, size_t workspace_bytes) {
    std::string path_own;
    const char* path = std::getenv("STRATA_HIPBLASLT_TUNING");
    bool from_package = false;
    if (!path || !*path) {
        const char* sw = std::getenv("STRATA_HIPBLASLT_TABLE");     // the launcher's switch: off / 0 / false = never the packaged table
        if (sw != nullptr && (std::strcmp(sw, "off") == 0 || std::strcmp(sw, "0") == 0 || std::strcmp(sw, "false") == 0)) return nullptr;
        from_package = true;                                          // the file is found below, once the card and the version are known
    }

    auto state = std::make_unique<HipLtState>();
    state->workspace = workspace;
    state->workspace_bytes = workspace_bytes;
    if (hipblasLtCreate(&state->handle) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt handle creation failed; using hipBLASEx\n");
        return nullptr;
    }

    int version = 0;
    if (hipblasLtGetVersion(state->handle, &version) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt version query failed; using hipBLASEx\n");
        return nullptr;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr, "prefill gemm: HIP device query failed; using hipBLASEx\n");
        return nullptr;
    }
    std::string arch(properties.gcnArchName);
    const auto suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);

    if (from_package) {
        const std::string dir = own_tuning_dir();
        if (dir.empty()) return nullptr;
        path_own = dir + arch + "-hipblaslt-" + std::to_string(version) + ".txt";
        std::FILE* f = std::fopen(path_own.c_str(), "rb");
        if (f == nullptr) return nullptr;       // no table for this card and hipBLASLt version beside the engine: plain hipBLAS as before
        std::fclose(f);
        path = path_own.c_str();
    }
    std::string error;
    if (!state->table.load(path, arch, version, error)) {
        std::fprintf(stderr, "prefill gemm: %s; using hipBLASEx\n", error.c_str());
        return nullptr;
    }
    std::fprintf(stderr, "prefill gemm: hipBLASLt tuning enabled (%zu rows, %s, version %d)\n",
                 state->table.rows().size(), arch.c_str(), version);
    return state;
}

HipLtCachedAlgo resolve_hipblaslt_algo(HipLtState& state, strata::prefill::hipblaslt::InputType type, int t,
                                       int n, int k, int ldy, float beta) {
    uint32_t beta_bits = 0;
    static_assert(sizeof(beta_bits) == sizeof(beta));
    std::memcpy(&beta_bits, &beta, sizeof(beta));
    const HipLtCallKey key{type, t, n, k, ldy, beta_bits};
    const auto cached = state.cache.find(key);
    if (cached != state.cache.end()) return cached->second;

    HipLtCachedAlgo resolved;
    const bool verbose = std::getenv("STRATA_HIPBLASLT_VERBOSE") != nullptr;
    const auto* row = state.table.closest(type, n, k, ldy, t);
    if (!row) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; no calibration for dtype=%s T=%d N=%d K=%d ldy=%d\n",
                         type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, t, n, k, ldy)) {
        return state.cache.emplace(key, resolved).first->second;
    }

    std::vector<int> solution_ids{row->solution_id};
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    if (hipblaslt_ext::getAlgosFromIndex(state.handle, solution_ids, candidates) != HIPBLAS_STATUS_SUCCESS ||
        candidates.empty() || candidates.front().state != HIPBLAS_STATUS_SUCCESS ||
        hipblaslt_ext::getIndexFromAlgo(candidates.front().algo) != row->solution_id) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d unavailable for T=%d N=%d K=%d ldy=%d\n",
                         row->solution_id, t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    const float alpha = 1.0f;
    size_t required_workspace = 0;
    auto algo = candidates.front().algo;
    if (hipblaslt_ext::matmulIsAlgoSupported(state.handle, desc.op, &alpha, desc.a, desc.b, &beta, desc.c, desc.c,
                                             algo, required_workspace) != HIPBLAS_STATUS_SUCCESS) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d rejects actual T=%d N=%d K=%d ldy=%d beta=%.9g\n",
                         row->solution_id, t, n, k, ldy, beta);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    resolved.supported = true;
    resolved.algo = algo;
    resolved.workspace_bytes = required_workspace;
    if (verbose) {
        std::fprintf(stderr,
                     "prefill gemm: Lt solution=%d dtype=%s T=%d N=%d K=%d ldy=%d beta=%.9g workspace=%zu\n",
                     row->solution_id, type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n,
                     k, ldy, beta, required_workspace);
    }
    return state.cache.emplace(key, resolved).first->second;
}

// the hipBLASLt solution index the tuning table gives this call (-1: none / unsupported)
int hipblaslt_solution_for(void* opaque_state, strata::prefill::hipblaslt::InputType type, int64_t t, int64_t n, int64_t k,
                           int64_t ldy, float beta) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX || ldy < n) return -1;
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported || resolved.workspace_bytes > state->workspace_bytes) return -1;
    auto algo = resolved.algo;
    return hipblaslt_ext::getIndexFromAlgo(algo);
}

bool try_hipblaslt(void* opaque_state, strata::prefill::hipblaslt::InputType type, const uint16_t* x,
                   const uint16_t* w, float* y, int64_t t, int64_t n, int64_t k, int64_t ldy, float beta,
                   void* stream, int64_t ldx = 0) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX ||
        ldy < n) {
        return false;
    }
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        return false;
    }
    if (resolved.workspace_bytes > state->workspace_bytes) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution needs %zu workspace bytes, have %zu\n",
                         resolved.workspace_bytes, state->workspace_bytes);
        }
        return false;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, (int) t, (int) n, (int) k, (int) ldy, (int) ldx)) return false;
    const float alpha = 1.0f;
    const hipblasStatus_t status = hipblasLtMatmul(state->handle, desc.op, &alpha, w, desc.a, x, desc.b, &beta, y,
                                                   desc.c, y, desc.c, &resolved.algo, state->workspace,
                                                   state->workspace_bytes, (hipStream_t) stream);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        ++state->lt_launches;
        return true;
    }

    std::fprintf(stderr, "prefill gemm: hipBLASLt launch failed with status %d\n", (int) status);
    if (beta != 0.0f) {
        std::fprintf(stderr, "prefill gemm: refusing a fallback after hipBLASLt failed with nonzero beta\n");
        std::exit(1);
    }
    auto* mutable_state = static_cast<HipLtState*>(opaque_state);
    uint32_t beta_bits = 0;
    std::memcpy(&beta_bits, &beta, sizeof(beta_bits));
    auto cached = mutable_state->cache.find(HipLtCallKey{type, (int) t, (int) n, (int) k, (int) ldy, beta_bits});
    if (cached != mutable_state->cache.end()) cached->second.supported = false;
    ++mutable_state->fallbacks;
    mutable_state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
    return false;
}
#endif

#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
// The rocBLAS solution table of f16_inplace (STRATA_ROCBLAS_TUNING: a table file, or a directory holding
// <arch>-rocblas-<version>.txt for this card and rocBLAS build; src/prefill/rocblas_tuning.hpp).  rocBLAS's own
// pick for some shapes is up to 6x slower than the best kernel it holds for them (gfx1030: tools/hip/tune_rocblas.cpp).
struct RocBlasState {
    rocblas_handle handle = nullptr;
    strata::prefill::rocblas_tuning::Table table;
    struct Pick { bool tuned = false; int solution = 0; int refusals = 0; };
    std::map<std::tuple<int, int, int, int>, Pick> cache;   // (T, N, K, ldy) -> the solution to run, or the default
    uint64_t tuned_launches = 0;
    uint64_t fallbacks = 0;

    ~RocBlasState() {
        if (std::getenv("STRATA_ROCBLAS_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: rocBLAS tuning summary tuned_launches=%llu fallbacks=%llu\n",
                         (unsigned long long) tuned_launches, (unsigned long long) fallbacks);
        }
        if (handle) rocblas_destroy_handle(handle);
    }
};

std::unique_ptr<RocBlasState> create_rocblas_state(void* stream) {
    const char* path = std::getenv("STRATA_ROCBLAS_TUNING");
    if (!path || !*path) return nullptr;
    auto state = std::make_unique<RocBlasState>();
    if (rocblas_create_handle(&state->handle) != rocblas_status_success) {
        std::fprintf(stderr, "prefill gemm: rocBLAS handle creation failed; its default kernels run\n");
        return nullptr;
    }
    rocblas_set_stream(state->handle, (hipStream_t) stream);
    char version[128] = {0};
    if (rocblas_get_version_string(version, sizeof version) != rocblas_status_success) {
        std::fprintf(stderr, "prefill gemm: rocBLAS version query failed; its default kernels run\n");
        return nullptr;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr, "prefill gemm: HIP device query failed; rocBLAS's default kernels run\n");
        return nullptr;
    }
    std::string arch(properties.gcnArchName);
    if (const auto suffix = arch.find(':'); suffix != std::string::npos) arch.resize(suffix);
    std::filesystem::path file(path);
    std::error_code ec;
    if (std::filesystem::is_directory(file, ec)) {
        // the table of this card and this rocBLAS build (its version string, trailing dots dropped)
        std::string tag(version);
        while (!tag.empty() && tag.back() == '.') tag.pop_back();
        file /= arch + "-rocblas-" + tag + ".txt";
    }
    std::string error;
    if (!state->table.load(file.string(), arch, version, error)) {
        std::fprintf(stderr, "prefill gemm: %s (%s); rocBLAS's default kernels run\n", error.c_str(),
                     file.string().c_str());
        return nullptr;
    }
    // rocBLAS loads its kernel library on the first product through a handle; a solution-index call before that
    // fails (rocblas_status_internal_error) although the index is valid.  One tiny default product opens it.
    {
        uint16_t* tiny = nullptr;
        if (hipMalloc(&tiny, 4 * 64 * 64 * 2) == hipSuccess) {
            const float one = 1.0f, zero = 0.0f;
            (void) rocblas_gemm_ex(state->handle, rocblas_operation_transpose, rocblas_operation_none, 64, 64, 64, &one,
                                   tiny, rocblas_datatype_f16_r, 64, tiny + 64 * 64, rocblas_datatype_f16_r, 64, &zero,
                                   tiny + 2 * 64 * 64, rocblas_datatype_f16_r, 128, tiny + 2 * 64 * 64,
                                   rocblas_datatype_f16_r, 128, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
            (void) hipStreamSynchronize((hipStream_t) stream);
            (void) hipFree(tiny);
        }
    }
    size_t tuned_rows = 0;
    for (const auto& row : state->table.rows()) tuned_rows += row.tuned ? 1 : 0;
    std::fprintf(stderr, "prefill gemm: rocBLAS tuning enabled (%zu rows, %zu with a solution, %s, rocBLAS %s)\n",
                 state->table.rows().size(), tuned_rows, arch.c_str(), version);
    return state;
}

// Gemm::f16_inplace's product through the table's solution for its shape.  False when the shape has no row, or the
// launch was refused (then the default kernel runs, and for that shape from now on).
bool try_rocblas_f16o(void* opaque_state, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N,
                      int64_t K, int64_t ldy) {
    auto* state = static_cast<RocBlasState*>(opaque_state);
    if (!state || T <= 0 || N <= 0 || K <= 0 || T > INT_MAX || N > INT_MAX || K > INT_MAX || ldy > INT_MAX / 2) return false;
    const auto key = std::make_tuple((int) T, (int) N, (int) K, (int) ldy);
    auto it = state->cache.find(key);
    if (it == state->cache.end()) {
        RocBlasState::Pick pick;
        if (const auto* row = state->table.closest((int) N, (int) K, (int) ldy, (int) T); row && row->tuned) {
            pick.tuned = true;
            pick.solution = row->solution;
        }
        if (std::getenv("STRATA_ROCBLAS_VERBOSE")) {
            if (pick.tuned)
                std::fprintf(stderr, "prefill gemm: rocBLAS solution %d for T=%lld N=%lld K=%lld ldy=%lld\n",
                             pick.solution, (long long) T, (long long) N, (long long) K, (long long) ldy);
            else
                std::fprintf(stderr, "prefill gemm: rocBLAS default kernel; no row for T=%lld N=%lld K=%lld ldy=%lld\n",
                             (long long) T, (long long) N, (long long) K, (long long) ldy);
        }
        it = state->cache.emplace(key, pick).first;
    }
    if (!it->second.tuned) return false;
    const float one = 1.0f, zero = 0.0f;
    const rocblas_status status = rocblas_gemm_ex(
        state->handle, rocblas_operation_transpose, rocblas_operation_none, (int) N, (int) T, (int) K, &one, W,
        rocblas_datatype_f16_r, (int) K, X, rocblas_datatype_f16_r, (int) K, &zero, Y, rocblas_datatype_f16_r,
        (int) (2 * ldy), Y, rocblas_datatype_f16_r, (int) (2 * ldy), rocblas_datatype_f32_r,
        rocblas_gemm_algo_solution_index, it->second.solution, 0);
    if (status == rocblas_status_success) {
        ++state->tuned_launches;
        return true;
    }
    // A refusal writes nothing (the default kernel runs now).  The first one can be rocBLAS still setting up for the
    // kernel (its device workspace: seen as invalid_value once, success after); a solution refused three times is
    // dropped for this shape.
    ++state->fallbacks;
    if (++it->second.refusals >= 3) {
        std::fprintf(stderr, "prefill gemm: rocBLAS solution %d refused (status %d) for T=%lld N=%lld K=%lld ldy=%lld "
                             "three times; the default kernel runs for this shape\n", it->second.solution, (int) status,
                     (long long) T, (long long) N, (long long) K, (long long) ldy);
        it->second.tuned = false;
    } else if (std::getenv("STRATA_ROCBLAS_VERBOSE")) {
        std::fprintf(stderr, "prefill gemm: rocBLAS solution %d refused (status %d) for T=%lld N=%lld K=%lld ldy=%lld; "
                             "the default kernel this time\n", it->second.solution, (int) status, (long long) T,
                     (long long) N, (long long) K, (long long) ldy);
    }
    return false;
}
#endif

}  // namespace

bool prompt_f16() {
#if defined(__HIPCC__)
    constexpr int kMaxDev = 64;
    static std::atomic<int8_t> cached[kMaxDev] = {};   // 0 unknown, 1 off, 2 on
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= kMaxDev) { (void) hipGetLastError(); return false; }
    if (const int8_t c = cached[dev].load(std::memory_order_relaxed)) return c == 2;
    // Opt-in (#835): only STRATA_HIP_PROMPT_F16=1 turns it on.  It changes the prompt path's numbers (an FP16 rounding of
    // each 16-bit GEMM's output), so no card gets it unasked.  On gfx103x it is ~2x faster, so we say so once.
    const char* e = std::getenv("STRATA_HIP_PROMPT_F16");
    const bool on = e != nullptr && e[0] == '1';
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, dev) != hipSuccess) { (void) hipGetLastError(); p.gcnArchName[0] = 0; }
    if (on) std::fprintf(stderr, "strata prefill: HIP device %d%s%s - STRATA_HIP_PROMPT_F16=1: the prompt's 16-bit GEMMs run FP16 in and out "
                                 "(rocBLAS is tuned only for that on gfx103x; not bit-identical to the default)%s",
                         dev, p.gcnArchName[0] ? " " : "", p.gcnArchName, "\n");
    else if (e == nullptr && std::strncmp(p.gcnArchName, "gfx103", 6) == 0)
        std::fprintf(stderr, "strata prefill: HIP device %d %s - tip: STRATA_HIP_PROMPT_F16=1 reads long prompts about 2x faster here "
                             "(FP16 prompt GEMMs, rocBLAS is tuned only for FP16 on gfx103x; the numbers differ slightly, see docs/AMD_HIP.md)%s",
                     dev, p.gcnArchName, "\n");
    cached[dev].store(on ? 2 : 1, std::memory_order_relaxed);
    return on;
#else
    return false;
#endif
}

// A padded X (ldx > K) no kernel took: its rows copied to a contiguous FP16 buffer of stride K.  Grown, never shrunk
// (one buffer per Gemm; a padded X only arrives from STRATA_PF_PAD's producers).  One 2-D copy: the same values, and
// the ordinary paths below then read them exactly as if the producer had written them unpadded.
const uint16_t* Gemm::unpad_x(const uint16_t* X, int64_t T, int64_t K, int64_t ldx) {
    if (X == nullptr || T <= 0 || K <= 0 || ldx <= K) return nullptr;
    const int64_t need = T * K;
    if (need > unpadded_x_elems_) {
        if (unpadded_x_ != nullptr) { cudaFree(unpadded_x_); unpadded_x_ = nullptr; unpadded_x_elems_ = 0; }
        void* p = nullptr;
        if (cudaMalloc(&p, (size_t) need * sizeof(uint16_t)) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        unpadded_x_ = (uint16_t*) p;
        unpadded_x_elems_ = need;
    }
    if (cudaMemcpy2DAsync(unpadded_x_, (size_t) K * sizeof(uint16_t), X, (size_t) ldx * sizeof(uint16_t),
                          (size_t) K * sizeof(uint16_t), (size_t) T, cudaMemcpyDeviceToDevice,
                          (cudaStream_t) stream_) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return unpadded_x_;
}

bool gemm_dequant_side() {
#if defined(__HIPCC__)
    return false;
#else
    static const bool on = [] { const char* v = std::getenv("STRATA_DEQUANT_SIDE"); return v && v[0] == '1'; }();
    return on;
#endif
}

Gemm::~Gemm() {
    for (void* e : side_ready_)
        if (e) (void) cudaEventDestroy((cudaEvent_t) e);
    for (void* e : side_free_)
        if (e) (void) cudaEventDestroy((cudaEvent_t) e);
    if (side_stream_) (void) cudaStreamDestroy((cudaStream_t) side_stream_);
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
    delete static_cast<strata::prefill::mmq::Context*>(mmq_ctx_);
    if (mmq_buf_) cudaFree(mmq_buf_);
#endif
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    delete static_cast<HipLtState*>(hipblaslt_state_);
#endif
#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
    delete static_cast<RocBlasState*>(rocblas_state_);
#endif
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (tc_w_) cudaFree(tc_w_);
    if (tc_x_) cudaFree(tc_x_);
    if (unpadded_x_) cudaFree(unpadded_x_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    workspace_ = workspace;
    note(cublasSetWorkspace(h, workspace_, ws_bytes), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws_bytes).release();
#endif
#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
    rocblas_state_ = create_rocblas_state(stream).release();
#endif
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    cublasSetWorkspace((cublasHandle_t) handle_, workspace_, ws_bytes);
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (hipblaslt_state_) {
        auto* state = static_cast<HipLtState*>(hipblaslt_state_);
        state->workspace = workspace_;
        state->workspace_bytes = ws_bytes;
    }
#endif
}

bool Gemm::set_dequant_side(uint16_t* scratch2, std::string& err) {
    side_scratch_ = scratch2;
    side_cur_ = 0;
    if (scratch2 == nullptr) return true;
    if (side_stream_ == nullptr) {   // once: the stream and the four events outlive every rebind
        cudaStream_t s = nullptr;
        if (const cudaError_t e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking); e != cudaSuccess) {
            err = std::string("prefill gemm: dequant side stream: ") + cudaGetErrorString(e);
            side_scratch_ = nullptr;
            return false;
        }
        side_stream_ = s;
        for (int i = 0; i < 2; ++i) {
            cudaEvent_t r = nullptr, f = nullptr;
            const bool made = cudaEventCreateWithFlags(&r, cudaEventDisableTiming) == cudaSuccess &&
                              cudaEventCreateWithFlags(&f, cudaEventDisableTiming) == cudaSuccess;
            side_ready_[i] = r;   // kept even on a failure: the destructor frees what was made
            side_free_[i] = f;
            if (!made) {
                err = "prefill gemm: dequant side events";
                side_scratch_ = nullptr;
                return false;
            }
        }
    }
    // both scratches are free once what the compute stream holds now is done (a rebind follows a stream sync, so
    // this is belt and braces; a wait on an event never recorded would pass anyway)
    for (int i = 0; i < 2; ++i)
        (void) cudaEventRecord((cudaEvent_t) side_free_[i], (cudaStream_t) stream_);
    return true;
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    // #240: every failure names the call and the real status, so "no VRAM" can be told from a broken install
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (const cudaError_t e = cudaMalloc(&workspace_, ws); e != cudaSuccess) {
        err = std::string("prefill gemm: workspace of 32 MiB: ") + cudaGetErrorString(e);
        return false;
    }
    note(cublasSetWorkspace(h, workspace_, ws), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws).release();
#endif
#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)
    rocblas_state_ = create_rocblas_state(stream).release();
#endif
    if (scratch_elems > 0) {
        if (const cudaError_t e = cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2); e != cudaSuccess) {
            err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB: " +
                  cudaGetErrorString(e);
            return false;
        }
    }
    scratch_elems_ = scratch_elems;
    return true;
}

#if !defined(__HIPCC__)
// ---- below sm_80: BF16 GEMMs without BF16 tensor cores (PR #655, #540, #395) ----------------------------------------
// Turing and Volta have no BF16 tensor cores: cublasGemmEx on CUDA_R_16BF inputs falls back to a SIMT fp32 kernel
// (magma_sgemmEx), a fifth of a 4K prompt on an RTX 2080 Ti (#655) and ~10% of a V100's prompt (#540).  BF16 -> FP16
// is exact for every value inside FP16's normal range (the 7-bit mantissa fits in 10 bits); weights and normalized
// activations sit there.  With the FP16 path, the weight is converted once per call and the activations in row
// slices, into the instance's own buffers, and the product runs as the FP16 GEMM (fp32 accumulate) on the tensor
// cores.  Finite values beyond FP16's range are clamped to +-65504 (#540) instead of becoming Inf.  A beta = 1
// product (STRATA_PREFILL_BF16X2's remainder, ~2^-9 of the original, deep in FP16's subnormal band) keeps cuBLAS.
//   Default: on for compute capability 7.0 - 7.4 (Volta: only the experimental STRATA_EXPERIMENTAL_SM60 build runs
//   there), OFF for 7.5 (RTX 20 in the ready-made engine: the sums round differently, so it is opt-in until it has
//   been gated).  STRATA_BF16_TC=1 / =0 turns it on / off on any 7.x card (=2, a test mode: on any card).
// Pascal (6.x) has no tensor cores and cuBLAS has no BF16 GEMM for it (#395, measured NOT_SUPPORTED on a P40): both
// operands are widened to fp32 by an exact shift and the product is cublasSgemm with the same fp32 accumulator.
namespace {
__global__ void bf16_to_f16_kernel(const uint16_t* __restrict__ in, __half* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float f = __uint_as_float((uint32_t) in[i] << 16);
        if (f > 65504.0f && !isinf(f)) f = 65504.0f;
        else if (f < -65504.0f && !isinf(f)) f = -65504.0f;
        out[i] = __float2half_rn(f);
    }
}
void bf16_to_f16(const uint16_t* in, uint16_t* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    bf16_to_f16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, reinterpret_cast<__half*>(out), n);
}
#if defined(STRATA_EXPERIMENTAL_SM60)   // Pascal runs only the experimental build
__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float((uint32_t) in[i] << 16);
}
void bf16_to_f32(const uint16_t* in, float* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    bf16_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, out, n);
}
#endif
// The current device's compute capability as 10 * major + minor, per device (a layer split can mix cards); 0: not
// known, read as "not an old card" so a failed query keeps the cuBLAS BF16 call.
int current_cc() {
    static std::atomic<int> cc[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    int v = cc[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        v = 10 * major + minor;
        cc[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}
// 0: cuBLAS's BF16 GEMM, 1: through FP16 (tensor cores), 2: through FP32 (Pascal)
int bf16_path() {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_BF16_TC");
        return v != nullptr && v[0] != '\0' ? std::atoi(v) : -1;
    }();
    const int cc = current_cc();
    if (forced == 2 && cc > 0) return 1;   // a test mode: the FP16 path on any card (gemm_bf16_parity on sm_80+)
    if (cc <= 0 || cc >= 80) return 0;
    if (cc < 70) return 2;
    if (forced >= 0) return forced != 0 ? 1 : 0;
    return cc < 75 ? 1 : 0;
}
bool grow(uint16_t*& p, int64_t& have, int64_t want) {   // `have`, `want`: 2-byte elements
    if (have >= want) return true;
    if (p) cudaFree(p);
    p = nullptr;
    have = 0;
    if (cudaMalloc((void**) &p, (size_t) want * 2) != cudaSuccess) { cudaGetLastError(); p = nullptr; return false; }
    have = want;
    return true;
}
constexpr int64_t kXSliceElems = 16ll << 20;   // 32 MiB of FP16 activations per slice (64 MiB as fp32)

#if defined(STRATA_EXPERIMENTAL_SM60)
// ---- Pascal, opt-in (STRATA_DENSE_F32=1): the dense projections' FP16-operand GEMMs as cublasSgemm ----------------
// Measured on a Tesla P40 (sm_61) with cuBLAS 12.4.5.8 (bench/results/2026-10-12-p40-sm61-kernels).  For FP16
// operands with FP32 sums cuBLAS has two Maxwell kernels there, maxwell_sgemm_fp16_64x64 and _32x128, and picks the
// slow one for most chunk sizes: the 10240 x 2560 qkv projection runs at 7.6 TFLOPS for T 7030-7040 and at 4.3 from
// T 7041 (49 -> 85 ms), and at 4.3-4.6 for two thirds of the T between 256 and 8192; the 12288-row query projection
// likewise.  A 7K-token prompt reads in 16.3 or 19.5 s by its token count.  cublasSgemm on FP32 copies
// (maxwell_sgemm_128x64) runs at 8.4-9.8 TFLOPS for every T from 512 (6.7+ from 256).  The copies are exact
// (FP16 -> FP32), and for the shapes and sizes taken here the two kernels add every output's products in the same
// order: the result is bitwise the cuBLAS FP16 call's (gemm_f32_parity: 5,201 chunk sizes on a P40 with cuBLAS 12.4, a
// measurement of this library, not a guarantee of cuBLAS's; below the sizes taken here - T under
// ~144 at 6144+ rows, under ~408 at 2560 rows - cuBLAS switches one of them to another kernel family and the bits
// differ, so those keep the FP16 call).  The weight goes through the BF16 path's conversion buffer (126 MiB for the
// largest, 21 MiB more than that path's own), the activations through its slices.
// Opt-in, and on Pascal (compute capability 6.x) only: the equality belongs to the cuBLAS build, so run
// gemm_f32_parity with yours before turning it on; newer cards have tensor cores for the FP16 call.
__global__ void f16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(__ushort_as_half(in[i]));
}
void f16_to_f32(const uint16_t* in, float* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    f16_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, out, n);
}
int g_dense_f32 = -1;   // gemm_set_dense_f32 (the parity test: 0 / 1 on any card); -1 = STRATA_DENSE_F32
bool dense_f32_on() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_DENSE_F32");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (g_dense_f32 >= 0) return g_dense_f32 != 0;
    if (!env) return false;
    const int cc = current_cc();
    return cc >= 60 && cc < 70;
}
// The smallest T taken for a weight of N rows and K columns (0: never): the measured shapes only - the 2560-column
// qkv (10240 rows), query (12288) and gate (6144) projections and the 2560-row output projections (6144 and 4096
// columns), each from a T where the FP32 route (its conversions included) is the faster one or within ~7% of the FP16
// call (the gate projection below ~1300 tokens, the 4096-column output projection near 4096) and well above the T
// where the two stop agreeing bitwise.  Every other shape (a row slice of a weight too) keeps the FP16 call.
int64_t dense_f32_min_t(int64_t N, int64_t K) {
    if (K == 2560 && (N == 10240 || N == 12288)) return 256;
    if (K == 2560 && N == 6144) return 512;
    if (N == 2560 && (K == 6144 || K == 4096)) return 4096;
    return 0;
}
// ---- Pascal, opt-in (STRATA_DENSE_F32_RESIDENT=1): the FP32 copies made once and kept ---------------------------------
// STRATA_DENSE_F32 dequantizes each dense weight to FP16 and widens it to FP32 on every chunk (the qkv / query / gate
// projections of a 12-layer stage: ~470 M values, ~12 ms per chunk on a GP100 die), and the BF16 path (on Pascal every
// BF16 product: the hyper-connection projections, the routers, ...) widens its weight on every call.  With this switch
// the FP32 copy of such a weight is made on its first use - by the same kernels from the same values, so the same
// FP32 values - and kept for the life of the process, keyed by the weight's device address (resident weights only:
// the dense weights' native blocks, a pack's resident BF16 tensors; never a scratch).  The products are the same
// cublasSgemm calls with the same sizes and slices, so the bits are those of the per-call route.
//   Room: a copy is made only while the card keeps STRATA_F32_RESIDENT_RESERVE_MIB (default 1024) free after it; a
//   weight whose copy did not fit is not tried again and keeps its per-call route (a dense weight: the FP16 call, which
//   is bitwise the same; a BF16 weight: the per-call widening).  STRATA_F32_RESIDENT_BF16=0 leaves the BF16 weights out.
bool f32_resident_on() {
    static const bool env = [] {
        const char* v = std::getenv("STRATA_DENSE_F32_RESIDENT");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (!env) return false;
    const int cc = current_cc();
    return cc >= 60 && cc < 70;
}
bool f32_resident_bf16() {
    static const bool v = [] { const char* e = std::getenv("STRATA_F32_RESIDENT_BF16"); return e == nullptr || e[0] != '0'; }();
    return v && f32_resident_on();
}
size_t f32_resident_reserve() {
    static const size_t v = [] {
        const char* e = std::getenv("STRATA_F32_RESIDENT_RESERVE_MIB");
        const long long m = e != nullptr && e[0] != '\0' ? std::atoll(e) : 1024;
        return (size_t) (m < 0 ? 0 : m) << 20;
    }();
    return v;
}
std::mutex g_shadow_mu;
std::unordered_map<const void*, float*> g_shadow;   // weight address -> its FP32 copy (nullptr: did not fit, not retried)
std::unordered_map<const void*, float*> g_shadow_t;  // the same weights' transposed copies (STRATA_GEMM_ALGO6=2)
uint64_t g_shadow_bytes[64] = {};
int g_shadow_count[64] = {}, g_shadow_refused[64] = {};
// the FP32 copy of the weight at `key` (`elems` values), made on first use by `make(dst)` on `st`; nullptr: no copy
template <class Make>
const float* shadow_get_in(std::unordered_map<const void*, float*>& into, const void* key, int64_t elems, cudaStream_t st,
                           Make make, const char* what) {
    std::lock_guard<std::mutex> lk(g_shadow_mu);
    if (const auto it = into.find(key); it != into.end()) return it->second;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return nullptr; }
    const size_t bytes = (size_t) elems * 4;
    size_t free_b = 0, total_b = 0;
    float* p = nullptr;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b >= bytes + f32_resident_reserve() &&
        cudaMalloc((void**) &p, bytes) == cudaSuccess) {
        make(p);
        if (cudaStreamSynchronize(st) != cudaSuccess) {   // (the copy is read by later chunks on this stream only)
            std::fprintf(stderr, "strata prefill: %s: making a copy failed on GPU %d\n", what, dev);
            std::exit(1);
        }
        const uint64_t before = g_shadow_bytes[dev];
        g_shadow_bytes[dev] += bytes;
        ++g_shadow_count[dev];
        if (before == 0 || (before >> 28) != (g_shadow_bytes[dev] >> 28))   // the first, then every 256 MiB
            std::fprintf(stderr, "strata prefill: %s: GPU %d keeps %d FP32 weight copies, %.0f MiB "
                                 "(%.0f MiB free)\n", what, dev, g_shadow_count[dev], g_shadow_bytes[dev] / 1048576.0,
                         (free_b - bytes) / 1048576.0);
    } else {
        cudaGetLastError();
        p = nullptr;
        if (g_shadow_refused[dev]++ < 4)
            std::fprintf(stderr, "strata prefill: %s: GPU %d: no room for a %.1f MiB FP32 copy (%.0f MiB "
                                 "free, reserve %.0f MiB; %d copies, %.0f MiB kept): this weight keeps its per-call route\n",
                         what, dev, bytes / 1048576.0, free_b / 1048576.0, f32_resident_reserve() / 1048576.0,
                         g_shadow_count[dev], g_shadow_bytes[dev] / 1048576.0);
    }
    into.emplace(key, p);
    return p;
}
template <class Make>
const float* shadow_get(const void* key, int64_t elems, cudaStream_t st, Make make) {
    return shadow_get_in(g_shadow, key, elems, st, make, "STRATA_DENSE_F32_RESIDENT");
}

// ---- Pascal, opt-in (STRATA_GEMM_ALGO6=1 / 2): a fixed cuBLAS algorithm for the prompt GEMMs ---------------------------
// Measured on a GP100 die with cuBLAS 12.9 (the engine's own library; campaign dsw1 §2, dsw4 §1-2): for the
// FP16-operand products of the dense projections cuBLAS's DEFAULT algorithm is not its fastest kernel at most chunk
// sizes, and the same holds for the cublasSgemm call STRATA_DENSE_F32 makes on the FP32 copies (T 512/768 of the
// 10240-row weight: 4.7-5.6 TF/s of the 6.45 the die has).  CUBLAS_GEMM_ALGO6 reproduces the DEFAULT call's output byte
// for byte - an algorithm changes the tile, not the k direction's accumulation order - for the shapes and chunk sizes
// the tables below take (w4_algo6_verify: the whole output memcmp'd at every T from 128 to 8192 of the five dense
// shapes, both routes), and runs 3-16% faster.
//   =1  the FP16 products from f16_algo6_t, and the FP32 route's product as cublasGemmEx(OP_T, OP_N, ..., ALGO6)
//       (+2.7..3.4% of the dense GEMM time).  T outside the tables, and every other shape, keep their calls.
//   =2  as =1, and the FP32 route's weight is also kept transposed (column-major, next to the untransposed copy
//       F32RES keeps) so its product is cublasGemmEx(OP_N, OP_T, ..., ALGO6) on that copy and the per-slice transposed
//       X: +6.0..6.5% instead of +3%.  The transposed copy is budgeted like the F32RES copies
//       (STRATA_F32_RESIDENT_RESERVE_MIB); a copy that does not fit leaves that shape on the =1 route.
// Off by default and off outside compute capability 6.x: with the switch unset the calls are exactly f32res's.
int gemm_algo6_mode() {
    static const int m = [] {
        const char* v = std::getenv("STRATA_GEMM_ALGO6");
        const int n = v != nullptr && v[0] != '\0' ? std::atoi(v) : 0;
        const int cc = current_cc();
        return (cc >= 60 && cc < 70) ? std::min(2, std::max(0, n)) : 0;
    }();
    return m;
}
// The FP16-operand products whose ALGO6 call is the DEFAULT call's output and faster, by shape and T (the windows are
// measured, not guessed: at T 800-1024 and 1536 of the 2560-row output projections ALGO6 is the slower one, and below
// T 512 cuBLAS's DEFAULT there moves to a kernel family whose bits ALGO6 does not reproduce).
bool f16_algo6_t(int64_t N, int64_t K, int64_t T) {
    if (K == 2560 && (N == 10240 || N == 12288 || N == 6144)) return T >= 700 && T <= 8192;
    if (N == 2560 && (K == 6144 || K == 4096)) return (T >= 700 && T <= 800) || (T >= 1660 && T <= 1920);
    return false;
}
// The FP32 copies' products that ALGO6 reproduces and speeds up (both the OP_T/OP_N and the OP_N/OP_T form): the three
// 2560-column dense weights, from the chunk size at which STRATA_DENSE_F32's route takes them.
bool f32_algo6_t(int64_t N, int64_t K, int64_t T) {
    if (K != 2560) return false;
    if (N == 10240 || N == 12288) return T >= 256 && T <= 8192;
    if (N == 6144) return T >= 512 && T <= 8192;
    return false;
}
// FP16 row-major N x K -> FP32 column-major N x K (op(A) = N): dst[i + k*N] = src[i*K + k], 32 x 32 tiles
__global__ void f16_transpose_w_kernel(const uint16_t* __restrict__ src, float* __restrict__ dst, int64_t N, int64_t K) {
    __shared__ uint16_t tile[32][33];
    const int64_t i0 = (int64_t) blockIdx.y * 32, k0 = (int64_t) blockIdx.x * 32;
    const int64_t i = i0 + threadIdx.y, k = k0 + threadIdx.x;
    if (i < N && k < K) tile[threadIdx.y][threadIdx.x] = src[i * K + k];
    __syncthreads();
    const int64_t i2 = i0 + threadIdx.x, k2 = k0 + threadIdx.y;   // swapped: consecutive threads write consecutive i
    if (i2 < N && k2 < K) dst[i2 + k2 * N] = __half2float(__ushort_as_half(tile[threadIdx.x][threadIdx.y]));
}
// FP16 row-major n x K -> FP32 column-major n x K (op(B) = T): dst[t + k*n] = src[t*K + k]
__global__ void f16_transpose_x_kernel(const uint16_t* __restrict__ src, float* __restrict__ dst, int64_t n, int64_t K) {
    __shared__ uint16_t tile[32][33];
    const int64_t t0 = (int64_t) blockIdx.y * 32, k0 = (int64_t) blockIdx.x * 32;
    const int64_t t = t0 + threadIdx.y, k = k0 + threadIdx.x;
    if (t < n && k < K) tile[threadIdx.y][threadIdx.x] = src[t * K + k];
    __syncthreads();
    const int64_t t2 = t0 + threadIdx.x, k2 = k0 + threadIdx.y;
    if (t2 < n && k2 < K) dst[t2 + k2 * n] = __half2float(__ushort_as_half(tile[threadIdx.x][threadIdx.y]));
}
void f16_transpose_w(const uint16_t* src, float* dst, int64_t N, int64_t K, cudaStream_t st) {
    f16_transpose_w_kernel<<<dim3((unsigned) ((K + 31) / 32), (unsigned) ((N + 31) / 32)), dim3(32, 32), 0, st>>>(src, dst, N, K);
}
void f16_transpose_x(const uint16_t* src, float* dst, int64_t n, int64_t K, cudaStream_t st) {
    f16_transpose_x_kernel<<<dim3((unsigned) ((K + 31) / 32), (unsigned) ((n + 31) / 32)), dim3(32, 32), 0, st>>>(src, dst, n, K);
}
bool grow_f32(float*& p, int64_t& have, int64_t want) {   // `have`, `want`: 4-byte elements
    if (have >= want) return true;
    if (p) cudaFree(p);
    p = nullptr;
    have = 0;
    if (cudaMalloc((void**) &p, (size_t) want * 4) != cudaSuccess) { cudaGetLastError(); p = nullptr; return false; }
    have = want;
    return true;
}

// ---- GP100, opt-in (STRATA_PF_HFMA=1): the prompt's dense projections on FP16 HFMA2 -------------------------------------
// GP100 issues FP16 pairs (fma.rn.f16x2) at twice the FP32 rate and cuBLAS has no kernel that uses it here: its
// FP16-operand GEMMs sum in FP32 (the cublasSgemm route above is the fastest of them).  gemm_hfma.cuh multiplies in FP16
// pairs, sums 64 products a lane in FP16 and widens that into an FP32 accumulator; the weight's FP16 copy is kept resident
// - the FP16 values dequant_f16 gives the other routes, made once on first use - so there is no dequantization or widening
// per chunk at all.  The sums are rounded in FP16, so the outputs are not bitwise those of any FP32 route; decode does not
// run through here.
//   Compute capability 6.0 only (6.1 issues FP16 at 1/64 of the FP32 rate).  A call the kernel cannot take - a K that is not
// a multiple of 32, a misaligned or short-strided X, a weight whose FP16 copy did not fit (the copies are budgeted like the
// FP32 ones: STRATA_F32_RESIDENT_RESERVE_MIB stays free after each) - launches nothing, runs the route it replaces, and is
// counted by weight and reason; gemm_hfma_report prints the counts once per read.
std::unordered_map<const void*, uint16_t*> g_shadow_h;   // weight address -> its FP16 copy (nullptr: did not fit, not retried)
uint64_t g_shadow_h_bytes[64] = {};
int g_shadow_h_count[64] = {}, g_shadow_h_refused[64] = {};
// shadow_get_in's twin for the FP16 copies (the same lock and reserve, their own books)
template <class Make>
const uint16_t* shadow_get_h(const void* key, int64_t elems, cudaStream_t st, Make make) {
    std::lock_guard<std::mutex> lk(g_shadow_mu);
    if (const auto it = g_shadow_h.find(key); it != g_shadow_h.end()) return it->second;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return nullptr; }
    const size_t bytes = (size_t) elems * 2;
    size_t free_b = 0, total_b = 0;
    uint16_t* p = nullptr;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b >= bytes + f32_resident_reserve() &&
        cudaMalloc((void**) &p, bytes) == cudaSuccess) {
        make(p);
        if (cudaStreamSynchronize(st) != cudaSuccess) {
            std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA: making a copy failed on GPU %d\n", dev);
            std::exit(1);
        }
        const uint64_t before = g_shadow_h_bytes[dev];
        g_shadow_h_bytes[dev] += bytes;
        ++g_shadow_h_count[dev];
        if (before == 0 || (before >> 28) != (g_shadow_h_bytes[dev] >> 28))   // the first, then every 256 MiB
            std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA: GPU %d keeps %d FP16 weight copies, %.0f MiB "
                                 "(%.0f MiB free)\n", dev, g_shadow_h_count[dev], g_shadow_h_bytes[dev] / 1048576.0,
                         (free_b - bytes) / 1048576.0);
    } else {
        cudaGetLastError();
        p = nullptr;
        if (g_shadow_h_refused[dev]++ < 4)
            std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA: GPU %d: no room for a %.1f MiB FP16 copy (%.0f MiB "
                                 "free, reserve %.0f MiB; %d copies, %.0f MiB kept): this weight keeps its old route\n",
                         dev, bytes / 1048576.0, free_b / 1048576.0, f32_resident_reserve() / 1048576.0,
                         g_shadow_h_count[dev], g_shadow_h_bytes[dev] / 1048576.0);
    }
    g_shadow_h.emplace(key, p);
    return p;
}
// what the kernel ran and what it left to the old route, per device since the last report
std::atomic<int64_t> g_hf_taken[64];
std::mutex g_hf_mu;
std::map<std::string, int64_t> g_hf_left[64];   // "weight: reason" -> calls (the weight without its layer: blk.12.attn_qkv.weight -> attn_qkv.weight)
void hfma_left(int dev, const char* name, const char* why) {
    std::string w = name != nullptr ? name : "?";
    if (w.compare(0, 4, "blk.") == 0) {
        size_t i = 4;
        while (i < w.size() && std::isdigit((unsigned char) w[i])) ++i;
        if (i < w.size() && w[i] == '.') w.erase(0, i + 1);
    }
    std::lock_guard<std::mutex> lk(g_hf_mu);
    ++g_hf_left[dev][w + ": " + why];
}

// ---- STRATA_PF_ROUTER_F16=1 (gemm.hpp: router_f16): what the router ran on and what it left to the FP32 router, per device since the last report; with
// STRATA_PF_ROUTER_F16_CHECK=1 the device's counters of the ids that differ from the FP32 router's (rf_cmp_kernel), read and zeroed by the report
std::atomic<int64_t> g_rf_taken[64], g_rf_sass[64], g_rf_chk[64];
std::mutex g_rf_mu;
std::map<std::string, int64_t> g_rf_left[64];   // reasons the FP32 router ran
std::map<std::string, int64_t> g_rf_cc[64];     // reasons the compiler's HFMA2 kernel ran instead of the hand-scheduled one
__device__ unsigned long long g_rf_cnt[8];      // tokens, tokens with another id set, ids outside the FP32 router's set, tokens in another order only, tokens with another top-1, max |dw| (float bits), sum |dw| * 1e9
__global__ void rf_cmp_kernel(const int32_t* __restrict__ a, const int32_t* __restrict__ b, const float* __restrict__ wa, const float* __restrict__ wb, int64_t T) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    int diff = 0;
    bool same_order = true;
    float wd = 0.0f;
    for (int i = 0; i < 10; ++i) {
        const int x = a[t * 10 + i];
        bool found = false;
        for (int j = 0; j < 10; ++j) found = found || (b[t * 10 + j] == x);
        if (!found) ++diff;
        if (x != b[t * 10 + i]) same_order = false;
        wd = fmaxf(wd, fabsf(wa[t * 10 + i] - wb[t * 10 + i]));
    }
    atomicAdd(&g_rf_cnt[0], 1ull);
    if (diff != 0) { atomicAdd(&g_rf_cnt[1], 1ull); atomicAdd(&g_rf_cnt[2], (unsigned long long) diff); }
    else if (!same_order) atomicAdd(&g_rf_cnt[3], 1ull);
    if (a[t * 10] != b[t * 10]) atomicAdd(&g_rf_cnt[4], 1ull);
    atomicMax(&g_rf_cnt[5], (unsigned long long) __float_as_uint(wd));
    atomicAdd(&g_rf_cnt[6], (unsigned long long) ((double) wd * 1e9));
}
void rf_report(int64_t tokens) {
    for (int d = 0; d < 64; ++d) {
        const int64_t taken = g_rf_taken[d].exchange(0), via_sass = g_rf_sass[d].exchange(0), chk = g_rf_chk[d].exchange(0);
        int64_t left = 0;
        std::string by_left, by_cc;
        {
            std::lock_guard<std::mutex> lk(g_rf_mu);
            for (const auto& [what, n] : g_rf_left[d]) { left += n; by_left += (by_left.empty() ? "" : ", ") + what + " x" + std::to_string(n); }
            for (const auto& [what, n] : g_rf_cc[d]) by_cc += (by_cc.empty() ? "" : ", ") + what + " x" + std::to_string(n);
            g_rf_left[d].clear();
            g_rf_cc[d].clear();
        }
        if (taken == 0 && left == 0 && chk == 0) continue;
        std::fprintf(stderr, "strata prefill: STRATA_PF_ROUTER_F16: read of %lld tokens, CUDA%d: %lld router products on the HFMA2 kernels (%lld on the hand-scheduled one; compiler kernel: %s), "
                             "%lld left to the FP32 router (%s)\n", (long long) tokens, d, (long long) taken, (long long) via_sass, by_cc.empty() ? "-" : by_cc.c_str(), (long long) left,
                     by_left.empty() ? "-" : by_left.c_str());
        if (chk > 0) {
            unsigned long long c[8] = {}, z[8] = {};
            int prev = 0;
            cudaGetDevice(&prev);
            cudaSetDevice(d);
            cudaDeviceSynchronize();
            cudaMemcpyFromSymbol(c, g_rf_cnt, sizeof c);
            cudaMemcpyToSymbol(g_rf_cnt, z, sizeof z);
            cudaSetDevice(prev);
            const uint32_t mxb = (uint32_t) c[5];
            float mx = 0.0f;
            std::memcpy(&mx, &mxb, 4);
            const double nt = (double) std::max<unsigned long long>(c[0], 1ull);
            std::fprintf(stderr, "strata prefill: STRATA_PF_ROUTER_F16_CHECK: read of %lld tokens, CUDA%d: %llu routed tokens against the FP32 router's: %llu with another top-10 set (%.4f%%), "
                                 "%llu ids outside the FP32 router's set (%.5f%% of %llu), %llu more in another order only, %llu with another top-1 id; routing weights: max |dw| %.3e, mean per token of the max |dw| %.3e\n",
                         (long long) tokens, d, c[0], c[1], 100.0 * (double) c[1] / nt, c[2], 100.0 * (double) c[2] / (10.0 * nt), (unsigned long long) (10.0 * nt), c[3], c[4], (double) mx,
                         (double) c[6] * 1e-9 / nt);
        }
    }
}

// ---- STRATA_PF_PLE_F16=1 (gemm.hpp: ple_f16_image, ple_f16): the PLE block's key and value projections on the dense HFMA2 kernels.  Per device since the last report: the products taken (and how many on the
// hand-scheduled kernel), the calls left to the old route by reason, and the activation image's range counters (device words: the largest |x| as float bits, the values past the FP16 range or not finite)
std::atomic<int64_t> g_pl_taken[64][2], g_pl_sass[64][2];
std::mutex g_pl_mu;
std::map<std::string, int64_t> g_pl_left[64];   // "key: reason" -> calls the old route ran
std::map<std::string, int64_t> g_pl_cc[64];     // reasons the compiler's HFMA2 kernel ran instead of the hand-scheduled one
float g_pl_wmax[64][2];                         // max |w| of the two weights' FP16 copies (under g_pl_mu)
__device__ unsigned long long g_pl_x[4];        // max |x| bits, bad values
__global__ void pl_image_kernel(const float4* __restrict__ x, uint2* __restrict__ y, int64_t n4) {
    float mx = 0.0f;
    unsigned bad = 0;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int64_t) gridDim.x * blockDim.x) {
        const float4 v = x[i];
        const float a[4] = {v.x, v.y, v.z, v.w};
        unsigned short h[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float f = fabsf(a[j]);
            if (!(f <= 65504.0f)) ++bad; else mx = fmaxf(mx, f);
            h[j] = __half_as_ushort(__float2half_rn(a[j]));
        }
        y[i] = make_uint2((unsigned) h[0] | ((unsigned) h[1] << 16), (unsigned) h[2] | ((unsigned) h[3] << 16));
    }
    for (int o = 16; o > 0; o >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        bad += __shfl_xor_sync(0xffffffffu, bad, o);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicMax(&g_pl_x[0], (unsigned long long) __float_as_uint(mx));
        if (bad != 0) atomicAdd(&g_pl_x[1], (unsigned long long) bad);
    }
}
void pl_report(int64_t tokens) {
    for (int d = 0; d < 64; ++d) {
        int64_t taken = 0, via_sass = 0;
        for (int p = 0; p < 2; ++p) { taken += g_pl_taken[d][p].exchange(0); via_sass += g_pl_sass[d][p].exchange(0); }
        int64_t left = 0;
        std::string by_left, by_cc;
        float wk = 0.0f, wv = 0.0f;
        {
            std::lock_guard<std::mutex> lk(g_pl_mu);
            for (const auto& [what, n] : g_pl_left[d]) { left += n; by_left += (by_left.empty() ? "" : ", ") + what + " x" + std::to_string(n); }
            for (const auto& [what, n] : g_pl_cc[d]) by_cc += (by_cc.empty() ? "" : ", ") + what + " x" + std::to_string(n);
            g_pl_left[d].clear();
            g_pl_cc[d].clear();
            wk = g_pl_wmax[d][0];
            wv = g_pl_wmax[d][1];
        }
        if (taken == 0 && left == 0) continue;
        unsigned long long c[4] = {}, z[4] = {};
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(d);
        cudaDeviceSynchronize();
        cudaMemcpyFromSymbol(c, g_pl_x, sizeof c);
        cudaMemcpyToSymbol(g_pl_x, z, sizeof z);
        cudaSetDevice(prev);
        const uint32_t mxb = (uint32_t) c[0];
        float mx = 0.0f;
        std::memcpy(&mx, &mxb, 4);
        std::fprintf(stderr, "strata prefill: STRATA_PF_PLE_F16: read of %lld tokens, CUDA%d: %lld PLE products (key, value) on the HFMA2 kernels (%lld on the hand-scheduled one; compiler kernel: %s), "
                             "%lld left to the FP32 route (%s); activation image: max |x| %.4g, %llu values past the FP16 range or not finite; FP16 weight copies: max |w| key %.4g, value %.4g "
                             "(bound of one 64-product FP16 sum: %.4g of 65504)\n",
                     (long long) tokens, d, (long long) taken, (long long) via_sass, by_cc.empty() ? "-" : by_cc.c_str(), (long long) left, by_left.empty() ? "-" : by_left.c_str(), (double) mx,
                     c[1], (double) wk, (double) wv, 64.0 * (double) mx * (double) std::max(wk, wv));
    }
}

// ---- GP100, opt-in (STRATA_PF_HFMA_SASS=1, with STRATA_PF_HFMA and / or STRATA_PF_HFMA_HC): the hand-scheduled SASS kernels of gemm_sass.cuh
// in place of the compiler's gemm_kernel / gemm_sk_kernel where they cover the call; the rest keeps the compiler kernel, counted by what and
// why (printed with the read's reports).  The outputs are the same bits.
bool sass_on() {
    static const bool on = [] { const char* e = std::getenv("STRATA_PF_HFMA_SASS"); return e != nullptr && std::atoi(e) != 0; }();
    return on;
}
std::atomic<int64_t> g_sass_taken[64][strata::prefill::sass::kKernCount];
std::mutex g_sass_mu;
std::map<std::string, int64_t> g_sass_left[64];   // "what: reason" -> calls
void sass_left(int dev, const char* what, const char* why) {
    std::string w = what != nullptr ? what : "?";
    if (w.compare(0, 4, "blk.") == 0) {   // blk.12.attn_qkv.weight -> attn_qkv.weight
        size_t i = 4;
        while (i < w.size() && std::isdigit((unsigned char) w[i])) ++i;
        if (i < w.size() && w[i] == '.') w.erase(0, i + 1);
    }
    std::lock_guard<std::mutex> lk(g_sass_mu);
    ++g_sass_left[dev][w + ": " + why];
}
// STRATA_PF_HFMA_SASS_VERIFY=1 (diagnostic, slow): after every hand-scheduled launch the compiler kernel runs into a scratch buffer and the words that
// differ are counted and named (a run with it must print none)
__global__ void sass_cmp_kernel(const unsigned* __restrict__ a, const unsigned* __restrict__ b, const int64_t n, unsigned* __restrict__ out) {
    for (int64_t i = (int64_t) blockIdx.x * 256 + threadIdx.x; i < n; i += (int64_t) gridDim.x * 256)
        if (a[i] != b[i]) {
            atomicAdd(&out[0], 1u);
            atomicMin(&out[1], (unsigned) i);
        }
}
bool sass_verify_on() {
    static const bool on = [] { const char* e = std::getenv("STRATA_PF_HFMA_SASS_VERIFY"); return e != nullptr && std::atoi(e) != 0; }();
    return on;
}
void sass_verify_dense(int dev, int k, const uint16_t* X, const uint16_t* wh, const float* Y, int64_t T, int64_t N, int64_t K, cudaStream_t st) {
    static float* ref[64] = {};
    static size_t cap[64] = {};
    static unsigned* d_out[64] = {};
    const size_t need = (size_t) T * (size_t) N;
    if (cap[dev] < need) {
        cudaStreamSynchronize(st);
        cudaFree(ref[dev]);
        ref[dev] = nullptr;
        cap[dev] = 0;
        if (cudaMalloc((void**) &ref[dev], need * 4) != cudaSuccess) { cudaGetLastError(); ref[dev] = nullptr; return; }
        cap[dev] = need;
    }
    if (d_out[dev] == nullptr && cudaMalloc((void**) &d_out[dev], 8) != cudaSuccess) { cudaGetLastError(); return; }
    const unsigned init[2] = {0u, 0xFFFFFFFFu};
    cudaMemcpyAsync(d_out[dev], init, 8, cudaMemcpyHostToDevice, st);
    strata::prefill::hfma::gemm(X, K, wh, ref[dev], T, N, K, N, st);
    sass_cmp_kernel<<<1024, 256, 0, st>>>(reinterpret_cast<const unsigned*>(Y), reinterpret_cast<const unsigned*>(ref[dev]), (int64_t) need, d_out[dev]);
    unsigned res[2] = {0u, 0u};
    cudaMemcpyAsync(res, d_out[dev], 8, cudaMemcpyDeviceToHost, st);
    const cudaError_t e = cudaStreamSynchronize(st);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_SASS_VERIFY: GPU %d: %s T=%lld N=%lld K=%lld: %s\n", dev, strata::prefill::sass::kKern[k].sym, (long long) T, (long long) N,
                     (long long) K, cudaGetErrorString(e));
        std::exit(1);
    }
    static std::atomic<int> printed{0};
    if (res[0] != 0 || printed.fetch_add(1) < 8)
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_SASS_VERIFY: GPU %d: %s T=%lld: %u of %zu words differ from the compiler kernel's%s (X %p Y %p)\n", dev,
                     strata::prefill::sass::kKern[k].sym, (long long) T, res[0], need,
                     res[0] ? (std::string(", first at row ") + std::to_string(res[1] / N) + " col " + std::to_string(res[1] % N)).c_str() : "", (const void*) X, (const void*) Y);
}
// the hc slice's twin: the compiler kernel as the engine would have run it (its own split Zc, the part buffers of its own) into scratch, compared with y
void sass_verify_hc(int dev, int prod, const uint16_t* x16, const uint16_t* w16, const float* y, int64_t rows, int64_t N, int64_t K, int Zc, float wmax, cudaStream_t st) {
    namespace hcn = strata::prefill::hfma::hc;
    static float* ref[64] = {};
    static float* part[64] = {};
    static size_t cap[64] = {}, capp[64] = {};
    static unsigned* d_out[64] = {};
    const size_t need = (size_t) rows * (size_t) N, needp = need * (size_t) std::max(Zc, 1);
    cudaStreamSynchronize(st);
    if (cap[dev] < need) {
        cudaFree(ref[dev]);
        ref[dev] = nullptr;
        cap[dev] = 0;
        if (cudaMalloc((void**) &ref[dev], need * 4) != cudaSuccess) { cudaGetLastError(); return; }
        cap[dev] = need;
    }
    if (capp[dev] < needp) {
        cudaFree(part[dev]);
        part[dev] = nullptr;
        capp[dev] = 0;
        if (cudaMalloc((void**) &part[dev], needp * 4) != cudaSuccess) { cudaGetLastError(); return; }
        capp[dev] = needp;
    }
    if (d_out[dev] == nullptr && cudaMalloc((void**) &d_out[dev], 8) != cudaSuccess) { cudaGetLastError(); return; }
    const unsigned init[2] = {0u, 0xFFFFFFFFu};
    cudaMemcpyAsync(d_out[dev], init, 8, cudaMemcpyHostToDevice, st);
    const dim3 grid((unsigned) ((N + 127) / 128), (unsigned) ((rows + 127) / 128), (unsigned) Zc);
    hcn::gemm_sk_kernel<<<grid, hfma::kThreads, 0, st>>>(reinterpret_cast<const uint4*>(x16), reinterpret_cast<const uint4*>(w16), Zc > 1 ? part[dev] : ref[dev],
                                                         (int) rows, (int) N, (int) K, (int) (K / 8), N, rows * N, Zc, wmax);
    if (Zc > 1) hcn::reduce_kernel<<<(unsigned) std::min<int64_t>(2048, (rows * N + 255) / 256), 256, 0, st>>>(part[dev], ref[dev], rows, N, N, Zc, rows * N);
    sass_cmp_kernel<<<1024, 256, 0, st>>>(reinterpret_cast<const unsigned*>(y), reinterpret_cast<const unsigned*>(ref[dev]), (int64_t) need, d_out[dev]);
    unsigned res[2] = {0u, 0u};
    cudaMemcpyAsync(res, d_out[dev], 8, cudaMemcpyDeviceToHost, st);
    const cudaError_t e = cudaStreamSynchronize(st);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_SASS_VERIFY: GPU %d: hc %s rows=%lld: %s\n", dev, prod == 0 ? "down" : "up", (long long) rows, cudaGetErrorString(e));
        std::exit(1);
    }
    static std::atomic<int> printed{0};
    if (res[0] != 0 || printed.fetch_add(1) < 8)
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_SASS_VERIFY: GPU %d: hc %s rows=%lld (compiler split %d): %u of %zu words differ from the compiler kernel's%s\n", dev,
                     prod == 0 ? "down" : "up", (long long) rows, Zc, res[0], need,
                     res[0] ? (std::string(", first at row ") + std::to_string(res[1] / N) + " col " + std::to_string(res[1] % N)).c_str() : "");
}
// "attn_qkv N10240 K2560 x75, ..." for the kernels this device ran since the last call (they are cleared), and the fallbacks "what: reason xN, ..."
void sass_take(int d, int64_t& n_taken, std::string& by_kernel, int64_t& n_left, std::string& by_reason, bool hc) {
    n_taken = 0;
    n_left = 0;
    by_kernel.clear();
    by_reason.clear();
    for (int k = 0; k < strata::prefill::sass::kKernCount; ++k) {
        if ((k >= strata::prefill::sass::kHcDown) != hc) continue;
        const int64_t n = g_sass_taken[d][k].exchange(0);
        if (n == 0) continue;
        n_taken += n;
        by_kernel += (by_kernel.empty() ? "" : ", ") + std::string("N") + std::to_string(strata::prefill::sass::kKern[k].N) + " K" +
                     std::to_string(strata::prefill::sass::kKern[k].K) + " x" + std::to_string((long long) n);
    }
    std::lock_guard<std::mutex> lk(g_sass_mu);
    std::map<std::string, int64_t> keep;
    for (const auto& [what, n] : g_sass_left[d]) {
        const bool is_hc = what.compare(0, 5, "down:") == 0 || what.compare(0, 3, "up:") == 0;
        if (is_hc != hc) { keep.emplace(what, n); continue; }
        n_left += n;
        by_reason += (by_reason.empty() ? "" : ", ") + what + " x" + std::to_string((long long) n);
    }
    g_sass_left[d] = std::move(keep);
}

// ---- GP100, opt-in (STRATA_PF_HFMA_HC=1): the hyper-connection read's BF16 projections on FP16 HFMA2 --------------------
// The down (N 320, K 10240) and up (N 10240, K 320) projections of every half-layer's hyper-connection read (the third,
// the inject projection, has N = 4: its 128-wide tile would be empty, it keeps its route).  On Pascal these BF16 products
// run as cublasSgemm on the BF16 weight's resident FP32 copy and the activation widened per slice; here the weight's FP16
// copy (half the bytes, the same shadow books as the dense projections' FP16 copies, the same reserve) and an FP16 image
// of the activation scaled by a power of two feed gemm_hfma_hc.cuh's split-K HFMA2 kernel.  A BF16 weight in the FP16 range
// is exact in FP16 (an 8-bit mantissa; only values below 2^-14 lose bits to the subnormals).  The sums are FP16 per 64
// products, so the outputs are not bitwise the FP32 route's; a call the route cannot take is counted by product and reason
// and runs the old route.
struct HcW { const uint16_t* p = nullptr; float wmax = 0.0f; const char* why = nullptr; };
std::mutex g_hcw_mu;
std::unordered_map<const void*, HcW> g_hcw;   // BF16 weight address -> its FP16 copy and max |w| (or why not)
struct HcDev {
    bool seen = false;
    int64_t taken[3] = {};                    // [down, up, other] calls
    std::map<std::string, int64_t> left;      // "product: reason" -> calls
    int copies = 0;
    double copy_mib = 0.0, min_free_mib = 1e30;
    float wmax_lo = 1e30f, wmax_hi = 0.0f;
};
HcDev g_hcd[64];
std::mutex g_hcd_mu;
struct HcTm {
    cudaEvent_t a = nullptr, b = nullptr;
    int prod = 0;
    double macs = 0.0;
};
std::vector<HcTm> g_hctm[64];
size_t g_hctm_n[64] = {};
void* g_hc_state_ptr[64] = {};
bool hc_timing_on() {
    static const bool on = std::getenv("STRATA_PREFILL_TIMING") != nullptr;
    return on;
}
int hc_prod(const char* name) {
    const std::string n = name != nullptr ? name : "";
    return n.find("down") != std::string::npos ? 0 : n.find("up") != std::string::npos ? 1 : 2;
}
const char* hc_prod_name(int p) { return p == 0 ? "down" : p == 1 ? "up" : "other"; }
void* hc_state_ptr(int dev) {
    std::lock_guard<std::mutex> lk(g_hcd_mu);
    if (g_hc_state_ptr[dev] == nullptr) {
        void* p = nullptr;
        if (cudaGetSymbolAddress(&p, strata::prefill::hfma::hc::g_hc) != cudaSuccess) {
            std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_HC: the device state symbol is missing on GPU %d\n", dev);
            std::exit(1);
        }
        g_hc_state_ptr[dev] = p;
    }
    return g_hc_state_ptr[dev];
}
void hc_left(int dev, const char* name, const char* why) {
    std::lock_guard<std::mutex> lk(g_hcd_mu);
    HcDev& d = g_hcd[dev];
    d.seen = true;
    ++d.left[std::string(hc_prod_name(hc_prod(name))) + ": " + why];
}
// the FP16 copy of a BF16 weight (once per weight, with its scan: finite and inside the FP16 range, else refused)
HcW hc_weight(const uint16_t* W, int64_t N, int64_t K, cudaStream_t st, int dev) {
    std::lock_guard<std::mutex> lk(g_hcw_mu);
    if (const auto it = g_hcw.find(W); it != g_hcw.end()) return it->second;
    HcW r;
    unsigned* d_stats = nullptr;
    unsigned h[4] = {};
    const int64_t n = N * K;
    if (cudaMalloc((void**) &d_stats, 4 * sizeof(unsigned)) != cudaSuccess) {
        cudaGetLastError();
        r.why = "no room for its scan";
    } else {
        cudaMemsetAsync(d_stats, 0, 4 * sizeof(unsigned), st);
        strata::prefill::hfma::hc::wscan_kernel<<<(unsigned) std::min<int64_t>(1024, (n + 255) / 256), 256, 0, st>>>(W, n, d_stats);
        cudaMemcpyAsync(h, d_stats, 4 * sizeof(unsigned), cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        cudaFree(d_stats);
        float wmax = 0.0f;
        std::memcpy(&wmax, &h[0], 4);
        if (h[1] != 0) r.why = "a non-finite BF16 weight";
        else if (h[2] != 0) r.why = "a BF16 weight outside the FP16 range";
        else {
            r.p = shadow_get_h(W, n, st, [&](uint16_t* dst) { bf16_to_f16(W, dst, n, st); });
            if (r.p == nullptr) {
                r.why = "no room for its FP16 copy";
            } else {
                r.wmax = wmax;
                size_t free_b = 0, total_b = 0;
                cudaMemGetInfo(&free_b, &total_b);
                std::lock_guard<std::mutex> dk(g_hcd_mu);
                HcDev& d = g_hcd[dev];
                d.seen = true;
                ++d.copies;
                d.copy_mib += (double) n * 2 / 1048576.0;
                d.min_free_mib = std::min(d.min_free_mib, free_b / 1048576.0);
                d.wmax_lo = std::min(d.wmax_lo, wmax);
                d.wmax_hi = std::max(d.wmax_hi, wmax);
            }
        }
    }
    g_hcw.emplace(W, r);
    return r;
}
int hc_pick_z(int64_t gx, int64_t gy, int64_t K, int nsm) {
    const int64_t G = (K / 32 + 3) / 4, b0 = gx * gy;
    int best = 1;
    double bw = 1e30;
    for (int z = 1; z <= std::min<int64_t>(8, G); ++z) {
        const double w = std::ceil((double) (b0 * z) / nsm) / z;
        if (w < bw - 1e-9) { bw = w; best = z; }
    }
    return best;
}
#endif
}  // namespace
#endif
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
void gemm_set_dense_f32(int mode) { g_dense_f32 = mode; }
// STRATA_DENSE_F32's product with the weight already FP32: the same slices of X, the same cublasSgemm calls
bool Gemm::sgemm_f32w(const uint16_t* X, const float* wf, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {
    const int64_t n_sl = (T * K + kXSliceElems - 1) / kXSliceElems;   // even slices of at most 64 MiB as fp32
    const int64_t x_rows = (T + n_sl - 1) / n_sl;
    if (tc_x_elems_ < 2 * x_rows * K) cudaStreamSynchronize((cudaStream_t) stream_);   // grow() frees what may be in use
    if (!grow(tc_x_, tc_x_elems_, 2 * x_rows * K)) { cudaGetLastError(); return false; }
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "prefill gemm: the dense projections' FP16 GEMMs run as cublasSgemm on resident FP32 copies "
                             "(STRATA_DENSE_F32=1 + STRATA_DENSE_F32_RESIDENT=1)\n");
    }
    float* const xf = reinterpret_cast<float*>(tc_x_);
    for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
        const int64_t n = std::min<int64_t>(x_rows, T - t0);
        f16_to_f32(X + t0 * K, xf, n * K, (cudaStream_t) stream_);
        sgemm_f32(xf, wf, Y + t0 * ldy, n, N, K, ldy, beta);
    }
    return true;
}
// The FP32 route's product: production's cublasSgemm call, or (STRATA_GEMM_ALGO6, inside its table) the same product
// through cublasGemmEx with a fixed algorithm - which w4_algo6_verify found byte for byte the cublasSgemm output's.
void Gemm::sgemm_f32(const float* xf, const float* wf, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {
    const float alpha = 1.0f;
    if (gemm_algo6_mode() >= 1 && f32_algo6_t(N, K, T)) {
        static bool said = false;
        if (!said) {
            said = true;
            std::fprintf(stderr, "prefill gemm: the FP32 dense products run as cublasGemmEx with ALGO6 "
                                 "(STRATA_GEMM_ALGO6=%d; bitwise cublasSgemm's at the sizes of the table)\n", gemm_algo6_mode());
        }
        ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, wf,
                        CUDA_R_32F, (int) K, xf, CUDA_R_32F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_ALGO6),
           "cublasGemmEx (FP32 operands, STRATA_GEMM_ALGO6)");
        return;
    }
    ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, wf, (int) K,
                   xf, (int) K, &beta, Y, (int) ldy),
       "cublasSgemm (FP16 operands, STRATA_DENSE_F32_RESIDENT)");
}
// STRATA_GEMM_ALGO6=2's product: op(A) = N on the resident column-major weight, op(B) = T on the per-slice transposed
// activation; cublasGemmEx(..., ALGO6), the same products in the same order as sgemm_f32w's call (w4_algo6_verify).
bool Gemm::sgemm_nt(const uint16_t* X, const float* wt, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {
    const int64_t n_sl = (T * K + kXSliceElems - 1) / kXSliceElems;
    const int64_t x_rows = (T + n_sl - 1) / n_sl;
    if (nt_x_elems_ < x_rows * K) cudaStreamSynchronize((cudaStream_t) stream_);
    if (!grow_f32(nt_x_, nt_x_elems_, x_rows * K)) { cudaGetLastError(); return false; }
    const float alpha = 1.0f;
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "prefill gemm: the FP32 dense products run as cublasGemmEx(OP_N, OP_T) with ALGO6 on the "
                             "transposed weight copies (STRATA_GEMM_ALGO6=2)\n");
    }
    for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
        const int64_t n = std::min<int64_t>(x_rows, T - t0);
        f16_transpose_x(X + t0 * K, nt_x_, n, K, (cudaStream_t) stream_);
        ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_N, CUBLAS_OP_T, (int) N, (int) n, (int) K, &alpha, wt,
                        CUDA_R_32F, (int) N, nt_x_, CUDA_R_32F, (int) n, &beta, Y + t0 * ldy, CUDA_R_32F, (int) ldy,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_ALGO6),
           "cublasGemmEx (FP32 operands, transposed, STRATA_GEMM_ALGO6=2)");
    }
    return true;
}
// STRATA_PF_HFMA=1's product (see the block above): false = nothing launched, native() runs; the call is counted under `name`
bool Gemm::native_hfma(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                       int64_t ldy, int64_t ldx, const char* name) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (T <= 0 || N <= 0) return true;
    if (ldx <= 0) ldx = K;
    if (ldy <= 0) ldy = N;
    const char* why = nullptr;
    if (current_cc() != 60) why = "not a GP100";
    else if (K <= 0 || K % 32 != 0) why = "K is not a multiple of 32";
    else if (T > INT_MAX || N > INT_MAX || K > INT_MAX || ldx > INT_MAX) why = "a size past 2^31";
    else if (ldx < K || ldx % 8 != 0) why = "X row stride";
    else if (ldy < N) why = "Y row stride";
    else if (((uintptr_t) X & 15) != 0) why = "X not 16-byte aligned";
    const uint16_t* wh = nullptr;
    if (why == nullptr) {
        wh = shadow_get_h(W_blocks, N * K, (cudaStream_t) stream_, [&](uint16_t* dst) {
            strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, dst, stream_);
        });
        if (wh == nullptr) why = "no room for its FP16 copy";
    }
    if (why != nullptr) {
        hfma_left(dev, name, why);
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "prefill gemm: the dense projections run as FP16 HFMA2 GEMMs on resident FP16 copies "
                             "(STRATA_PF_HFMA=1; FP16 sums of 64 products widened to FP32, not bitwise the FP32 routes')\n");
    }
    if (sass_on()) {   // the hand-scheduled twin of gemm_kernel where it covers the call: the same bits
        int k = -1;
        const char* r = strata::prefill::sass::dense(X, ldx, wh, Y, T, N, K, ldy, (cudaStream_t) stream_, dev, &k);
        if (r == nullptr) {
            static bool said_sass = false;
            if (!said_sass) {
                said_sass = true;
                std::fprintf(stderr, "prefill gemm: the dense FP16 HFMA2 GEMMs run as hand-scheduled SASS kernels where they cover the call "
                                     "(STRATA_PF_HFMA_SASS=1; the compiler kernels' bits, 81%% of the HFMA2 peak against 64%%)\n");
            }
            g_sass_taken[dev][k].fetch_add(1, std::memory_order_relaxed);
            g_hf_taken[dev].fetch_add(1, std::memory_order_relaxed);
            if (sass_verify_on()) sass_verify_dense(dev, k, X, wh, Y, T, N, K, (cudaStream_t) stream_);
            return true;
        }
        sass_left(dev, name, r);
    }
    strata::prefill::hfma::gemm(X, ldx, wh, Y, T, N, K, ldy, (cudaStream_t) stream_);
    g_hf_taken[dev].fetch_add(1, std::memory_order_relaxed);
    return true;
}
void gemm_hfma_report(int64_t tokens) {
    rf_report(tokens);   // STRATA_PF_ROUTER_F16 (nothing printed without it)
    pl_report(tokens);   // STRATA_PF_PLE_F16 (nothing printed without it)
    std::lock_guard<std::mutex> lk(g_hf_mu);
    for (int d = 0; d < 64; ++d) {
        const int64_t taken = g_hf_taken[d].exchange(0);
        if (taken == 0 && g_hf_left[d].empty()) continue;
        int64_t left = 0;
        std::string by;
        for (const auto& [what, n] : g_hf_left[d]) {
            left += n;
            by += (by.empty() ? " (" : ", ") + what + " x" + std::to_string(n);
        }
        int copies = 0;
        double copy_mib = 0.0;
        {
            std::lock_guard<std::mutex> sk(g_shadow_mu);
            copies = g_shadow_h_count[d];
            copy_mib = g_shadow_h_bytes[d] / 1048576.0;
        }
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA: read of %lld tokens, CUDA%d: %lld dense products on the HFMA2 "
                             "kernel, %lld left to the old route%s%s; %d FP16 weight copies kept, %.0f MiB\n", (long long) tokens, d,
                     (long long) taken, (long long) left, by.c_str(), by.empty() ? "" : ")", copies, copy_mib);
        g_hf_left[d].clear();
        if (sass_on()) {
            int64_t n_taken = 0, n_left = 0;
            std::string by_kernel, by_reason;
            sass_take(d, n_taken, by_kernel, n_left, by_reason, false);
            std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_SASS: read of %lld tokens, CUDA%d: %lld dense products on the hand-scheduled kernels (%s), "
                                 "%lld left to the compiler kernel (%s)\n", (long long) tokens, d, (long long) n_taken, by_kernel.empty() ? "-" : by_kernel.c_str(),
                         (long long) n_left, by_reason.empty() ? "-" : by_reason.c_str());
        }
    }
}
// STRATA_PF_HFMA_HC=1's product (see the block above): false = nothing launched, bf16() runs; the call is counted under `name`.
// `resident`: W is the pack's own resident tensor (not a scratch view, whose address is not a key); `x_lo`: a BF16X2 low part
// follows (a second accumulating product this route does not do).
bool Gemm::bf16_hfma(const uint16_t* X, bool x_f16, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                     int64_t ldx, bool resident, bool x_lo, const char* name, HcCompanion* comp) {
    namespace hcn = strata::prefill::hfma::hc;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (T <= 0 || N <= 0) return false;
    if (ldx <= 0) ldx = K;
    if (ldy <= 0) ldy = N;
    const cudaStream_t st = (cudaStream_t) stream_;
    const char* why = nullptr;
    if (current_cc() != 60) why = "not a GP100";
    else if (x_lo) why = "the BF16X2 low part (a second product)";
    else if (!resident) why = "a scratch view of the weight (STRATA_HC_Q8 / REQ8)";
    else if (K <= 0 || K % 32 != 0) why = "K is not a multiple of 32";
    else if (N < 64) why = "N below 64 (a 128-wide tile would be empty)";
    else if (T > INT_MAX || N > INT_MAX || K > INT_MAX || ldx > INT_MAX) why = "a size past 2^31";
    else if (ldx < K || ldx % 8 != 0) why = "X row stride";
    else if (ldy < N) why = "Y row stride";
    else if (((uintptr_t) X & 15) != 0) why = "X not 16-byte aligned";
    else if (((uintptr_t) W & 15) != 0) why = "W not 16-byte aligned";
    HcW hw;
    if (why == nullptr) {
        hw = hc_weight(W, N, K, st, dev);
        if (hw.p == nullptr) why = hw.why != nullptr ? hw.why : "no FP16 copy";
    }
    // STRATA_PF_HFMA_SMALL: the fused max + FP16 image (n8 below 2^31), and the inject product beside the down product when its weight can
    // take the route too.  The shift of the pair is picked from the larger max |w| of the two: the very value without the companion.
    constexpr int64_t kSl = 1024;
    const bool small = why == nullptr && pf_hfma_small() && kSl * (K / 8) < ((int64_t) 1 << 31);
    HcW hw2;
    if (small && comp != nullptr && comp->W != nullptr) {
        const char* w2why = nullptr;
        if (comp->N != 4 || K != 10240) w2why = "inject: not N 4 / K 10240";
        else if (!comp->resident) w2why = "inject: a scratch view of the weight";
        else if (comp->ldy != 0 && comp->ldy < comp->N) w2why = "inject: Y row stride";
        else if (((uintptr_t) comp->W & 15) != 0) w2why = "inject: W not 16-byte aligned";
        else if (ldx != K) w2why = "inject: X row stride is not K";
        else {
            hw2 = hc_weight(comp->W, comp->N, K, st, dev);
            if (hw2.p == nullptr) w2why = hw2.why != nullptr ? hw2.why : "inject: no FP16 copy";
        }
        if (w2why != nullptr) {
            hc_left(dev, comp->name, w2why);
            hw2 = HcW{};
        }
    }
    const bool with_inj = hw2.p != nullptr;
    const float wm = with_inj ? std::max(hw.wmax, hw2.wmax) : hw.wmax;
    // the slices (at most 1024 tokens: the FP16 image of the activation and the split-K parts fit a small scratch) and their splits
    int nsm = 56;
    int zs[16] = {};
    int64_t need_x = 0, need_p = 0;
    if (why == nullptr) {
        int v = 0;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && v > 0) nsm = v; else cudaGetLastError();
        const int64_t n_sl = (T + kSl - 1) / kSl;
        if (n_sl > 16) {
            why = "T past 16384";
        } else {
            for (int64_t i = 0; i < n_sl; ++i) {
                const int64_t rows = std::min(kSl, T - i * kSl);
                zs[i] = hc_pick_z((N + 127) / 128, (rows + 127) / 128, K, nsm);
                // STRATA_PF_HFMA_SASS: a split whose parts are one 4-stage group each (up: K 320 = 3 groups, Z = 3) sums to the very same FP32
                // bits as no split (0 + f0, then + f1, + f2, in that order; the scale is a power of two) and costs the part buffers' writes and
                // the reduce kernel's reads (0.5 ms of a 1.1 ms product): no split
                if (sass_on() && zs[i] > 1 && zs[i] == (int) ((K / 32 + 3) / 4)) zs[i] = 1;
                need_x = std::max(need_x, rows * K);
                if (zs[i] > 1) need_p = std::max(need_p, 2 * (int64_t) zs[i] * rows * N);   // floats as 2-byte elements
            }
            if (tc_x_elems_ < need_x || tc_w_elems_ < need_p) cudaStreamSynchronize(st);   // grow() frees what may be in use
            if (!grow(tc_x_, tc_x_elems_, need_x) || (need_p > 0 && !grow(tc_w_, tc_w_elems_, need_p))) {
                cudaGetLastError();
                why = "no room for the activation scratch";
            }
        }
    }
    if (why != nullptr) {
        hc_left(dev, name, why);
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "prefill gemm: the hyper-connection projections (BF16) run as FP16 HFMA2 GEMMs on resident FP16 copies "
                             "(STRATA_PF_HFMA_HC=1; the activation scaled by a power of two per slice, split-K, FP16 sums of 64 products "
                             "widened to FP32, not bitwise the FP32 route's)\n");
    }
    if (small) {
        static bool said_small = false;
        if (!said_small) {
            said_small = true;
            std::fprintf(stderr, "prefill gemm: STRATA_PF_HFMA_SMALL=1: the hyper-connection read's max |x| and FP16 image in one pass (a second only for a slice that needs a "
                                 "shift), the inject product (N 4) on FP16 HFMA2 pairs beside the down product's image, gr_mix_r four values a thread\n");
        }
    }
    cudaEvent_t ev_a = nullptr, ev_b = nullptr;
    const int prod = hc_prod(name);
    if (hc_timing_on()) {
        std::lock_guard<std::mutex> lk(g_hcd_mu);
        auto& v = g_hctm[dev];
        if (g_hctm_n[dev] == v.size()) {
            HcTm t;
            cudaEventCreate(&t.a);
            cudaEventCreate(&t.b);
            v.push_back(t);
        }
        HcTm& t = v[g_hctm_n[dev]++];
        t.prod = prod;
        t.macs = (double) T * (double) N * (double) K;
        ev_a = t.a;
        ev_b = t.b;
    }
    void* sp = hc_state_ptr(dev);
    float* sass_ysc = nullptr;
    unsigned* sass_sink = nullptr;
    if (sass_on()) {
        std::lock_guard<std::mutex> lk(g_hcd_mu);
        static void* ysc_ptr[64] = {};
        static void* sink_ptr[64] = {};
        if (ysc_ptr[dev] == nullptr) {
            if (cudaGetSymbolAddress(&ysc_ptr[dev], hcn::g_hc_ysc) != cudaSuccess || cudaGetSymbolAddress(&sink_ptr[dev], hcn::g_hc_sink) != cudaSuccess) {
                (void) cudaGetLastError();
                ysc_ptr[dev] = sink_ptr[dev] = nullptr;
            }
        }
        sass_ysc = (float*) ysc_ptr[dev];
        sass_sink = (unsigned*) sink_ptr[dev];
    }
    if (ev_a != nullptr) cudaEventRecord(ev_a, st);
    uint16_t* const x16 = tc_x_;
    float* const part = reinterpret_cast<float*>(tc_w_);
    for (int64_t i = 0, t0 = 0; t0 < T; ++i, t0 += kSl) {
        const int64_t rows = std::min(kSl, T - t0);
        const int Z = zs[i];
        const int64_t n8 = rows * K / 8, K8 = K / 8, ldx8 = ldx / 8;
        const unsigned blocks = (unsigned) std::max<int64_t>(1, std::min<int64_t>(2048, (n8 + 1023) / 1024));
        namespace kt = strata::kernels::ktime;   // (no-ops unless STRATA_PREFILL_TIMING=2)
        kt::begin(kt::kHcXmax, st);
        cudaMemsetAsync(sp, 0, sizeof(unsigned), st);   // HcState::cur is its first member
        const uint4* xs = reinterpret_cast<const uint4*>(X + t0 * ldx);
        if (small) {   // STRATA_PF_HFMA_SMALL: one pass takes the max and writes the image at shift 0; the second kernel counts the slice and redoes the image only for a shift > 0
            if (x_f16) hcn::xcvt_kernel<true><<<blocks, 256, 0, st>>>(xs, reinterpret_cast<uint4*>(x16), (unsigned) n8, (unsigned) K8, (unsigned) ldx8);
            else hcn::xcvt_kernel<false><<<blocks, 256, 0, st>>>(xs, reinterpret_cast<uint4*>(x16), (unsigned) n8, (unsigned) K8, (unsigned) ldx8);
            kt::begin(kt::kHcCvt, st);
            // (a smaller grid: at shift 0 every block leaves at once, and the launch of 1280 such blocks costs 30 us; a shifted slice, which
            // the reads so far never had, strides over the slice with these 4 blocks an SM)
            const unsigned fix_blocks = std::min<unsigned>(blocks, 4u * (unsigned) nsm);
            if (x_f16) hcn::cvt_fix_kernel<true><<<fix_blocks, 256, 0, st>>>(xs, reinterpret_cast<uint4*>(x16), (unsigned) n8, (unsigned) K8, (unsigned) ldx8, wm);
            else hcn::cvt_fix_kernel<false><<<fix_blocks, 256, 0, st>>>(xs, reinterpret_cast<uint4*>(x16), (unsigned) n8, (unsigned) K8, (unsigned) ldx8, wm);
        } else {
            if (x_f16) hcn::xmax_kernel<true><<<blocks, 256, 0, st>>>(xs, n8, K8, ldx8);
            else hcn::xmax_kernel<false><<<blocks, 256, 0, st>>>(xs, n8, K8, ldx8);
            kt::begin(kt::kHcCvt, st);
            if (x_f16) hcn::cvt_kernel<true><<<blocks, 256, 0, st>>>(xs, reinterpret_cast<uint4*>(x16), n8, K8, ldx8, wm);
            else hcn::cvt_kernel<false><<<blocks, 256, 0, st>>>(xs, reinterpret_cast<uint4*>(x16), n8, K8, ldx8, wm);
        }
        const dim3 grid((unsigned) ((N + 127) / 128), (unsigned) ((rows + 127) / 128), (unsigned) Z);
        float* const y = Y + t0 * ldy;
        kt::begin(prod == 0 ? kt::kHcGemmDown : prod == 1 ? kt::kHcGemmUp : kt::kHcGemmInj, st);
        bool by_sass = false;
        if (sass_ysc != nullptr && sass_sink != nullptr) {   // the hand-scheduled twin of gemm_sk_kernel: it takes the scale from a word the kernel below writes
            hcn::ysc_kernel<<<1, 1, 0, st>>>(sass_ysc, wm);
            const char* r = strata::prefill::sass::hc_slice(prod == 0, x16, hw.p, y, part, rows, N, K, ldy, Z, sass_ysc,
                                                             Z == 1 ? (unsigned*) ((char*) sp + offsetof(hcn::HcState, bad_y)) : sass_sink, st, dev);
            if (r == nullptr) {
                by_sass = true;
                g_sass_taken[dev][prod == 0 ? strata::prefill::sass::kHcDown : strata::prefill::sass::kHcUp].fetch_add(1, std::memory_order_relaxed);
                static bool said_sass = false;
                if (!said_sass) {
                    said_sass = true;
                    std::fprintf(stderr, "prefill gemm: the hyper-connection FP16 HFMA2 GEMMs run as hand-scheduled SASS kernels where they cover the call "
                                         "(STRATA_PF_HFMA_SASS=1; the compiler kernel's bits; K 320 is never split)\n");
                }
            } else {
                sass_left(dev, hc_prod_name(prod), r);
            }
        } else if (sass_on()) {
            sass_left(dev, hc_prod_name(prod), "device symbols missing");
        }
        if (!by_sass)
            hcn::gemm_sk_kernel<<<grid, hfma::kThreads, 0, st>>>(reinterpret_cast<const uint4*>(x16), reinterpret_cast<const uint4*>(hw.p),
                                                                 Z > 1 ? part : y, (int) rows, (int) N, (int) K, (int) K8, Z > 1 ? N : ldy,
                                                                 rows * N, Z, wm);
        if (Z > 1) {
            kt::begin(kt::kHcReduce, st);
            hcn::reduce_kernel<<<(unsigned) std::min<int64_t>(2048, (rows * N + 255) / 256), 256, 0, st>>>(part, y, rows, N, ldy, Z, rows * N);
        }
        if (with_inj) {   // STRATA_PF_HFMA_SMALL: the inject product on the very image the down product just read
            kt::begin(kt::kHcGemmInj, st);
            const int64_t ldy2 = comp->ldy > 0 ? comp->ldy : comp->N;
            hcn::inj_kernel<<<(unsigned) ((rows + hcn::kInjRows - 1) / hcn::kInjRows), 256, 0, st>>>(
                reinterpret_cast<const uint4*>(x16), reinterpret_cast<const uint4*>(hw2.p), comp->Y + t0 * ldy2, (int) rows, ldy2, wm);
        }
        if (by_sass && sass_verify_on())
            sass_verify_hc(dev, prod, x16, hw.p, y, rows, N, K, hc_pick_z((N + 127) / 128, (rows + 127) / 128, K, nsm), wm, st);
    }
    if (const cudaError_t le = cudaGetLastError(); le != cudaSuccess) {
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_HC: a launch failed: %s\n", cudaGetErrorString(le));
        std::exit(1);
    }
    if (ev_b != nullptr) cudaEventRecord(ev_b, st);
    std::lock_guard<std::mutex> lk(g_hcd_mu);
    g_hcd[dev].seen = true;
    ++g_hcd[dev].taken[prod];
    if (with_inj) {
        ++g_hcd[dev].taken[2];
        comp->done = true;
    }
    return true;
}
void gemm_hfma_hc_report(int64_t tokens) {
    namespace hcn = strata::prefill::hfma::hc;
    std::lock_guard<std::mutex> lk(g_hcd_mu);
    int back = 0;
    cudaGetDevice(&back);
    for (int dev = 0; dev < 64; ++dev) {
        HcDev& d = g_hcd[dev];
        if (!d.seen) continue;
        hcn::HcState stt{};
        if (g_hc_state_ptr[dev] != nullptr) {
            cudaSetDevice(dev);
            if (cudaMemcpyFromSymbol(&stt, hcn::g_hc, sizeof stt) != cudaSuccess) { cudaGetLastError(); std::memset(&stt, 0, sizeof stt); }
            hcn::HcState zero{};
            if (cudaMemcpyToSymbol(hcn::g_hc, &zero, sizeof zero) != cudaSuccess) cudaGetLastError();
        }
        std::string kt;
        if (g_hctm_n[dev] > 0) {
            struct Acc { double ms = 0.0, macs = 0.0; int64_t n = 0; };
            Acc acc[3];
            cudaSetDevice(dev);
            for (size_t i = 0; i < g_hctm_n[dev]; ++i) {
                float ms = 0.0f;
                if (cudaEventSynchronize(g_hctm[dev][i].b) == cudaSuccess && cudaEventElapsedTime(&ms, g_hctm[dev][i].a, g_hctm[dev][i].b) == cudaSuccess) {
                    Acc& a = acc[g_hctm[dev][i].prod];
                    a.ms += ms; a.macs += g_hctm[dev][i].macs; ++a.n;
                } else {
                    cudaGetLastError();
                }
            }
            g_hctm_n[dev] = 0;
            double ms_all = 0.0, macs_all = 0.0;
            for (int p = 0; p < 3; ++p) {
                if (acc[p].n == 0) continue;
                char b[160];
                std::snprintf(b, sizeof b, "%s%s x%lld %.1f ms %.0f GMAC/s", kt.empty() ? "" : "; ", hc_prod_name(p), (long long) acc[p].n, acc[p].ms,
                              acc[p].ms > 0 ? acc[p].macs / acc[p].ms * 1e-6 : 0.0);
                kt += b;
                ms_all += acc[p].ms; macs_all += acc[p].macs;
            }
            char b[120];
            std::snprintf(b, sizeof b, "; all %.1f ms %.0f GMAC/s", ms_all, ms_all > 0 ? macs_all / ms_all * 1e-6 : 0.0);
            kt += b;
        }
        std::string lft;
        int64_t n_left = 0;
        for (const auto& [what, n] : d.left) { lft += (lft.empty() ? "" : ", ") + what + " x" + std::to_string((long long) n); n_left += n; }
        std::string kh;
        for (int i = 0; i <= hcn::KMAXH; ++i)
            if (stt.khist[i] != 0) {
                char b[40];
                std::snprintf(b, sizeof b, "%s%d x%u", kh.empty() ? "" : ", ", i, stt.khist[i]);
                kh += b;
            }
        float xmax = 0.0f;
        std::memcpy(&xmax, &stt.xmax_all, 4);
        size_t free_b = 0, total_b = 0;
        cudaSetDevice(dev);
        cudaMemGetInfo(&free_b, &total_b);
        std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_HC: read of %lld tokens, CUDA%d: %lld hyper-connection projections on the HFMA2 kernel "
                             "(down %lld, up %lld, other %lld), %lld left to the old route%s%s%s; %d FP16 weight copies made this read, %.0f MiB (all "
                             "FP16 copies kept: %d, %.0f MiB); input max |x| %.2f over %u slices, activation shift k (2^-k) per slice: %s; max |w| of the "
                             "copies made: %.3g .. %.3g; non-finite: input in %u slices, output %u values; free VRAM now %.0f MiB (lowest at a copy: %.0f); "
                             "kernel time: %s\n",
                     (long long) tokens, dev, (long long) (d.taken[0] + d.taken[1] + d.taken[2]), (long long) d.taken[0], (long long) d.taken[1],
                     (long long) d.taken[2], (long long) n_left, lft.empty() ? "" : " (", lft.c_str(), lft.empty() ? "" : ")", d.copies, d.copy_mib,
                     g_shadow_h_count[dev], g_shadow_h_bytes[dev] / 1048576.0, xmax, stt.slices, kh.empty() ? "-" : kh.c_str(),
                     d.wmax_hi > 0 ? d.wmax_lo : 0.0f, d.wmax_hi, stt.bad_x, stt.bad_y, free_b / 1048576.0,
                     d.min_free_mib < 1e29 ? d.min_free_mib : -1.0, kt.empty() ? "-" : kt.c_str());
        if (sass_on()) {
            int64_t n_taken = 0, n_left = 0;
            std::string by_kernel, by_reason;
            sass_take(dev, n_taken, by_kernel, n_left, by_reason, true);
            std::fprintf(stderr, "strata prefill: STRATA_PF_HFMA_SASS: read of %lld tokens, CUDA%d: %lld hyper-connection slices on the hand-scheduled kernels (%s), "
                                 "%lld left to the compiler kernel (%s)\n", (long long) tokens, dev, (long long) n_taken, by_kernel.empty() ? "-" : by_kernel.c_str(),
                         (long long) n_left, by_reason.empty() ? "-" : by_reason.c_str());
        }
        d = HcDev{};
    }
    cudaSetDevice(back);
}
#else
void gemm_set_dense_f32(int) {}
bool Gemm::native_hfma(const uint16_t*, int, const void*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, const char*) {
    return false;   // not the experimental sm_60 CUDA build: native() runs
}
void gemm_hfma_report(int64_t) {}
bool Gemm::bf16_hfma(const uint16_t*, bool, const uint16_t*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, bool, bool, const char*, HcCompanion*) {
    return false;   // not the experimental sm_60 CUDA build: bf16() runs
}
void gemm_hfma_hc_report(int64_t) {}
bool Gemm::bf16_router_gate(const uint16_t*, const uint16_t*, const uint16_t*, bool, float*, int64_t, const uint16_t*, float*, int64_t, int64_t) {
    return false;   // not the experimental sm_60 CUDA build: bf16() runs
}
bool Gemm::router_f16(const uint16_t*, const uint16_t*, float*, int64_t, int64_t, int64_t, bool) {
    return false;   // not the experimental sm_60 CUDA build: the FP32 router runs
}
void Gemm::router_f16_cmp(const int32_t*, const int32_t*, const float*, const float*, int64_t, void*) {}
void Gemm::ple_f16_image(const float*, uint16_t*, int64_t) {}
bool Gemm::ple_f16(const uint16_t*, const uint16_t*, float*, int64_t, int64_t, int64_t, int) {
    return false;   // not the experimental sm_60 CUDA build: the FP32 route runs
}
#endif
bool Gemm::bf16_hcd_exact(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K) {
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    static std::atomic<bool> told{false};
    if (!hipblaslt_state_ || N != 320 || K != 10240 || T < 1) {
        if (N == 320 && K == 10240 && !told.exchange(true))
            std::fprintf(stderr, "strata: STRATA_HCD_EXACT: no hipBLASLt tuning table is loaded, so the exact kernel has nothing to match: "
                                 "the hyper-connection down projection stays on hipBLAS\n");
        return false;
    }
    // The kernel copies hipBLASLt solution 1176 / 1177's summation order, which is only true of the library build the
    // table was calibrated with: take it only when the table makes hipBLASLt pick one of those two for this T.
    const int id = hipblaslt_solution_for(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, T, N, K, N, 0.0f);
    if (id != 1176 && id != 1177) {
        if (T >= 4096 && !told.exchange(true))   // (a short chunk may legitimately get another solution: not worth a line)
            std::fprintf(stderr, "strata: STRATA_HCD_EXACT: the tuning table gives the hyper-connection down projection (N 320, K 10240) "
                                 "solution %d at T %lld, not 1176 / 1177: hipBLASLt runs it (a changed table or library; "
                                 "tools/hip/gfx1151-hipblaslt-100401.txt is the one measured)\n",
                         id, (long long) T);
        return false;
    }
    return strata_pf_hcdown_exact_bf16(X, ldx, W, Y, T, N, K, stream_);
#else
    (void) X; (void) ldx; (void) W; (void) Y; (void) T; (void) N; (void) K;
    return false;
#endif
}

#if defined(__HIPCC__)
// ---- RDNA2 (gfx103x): the prompt path's GEMMs as FP32 SGEMMs ------------------------------------------------------
// The GEMMs here take 16-bit inputs (FP16 or BF16) and write FP32 (rocBLAS's "HS" / "BS" types).  The rocBLAS the engine
// ships has tuned gfx1030 kernels for FP16 -> FP16 ("HH"), int8 and FP32 -> FP32 ("SS") only: every 16-bit-input,
// FP32-output GEMM runs a "fallback" kernel at ~5 TFLOPS.  Measured on an RX 6800 (ROCm 10.2.0a20260930) with the
// engine's exact calls (opA = T, opB = N, fp32 compute): at the Coder's prompt shapes the FP16 GEMMs of one fresh
// 12K-token prompt take 14.1 s, the same GEMMs as SGEMM on widened copies 4.7 s including the widening (e.g.
// N 10240 x T 8192 x K 2560: 86.7 -> 27.7 ms); the BF16 hyper-connection GEMMs 10.9 -> 7.5 ms (N 320, K 10240).
// Widening FP16 / BF16 to FP32 is exact and SGEMM accumulates in FP32 like the native call, so the products are the
// same up to summation order.  The weight is widened once per call, the activations in slices of up to 128 MiB of
// FP32 (an 8,192-token chunk at K = 2560 is one slice).  Small N (< 64: routers, gates) keeps the native call:
// there the widening costs more than the GEMM gains.  If the FP32 buffers cannot be allocated, the native call runs.
//   OPT-IN here: STRATA_RDNA2_SGEMM=1 turns it on for gfx103x devices (other cards keep the native call, also in a
//   mixed layer split); =all turns it on for every AMD card (A/B arms, tests); unset or 0 is the native path.  The
//   author ran it as the gfx103x default (RX 6800 343 -> 534 tok/s); it is not bit-identical to the native call.
namespace {
__global__ void rdna2_bf16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float((uint32_t) in[i] << 16);
}
__global__ void rdna2_f16_to_f32_kernel(const __half* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}
void rdna2_widen(const uint16_t* in, float* out, int64_t n, bool bf16, cudaStream_t st) {
    if (n <= 0) return;
    const unsigned blocks = (unsigned) ((n + 255) / 256);
    if (bf16) rdna2_bf16_to_f32_kernel<<<blocks, 256, 0, st>>>(in, out, n);
    else rdna2_f16_to_f32_kernel<<<blocks, 256, 0, st>>>(reinterpret_cast<const __half*>(in), out, n);
}
// whether this GEMM runs as an SGEMM (see above)
bool rdna2_sgemm_on(int64_t N) {
    static const int mode = [] {   // 0 off (unset), 1 gfx103x devices, 2 every AMD device
        const char* v = std::getenv("STRATA_RDNA2_SGEMM");
        if (v == nullptr || v[0] == '\0') return 0;
        if (std::strcmp(v, "all") == 0) return 2;
        return std::atoi(v) > 0 ? 1 : 0;
    }();
    if (mode == 0 || N < 64) return false;
    if (mode == 2) return true;
    static std::atomic<int> arch[64] = {};   // per device: 0 not known, 1 gfx103x, 2 another architecture
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) { (void) hipGetLastError(); return false; }
    int v = arch[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) { (void) hipGetLastError(); return false; }
        v = std::strncmp(prop.gcnArchName, "gfx103", 6) == 0 ? 1 : 2;
        arch[dev].store(v, std::memory_order_relaxed);
    }
    return v == 1;
}
// `have`, `want`: FP32 elements (the buffers are the instance's 2-byte tc_ buffers, sized in 2-byte elements)
bool rdna2_grow(uint16_t*& p, int64_t& have_u16, int64_t want_f32) {
    const int64_t want = 2 * want_f32;
    if (have_u16 >= want) return true;
    if (p) cudaFree(p);
    p = nullptr;
    have_u16 = 0;
    if (cudaMalloc((void**) &p, (size_t) want * 2) != cudaSuccess) { (void) cudaGetLastError(); p = nullptr; return false; }
    have_u16 = want;
    return true;
}
constexpr int64_t kRdna2SliceF32 = 32ll << 20;   // 128 MiB of FP32 activations per slice
}  // namespace

// Y^T[N, T] (+)= W[N, K] . X^T[K, T] as SGEMMs on FP32 copies; false: not this card / shape, or no buffers (the caller
// runs its native GEMM, nothing was written)
bool Gemm::rdna2_sgemm(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                       float beta, bool bf16) {
    if (K <= 0 || !rdna2_sgemm_on(N)) return false;
    const int64_t rows = std::max<int64_t>(1, std::min<int64_t>(T, kRdna2SliceF32 / K));
    // growing frees a buffer an earlier GEMM on this stream may still read (as the sm_60 FP32 route does)
    if (tc_w_elems_ < 2 * N * K || tc_x_elems_ < 2 * rows * K) (void) hipStreamSynchronize((hipStream_t) stream_);
    if (!rdna2_grow(tc_w_, tc_w_elems_, N * K) || !rdna2_grow(tc_x_, tc_x_elems_, rows * K)) return false;
    float* const wf = reinterpret_cast<float*>(tc_w_);
    float* const xf = reinterpret_cast<float*>(tc_x_);
    const float alpha = 1.0f;
    rdna2_widen(W, wf, N * K, bf16, (cudaStream_t) stream_);
    for (int64_t t0 = 0; t0 < T; t0 += rows) {
        const int64_t n = std::min<int64_t>(rows, T - t0);
        rdna2_widen(X + t0 * K, xf, n * K, bf16, (cudaStream_t) stream_);
        // every slice is a disjoint block of Y, so each gets the caller's beta
        ck(hipblasSgemm((hipblasHandle_t) handle_, HIPBLAS_OP_T, HIPBLAS_OP_N, (int) N, (int) n, (int) K, &alpha, wf,
                        (int) K, xf, (int) K, &beta, Y + t0 * ldy, (int) ldy),
           "hipblasSgemm (RDNA2)");
    }
    return true;
}
#endif

bool Gemm::bf16_converts_x() const {
#if defined(__HIPCC__) || defined(STRATA_EXPERIMENTAL_SM60)
    return false;
#else
    return bf16_path() == 1;
#endif
}
__global__ void f16_to_bf16_kernel(const __half* __restrict__ in, uint16_t* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (uint16_t) (__float_as_uint(__half2float(in[i])) >> 16);   // exact: the value was a BF16
}
void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta, int64_t ldx, bool x_f16, bool w_resident) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (ldx <= K) ldx = 0;
#ifdef STRATA_USE_HIP
    // #313 (opt-in STRATA_WMMA_GEMM=1; STRATA_WMMA_BF16=0 excludes bf16): RDNA3 / RDNA3.5 WMMA dense GEMM, before
    // hipBLASLt; it returns false for shapes it does not take (and off gfx11) and this falls through
    static const bool wmma_on = [] { const char* v = std::getenv("STRATA_WMMA_GEMM"); return v && v[0] != 0 && v[0] != '0'; }();
    static const bool bf16_on = [] { const char* v = std::getenv("STRATA_WMMA_BF16"); return !v || (v[0] != 0 && v[0] != '0'); }();
    if (wmma_on && bf16_on && ldx == 0 && T >= 16 && (beta == 0.0f || beta == 1.0f) &&
        strata_wmma_gemm_bf16(X, W, Y, T, N, K, ldy, beta, stream_)) {
        return;
    }
#endif
#if defined(__HIPCC__)
    if (f16_io_) {   // X is the FP16 image (set_act_f16); W goes through the dequantization scratch as FP16
        // the only accumulating BF16 product is the BF16X2 low part, which bf16x2_mode() turns off with this path
        if (beta != 0.0f) { std::fprintf(stderr, "prefill gemm: an accumulating BF16 GEMM on the FP16 prompt path\n"); std::exit(1); }
        const int64_t rows = std::min<int64_t>(N, scratch_elems_ / K);
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = std::min(rows, N - r0);
            bf16_to_f16_rows<<<1024, 256, 0, (cudaStream_t) stream_>>>(W + r0 * K, (__half*) scratch_, n * K);
            f16_inplace(X, scratch_, Y + r0, T, n, K, ldy);
        }
        return;
    }
#endif
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, X, W, Y, T, N, K, ldy,
                      beta, stream_, ldx)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt bf16");
        return;
    }
#endif
#if defined(__HIPCC__)
    // a padded X (STRATA_PF_PAD, ldx > K) keeps the native call: the widening copies contiguous rows
    if (ldx == 0 && rdna2_sgemm(X, W, Y, T, N, K, ldy, beta, true)) return;
#endif
#if !defined(__HIPCC__)
#if defined(STRATA_EXPERIMENTAL_SM60)
    constexpr bool kPascalPath = true;
#else
    constexpr bool kPascalPath = false;
#endif
    if (const int path = K > 0 ? bf16_path() : 0; (path == 1 && N > 1 && beta == 0.0f) || (path == 2 && kPascalPath)) {
        // FP16 (path 1) or FP32 (path 2, Pascal) copies of the weight and of X, converted in tiles.  Issue #1757: when
        // the buffers did not fit in the VRAM that was left (a stage of a layer split keeps ~100 MiB), this fell through
        // to the raw BF16 cuBLAS call, which a Pascal card answers with CUBLAS_STATUS_INTERNAL_ERROR (14).  Now the
        // weight is converted in tiles of N rows (the whole weight when it fits, as before) and X in slices; a card with
        // no room even for a tile says so.  A single output row stays cuBLAS's GEMV for path 1 (faster than the
        // conversions).  Every tile is a disjoint block of Y, so each gets the caller's beta.
        const int mult = path == 2 ? 2 : 1;   // 2-byte elements per converted value
        int64_t w_rows = N;
        int64_t x_rows = std::max<int64_t>(1, std::min<int64_t>(T, kXSliceElems / K));
        bool fit = false;
        for (;;) {
            if (grow(tc_w_, tc_w_elems_, mult * w_rows * K) && grow(tc_x_, tc_x_elems_, mult * x_rows * K)) { fit = true; break; }
            if (w_rows <= 64 && x_rows <= 64) break;
            w_rows = std::max<int64_t>(std::min<int64_t>(w_rows, 64), w_rows / 2);
            x_rows = std::max<int64_t>(std::min<int64_t>(x_rows, 64), x_rows / 2);
        }
        if (!fit && path == 2) {   // Pascal has no BF16 cuBLAS call to fall back to; Volta keeps its raw one below
            std::fprintf(stderr, "prefill gemm: no GPU memory left for the %s conversion buffers of a %lld x %lld BF16 weight "
                                 "(Pascal cards run the BF16 products through them); free VRAM on this card: raise "
                                 "--vram-reserve-mib, lower --max-context or --kv-resident, or give it fewer layers\n",
                         "FP32", (long long) N, (long long) K);
            std::exit(1);
        }
        if (fit) {
            for (int64_t n0 = 0; n0 < N; n0 += w_rows) {
                const int64_t nn = std::min<int64_t>(w_rows, N - n0);
                if (path == 2) {
#if defined(STRATA_EXPERIMENTAL_SM60)
                    // STRATA_DENSE_F32_RESIDENT: the whole weight's FP32 copy, made once (the same widening), read at
                    // this tile's rows; the same tiles and slices, so the same Sgemm calls
                    const float* const ws = w_resident && f32_resident_bf16()
                        ? shadow_get(W, N * K, (cudaStream_t) stream_, [&](float* dst) { bf16_to_f32(W, dst, N * K, (cudaStream_t) stream_); })
                        : nullptr;
                    float* const xf = reinterpret_cast<float*>(tc_x_);
                    const float* wf = reinterpret_cast<float*>(tc_w_);
                    if (ws != nullptr) wf = ws + n0 * K;
                    else bf16_to_f32(W + n0 * K, reinterpret_cast<float*>(tc_w_), nn * K, (cudaStream_t) stream_);
                    for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                        const int64_t n = std::min<int64_t>(x_rows, T - t0);
                        bf16_to_f32(X + t0 * K, xf, n * K, (cudaStream_t) stream_);
                        ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) nn, (int) n, (int) K, &alpha,
                                       wf, (int) K, xf, (int) K, &beta, Y + t0 * ldy + n0, (int) ldy),
                           "cublasSgemm (bf16 on Pascal)");
                    }
#endif
                } else {
                    bf16_to_f16(W + n0 * K, tc_w_, nn * K, (cudaStream_t) stream_);
                    for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                        const int64_t n = std::min<int64_t>(x_rows, T - t0);
                        if (x_f16) {   // already the converted values (the same slices: the same GEMMs)
                            f16(X + t0 * K, tc_w_, Y + t0 * ldy + n0, n, nn, K, ldy, beta);
                        } else {
                            bf16_to_f16(X + t0 * K, tc_x_, n * K, (cudaStream_t) stream_);
                            f16(tc_x_, tc_w_, Y + t0 * ldy + n0, n, nn, K, ldy, beta);
                        }
                    }
                }
            }
            return;
        }
    }
#endif
#if !defined(__HIPCC__)
    if (x_f16) {   // not the tensor-core path after all (one output row, or no room): X back to BF16, a slice at a time
        const int64_t x_rows = std::max<int64_t>(1, std::min<int64_t>(T, kXSliceElems / K));
        if (!grow(tc_x_, tc_x_elems_, x_rows * K)) {
            std::fprintf(stderr, "prefill gemm: no GPU memory for converting an FP16 image back to BF16\n");
            std::exit(1);
        }
        for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
            const int64_t n = std::min<int64_t>(x_rows, T - t0);
            f16_to_bf16_kernel<<<(unsigned) ((n * K + 255) / 256), 256, 0, (cudaStream_t) stream_>>>(
                reinterpret_cast<const __half*>(X + t0 * K), tc_x_, n * K);
            bf16(tc_x_, W, Y + t0 * ldy, n, N, K, ldy, beta, 0, false);
        }
        return;
    }
#endif
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) (ldx ? ldx : K), &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx");
}

#if defined(STRATA_EXPERIMENTAL_SM60)
// STRATA_PF_ROUTER_V2: see gemm.hpp.  Gemm::bf16's path-2 branch for two weights over one widened slice: the same slices (x_rows), the same cublasSgemm arguments per weight, the same
// widening kernel (bf16_to_f32) - a value is widened once instead of twice.  Taken only where bf16() would run each weight whole (the weights' and the slice's buffers fit), the router's
// FP32 copy is the resident one and the shared gate's is widened into tc_w_ as bf16() does for a non-resident weight.
bool Gemm::bf16_router_gate(const uint16_t* X, const uint16_t* X_lo, const uint16_t* Wr, bool r_resident, float* Yr, int64_t Nr, const uint16_t* Wg, float* Yg, int64_t T, int64_t K) {
    using namespace strata::prefill;
    if (!pf_router_v2()) return false;
    if (T <= 0 || Nr <= 0 || K <= 0 || bf16_path() != 2) { pf2_fallback(kPf2Router, kPf2Arch); return false; }
    if (!r_resident || !f32_resident_bf16()) { pf2_fallback(kPf2Router, kPf2Geometry); return false; }
    const int64_t x_rows = std::max<int64_t>(1, std::min<int64_t>(T, kXSliceElems / K));
    if (!grow(tc_w_, tc_w_elems_, 2 * std::max<int64_t>(Nr, 1) * K) || !grow(tc_x_, tc_x_elems_, 2 * x_rows * K)) { pf2_fallback(kPf2Router, kPf2Length); return false; }   // bf16() would tile: it runs
    const float alpha = 1.0f;
    const float* wr = shadow_get(Wr, Nr * K, (cudaStream_t) stream_, [&](float* dst) { bf16_to_f32(Wr, dst, Nr * K, (cudaStream_t) stream_); });
    if (wr == nullptr) { pf2_fallback(kPf2Router, kPf2Geometry); return false; }
    float* const wg = reinterpret_cast<float*>(tc_w_);
    float* const xf = reinterpret_cast<float*>(tc_x_);
    bf16_to_f32(Wg, wg, K, (cudaStream_t) stream_);   // the shared gate's weight, as bf16() widens a non-resident one into tc_w_
    for (int pass = 0; pass < (X_lo != nullptr ? 2 : 1); ++pass) {
        const uint16_t* Xp = pass == 0 ? X : X_lo;
        const float beta = pass == 0 ? 0.0f : 1.0f;
        for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
            const int64_t n = std::min<int64_t>(x_rows, T - t0);
            bf16_to_f32(Xp + t0 * K, xf, n * K, (cudaStream_t) stream_);
            ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) Nr, (int) n, (int) K, &alpha, wr, (int) K, xf, (int) K, &beta, Yr + t0 * Nr, (int) Nr),
               "cublasSgemm (bf16 on Pascal)");
            ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, 1, (int) n, (int) K, &alpha, wg, (int) K, xf, (int) K, &beta, Yg + t0, 1),
               "cublasSgemm (bf16 on Pascal)");
        }
    }
    pf2_launched(kPf2Router);
    return true;
}
// STRATA_PF_ROUTER_F16: see gemm.hpp.  The router weight's FP16 copy is hc_weight's (the BF16 weight is exact in FP16 inside the range; a weight outside it, or non-finite, is refused and the
// FP32 router runs); the product is the attn_k / attn_v shape (N 512, K 2560) on the hand-scheduled kernel, or the compiler's HFMA2 kernel (the same bits) for a call the hand-scheduled one does not cover.
bool Gemm::router_f16(const uint16_t* Xh, const uint16_t* Wr, float* Y, int64_t T, int64_t N, int64_t K, bool x_lo) {
    using namespace strata::prefill;
    if (!pf_router_f16()) return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (T <= 0 || N <= 0) return false;
    const cudaStream_t st = (cudaStream_t) stream_;
    const char* why = nullptr;
    if (current_cc() != 60) why = "not a GP100";
    else if (x_lo) why = "the BF16X2 low part (a second product)";
    else if (K <= 0 || K % 32 != 0) why = "K is not a multiple of 32";
    else if (N < 64) why = "N below 64";
    else if (T > INT_MAX || N > INT_MAX || K > INT_MAX) why = "a size past 2^31";
    else if (((uintptr_t) Xh & 15) != 0) why = "X not 16-byte aligned";
    else if (((uintptr_t) Wr & 15) != 0) why = "W not 16-byte aligned";
    HcW hw;
    if (why == nullptr) {
        hw = hc_weight(Wr, N, K, st, dev);
        if (hw.p == nullptr) why = hw.why != nullptr ? hw.why : "no FP16 copy";
    }
    if (why != nullptr) {
        std::lock_guard<std::mutex> lk(g_rf_mu);
        ++g_rf_left[dev][why];
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "prefill gemm: STRATA_PF_ROUTER_F16=1: the router's logits run as FP16 HFMA2 GEMMs on the layer input's FP16 image and the router weight's FP16 copy "
                             "(FP16 sums of 64 products widened to FP32, not bitwise the FP32 router's: a near-tied expert id can change)\n");
    }
    const char* r = "STRATA_PF_HFMA_SASS is off";
    int k = -1;
    if (sass_on()) r = strata::prefill::sass::dense(Xh, K, hw.p, Y, T, N, K, N, st, dev, &k);
    if (r == nullptr) {
        ++g_rf_sass[dev];
    } else {
        {
            std::lock_guard<std::mutex> lk(g_rf_mu);
            ++g_rf_cc[dev][r];
        }
        strata::prefill::hfma::gemm(Xh, K, hw.p, Y, T, N, K, N, st);
    }
    ++g_rf_taken[dev];
    return true;
}
void Gemm::router_f16_cmp(const int32_t* ids, const int32_t* ids_ref, const float* w, const float* w_ref, int64_t T, void* stream) {
    int dev = 0;
    if (T <= 0 || cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return; }
    rf_cmp_kernel<<<(unsigned) ((T + 127) / 128), 128, 0, (cudaStream_t) stream>>>(ids, ids_ref, w, w_ref, T);
    ++g_rf_chk[dev];
}
// STRATA_PF_PLE_F16: see gemm.hpp.  The image is the plain FP16 rounding of the dequantized rows (no scale: the counters say how far inside the range they are); the product is router_f16's - the weight's FP16 copy
// is hc_weight's, the hand-scheduled kernel where it covers the call (N 10240, K 2560, T >= 128), else the compiler's HFMA2 kernel with the same bits.
void Gemm::ple_f16_image(const float* x, uint16_t* y, int64_t n) {
    if (n <= 0) return;
    const int64_t n4 = n / 4;
    const unsigned blocks = (unsigned) std::min<int64_t>(4096, (n4 + 255) / 256);
    pl_image_kernel<<<blocks, 256, 0, (cudaStream_t) stream_>>>(reinterpret_cast<const float4*>(x), reinterpret_cast<uint2*>(y), n4);
}
bool Gemm::ple_f16(const uint16_t* Xh, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int prod) {
    using namespace strata::prefill;
    prod = prod == 0 ? 0 : 1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (T <= 0 || N <= 0) return false;
    const cudaStream_t st = (cudaStream_t) stream_;
    const char* why = nullptr;
    if (current_cc() != 60) why = "not a GP100";
    else if (K <= 0 || K % 32 != 0) why = "K is not a multiple of 32";
    else if (N < 64) why = "N below 64";
    else if (T > INT_MAX || N > INT_MAX || K > INT_MAX) why = "a size past 2^31";
    else if (((uintptr_t) Xh & 15) != 0) why = "X not 16-byte aligned";
    else if (((uintptr_t) W & 15) != 0) why = "W not 16-byte aligned";
    HcW hw;
    if (why == nullptr) {
        hw = hc_weight(W, N, K, st, dev);
        if (hw.p == nullptr) why = hw.why != nullptr ? hw.why : "no FP16 copy";
    }
    if (why != nullptr) {
        std::lock_guard<std::mutex> lk(g_pl_mu);
        ++g_pl_left[dev][std::string(prod == 0 ? "key: " : "value: ") + why];
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "prefill gemm: STRATA_PF_PLE_F16=1: the PLE block's key and value projections run as FP16 HFMA2 GEMMs on the FP16 image of the dequantized n-gram rows and the weights' resident FP16 copies "
                             "(FP16 sums of 64 products widened to FP32, not bitwise the BF16-rounded rows and cublasSgemm route's)\n");
    }
    {
        std::lock_guard<std::mutex> lk(g_pl_mu);
        g_pl_wmax[dev][prod] = hw.wmax;
    }
    const char* r = "STRATA_PF_HFMA_SASS is off";
    int k = -1;
    if (sass_on()) r = strata::prefill::sass::dense(Xh, K, hw.p, Y, T, N, K, N, st, dev, &k);
    if (r == nullptr) {
        ++g_pl_sass[dev][prod];
    } else {
        {
            std::lock_guard<std::mutex> lk(g_pl_mu);
            ++g_pl_cc[dev][std::string(prod == 0 ? "key: " : "value: ") + r];
        }
        strata::prefill::hfma::gemm(Xh, K, hw.p, Y, T, N, K, N, st);
    }
    ++g_pl_taken[dev][prod];
    return true;
}
#endif

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
#ifdef STRATA_USE_HIP
    // #313 (opt-in STRATA_WMMA_GEMM=1): RDNA3 / RDNA3.5 WMMA dense GEMM, before hipBLASLt (falls through when false)
    static const bool wmma_on = [] { const char* v = std::getenv("STRATA_WMMA_GEMM"); return v && v[0] != 0 && v[0] != '0'; }();
    static const bool pf_on = [] { const char* v = std::getenv("STRATA_PF_GEMM"); return v && v[0] == '1'; }();
    static const int64_t pf_min_t = [] { const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T"); return v ? (int64_t) std::atoll(v) : (int64_t) 0; }();
    if (pf_on && T >= pf_min_t && strata_pf_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream_)) return;   // S23: opt-in
    if (wmma_on && T >= 16 && (beta == 0.0f || beta == 1.0f) &&
        strata_wmma_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream_)) {
        return;
    }
#endif
#if defined(__HIPCC__)
    if (f16_io_ && beta == 0.0f) { f16_inplace(X, W, Y, T, N, K, ldy); return; }
#endif
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::f16, X, W, Y, T, N, K, ldy,
                      beta, stream_)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt f16");
        return;
    }
#endif
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
    // Pascal, STRATA_DENSE_F32=1: cublasSgemm on FP32 copies (dense_f32_on above: where it is bitwise the FP16 call)
    if (const int64_t min_t = beta == 0.0f && !no_f32_ ? dense_f32_min_t(N, K) : 0; min_t > 0 && T >= min_t && dense_f32_on()) {
        const int64_t n_sl = (T * K + kXSliceElems - 1) / kXSliceElems;   // even slices of at most 64 MiB as fp32
        const int64_t x_rows = (T + n_sl - 1) / n_sl;
        // grow() frees a buffer an earlier GEMM on this stream may still read
        if (tc_w_elems_ < 2 * N * K || tc_x_elems_ < 2 * x_rows * K) cudaStreamSynchronize((cudaStream_t) stream_);
        if (grow(tc_w_, tc_w_elems_, 2 * N * K) && grow(tc_x_, tc_x_elems_, 2 * x_rows * K)) {
            static bool said = false;
            if (!said) {
                said = true;
                std::fprintf(stderr, "prefill gemm: the dense projections' FP16 GEMMs run as cublasSgemm on FP32 copies "
                                     "(STRATA_DENSE_F32=1; gemm_f32_parity checks the bits against this cuBLAS)\n");
            }
            float* const wf = reinterpret_cast<float*>(tc_w_);
            float* const xf = reinterpret_cast<float*>(tc_x_);
            f16_to_f32(W, wf, N * K, (cudaStream_t) stream_);
            for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                const int64_t n = std::min<int64_t>(x_rows, T - t0);
                f16_to_f32(X + t0 * K, xf, n * K, (cudaStream_t) stream_);
                sgemm_f32(xf, wf, Y + t0 * ldy, n, N, K, ldy, beta);
            }
            return;
        }
        cudaGetLastError();   // no room for the copies: the FP16 call below
    }
#endif
#if defined(__HIPCC__)
    if (rdna2_sgemm(X, W, Y, T, N, K, ldy, beta, false)) return;
#endif
#if !defined(__HIPCC__)
    const auto f16_gemm = [&](cublasGemmAlgo_t algo) {
        return cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha,
                            W, CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                            CUBLAS_COMPUTE_32F, algo);
    };
    // Issue #1650: with a second CUDA device initialized in the process, cuBLAS 12 answers the default algorithm of a
    // few shapes - narrow ones, and the set moves with K - with CUBLAS_STATUS_INTERNAL_ERROR; a fixed algorithm keeps
    // a kernel that state does not break.  Only the shapes that fail are moved (the failing call itself is harmless);
    // the wide shapes, for which the fixed algorithms are markedly slower, stay on the default.
    const int64_t shape = ((int64_t) K << 32) | (uint32_t) N;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    for (int i = 0; i < f16_algo_n_; ++i)
        if (f16_algo_shape_[i] == shape) { algo = CUBLAS_GEMM_ALGO2; break; }
#if defined(STRATA_EXPERIMENTAL_SM60)
    // STRATA_GEMM_ALGO6: the shapes and chunk sizes whose ALGO6 call is this DEFAULT call's output, and faster
    if (algo == CUBLAS_GEMM_DEFAULT && gemm_algo6_mode() >= 1 && f16_algo6_t(N, K, T)) {
        static bool said = false;
        if (!said) {
            said = true;
            std::fprintf(stderr, "prefill gemm: the dense projections' FP16 products run with ALGO6 "
                                 "(STRATA_GEMM_ALGO6=%d; bitwise their DEFAULT calls at the shapes of the table)\n",
                         gemm_algo6_mode());
        }
        algo = CUBLAS_GEMM_ALGO6;
    }
#endif
    cublasStatus_t st = f16_gemm(algo);
    if (st == CUBLAS_STATUS_INTERNAL_ERROR && algo == CUBLAS_GEMM_ALGO6)   // (STRATA_GEMM_ALGO6: issue #1650's state)
        st = f16_gemm(algo = CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_INTERNAL_ERROR && algo == CUBLAS_GEMM_DEFAULT) {
        if (f16_algo_n_ < kF16AlgoSlots) f16_algo_shape_[f16_algo_n_++] = shape;
        st = f16_gemm(CUBLAS_GEMM_ALGO2);
    }
    ck(st, "cublasGemmEx f16");
#else
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
#endif
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx f16");
}

#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
bool Gemm::native_mmq(const uint16_t* X, int type, const void* W, float* Y, int64_t T, int64_t N, int64_t K,
                      int64_t ldy) {
    namespace mmq = strata::prefill::mmq;
    constexpr int64_t kRows = 1024, kMaxK = 8192;
    static const bool enabled = [] {
        const char* e = std::getenv("STRATA_DENSE_MMQ");
        return e && e[0] == '1';
    }();
    if (!enabled || mmq_failed_ || !mmq::built() || !mmq::fits(type, N)) return false;
    // K must be a multiple of 256: llama.cpp's MMQ loads the weights in 256-value K chunks, and the
    // chunk past a partial row reads past the row (the next row's bytes - or, for the last weight
    // row, bytes past the tensor, which no caller guarantees to be zeros). The MoE path is safe
    // because its gather buffers carry a zeroed tail; a dense tensor is passed as-is, so a K that
    // is not a full chunk multiple (e.g. the 640-value shared-expert down: 20 blocks of 32) stays on
    // the dequantize + cuBLAS path.
    if (K > kMaxK || K % 256 != 0) return false;
    int64_t R = scratch_elems_ / (2 * K);   // FP32 activations for a chunk fit in the FP16 dequant scratch
    if (R > kRows) R = kRows;
    if (R > T) R = T;
    if (R < 1) return false;
    if (ldy <= 0) ldy = N;

    const auto up = [](size_t v) { return (v + 255) & ~(size_t) 255; };
    const size_t ident_off = up(mmq::q8_bytes(kRows, kMaxK));
    const size_t bounds_off = ident_off + up((size_t) kRows * 4);
    if (!mmq_buf_) {
        if (cudaMalloc(&mmq_buf_, bounds_off + 256) != cudaSuccess) {
            cudaGetLastError();
            mmq_failed_ = true;
            return false;
        }
        mmq_ctx_ = new mmq::Context();
        mmq::iota((int32_t*) ((uint8_t*) mmq_buf_ + ident_off), kRows, stream_);
        std::fprintf(stderr, "prefill gemm: dense MMQ on (STRATA_DENSE_MMQ=1)\n");
    }
    void* xq = mmq_buf_;
    const int32_t* ident = (const int32_t*) ((uint8_t*) mmq_buf_ + ident_off);
    int32_t* bounds = (int32_t*) ((uint8_t*) mmq_buf_ + bounds_off);
    float* xf = (float*) scratch_;
    auto* ctx = static_cast<mmq::Context*>(mmq_ctx_);

    for (int64_t r0 = 0; r0 < T; r0 += R) {
        const int64_t rows = (T - r0 < R) ? T - r0 : R;
        mmq::f16_to_f32(X + r0 * K, xf, rows * K, stream_);
        mmq::quantize(xf, nullptr, xq, type, K, K, rows, stream_);
        mmq::set_bounds(bounds, (int32_t) rows, stream_);
        mmq::Product p;
        p.w = W;
        p.type = type;
        p.w_rows = N;
        p.w_cols = K;
        p.expert_bytes = mmq::matrix_bytes(type, N, K);
        p.n = 1;
        p.xq = xq;
        p.bounds = bounds;
        p.ids = ident;
        p.total_rows = rows;
        p.max_rows = rows;
        p.dst = Y + r0 * ldy;
        p.ld_dst = ldy;
        ctx->run(p, stream_);
    }
    return true;
}
#endif
void Gemm::f16_inplace(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {
#if defined(__HIPCC__)
    const float one = 1.0f, zero = 0.0f;
    bool done = false;
#if defined(STRATA_ROCBLAS_AVAILABLE)
    done = try_rocblas_f16o(rocblas_state_, X, W, Y, T, N, K, ldy);   // the table's kernel for this shape, if any
#endif
    if (!done) {
        ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, W,
                        CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &zero, Y, CUDA_R_16F, (int) (2 * ldy),
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
           "cublasGemmEx f16 out");
    }
    static const bool dbg_nan = std::getenv("STRATA_DBG_NAN") != nullptr;
    widen_rows_f16<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(Y, N, ldy, dbg_nan ? 1 : 0);
    if (dbg_nan) {   // debug only: a sync per GEMM
        unsigned long long bad = 0, zero_count = 0;
        (void) hipStreamSynchronize((hipStream_t) stream_);
        if (hipMemcpyFromSymbol(&bad, HIP_SYMBOL(g_f16_nonfinite), sizeof bad) == hipSuccess && bad != 0) {
            std::fprintf(stderr, "strata prefill: STRATA_DBG_NAN: %llu non-finite FP16 GEMM outputs (T=%lld N=%lld K=%lld; FP16 ends at 65504)%s",
                         bad, (long long) T, (long long) N, (long long) K, "\n");
            (void) hipMemcpyToSymbol(HIP_SYMBOL(g_f16_nonfinite), &zero_count, sizeof zero_count);
        }
    }
#else
    (void) X; (void) W; (void) Y; (void) T; (void) N; (void) K; (void) ldy;
#endif
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta, int64_t ldx) {
#ifdef STRATA_USE_HIP
    // S23 (opt-in STRATA_PF_PAD=1, with STRATA_PF_GEMM=1 on chunks of STRATA_PF_SWITCH_MIN_T+ tokens): the weight
    // dequantized into the scratch with row stride K + 64 halves, so no 4 KB-multiple stride camps on the memory
    // channels (gemm_probe8: K 6144 19 -> 29 TFLOPS from W alone, 35 with X padded too).  Bitwise the same products.
    static const bool pad_on = [] { const char* v = std::getenv("STRATA_PF_PAD"); return v && v[0] == '1'; }();
    static const bool pf_on = [] { const char* v = std::getenv("STRATA_PF_GEMM"); return v && v[0] == '1'; }();
    static const int64_t pf_min_t = [] { const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T"); return v ? (int64_t) std::atoll(v) : (int64_t) 0; }();
    // (strata_pf_padded_gemm_available(): the CURRENT device runs the padded gfx11 product.  On a gfx1151 + gfx1201 split the process-wide switch
    // is on for both stages; the gfx12 stage would dequantize the weight padded, find no product, and do it again unpadded below.)
    if (pad_on && pf_on && T >= pf_min_t && T >= 64 && N >= 512 && K % 32 == 0 && N * (K + 64) <= scratch_elems_ &&
        strata_pf_padded_gemm_available()) {
        const int64_t ldw = K + 64;
        if (strata::kernels::dequant_f16_ld(ggml_type, W_blocks, 0, N, K, ldw, scratch_, stream_) &&
            strata_pf_gemm_f16_ld(X, ldx > 0 ? ldx : K, scratch_, ldw, Y, T, N, K, ldy, beta, stream_)) {
            return;
        }
    }
#endif
    if (ldx > 0 && ldx != K) {
#ifdef STRATA_USE_HIP
        // X arrived padded but the weight cannot be (an i-quant without a strided dequant, or the scratch is too
        // small for N x (K + 64)): the weight unpadded, X read at its stride - the same products, the same bits
        const int64_t rows = scratch_elems_ / K;
        bool ok = rows > 0 && strata_pf_padded_gemm_available();
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; ok && r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            ok = strata_pf_gemm_f16_ld(X, ldx, scratch_, K, Y + r0, T, n, K, ldy, beta, stream_);
        }
        if (ok) return;
#endif
        // Nothing took a padded X: the padded product is the gfx11 matrix cores' (STRATA_PF_PAD asks
        // strata_pf_padded_gemm_available() before padding now, so this is a stale producer or another card's
        // activation).  Copy the rows unpadded and run the ordinary paths - the same products, the same bits - rather
        // than ending the engine.
        const uint16_t* Xu = unpad_x(X, T, K, ldx);
        if (Xu != nullptr) {
            native(Xu, ggml_type, W_blocks, Y, T, N, K, ldy, beta, 0);
            return;
        }
        std::fprintf(stderr, "prefill gemm: a padded X (ldx %lld, K %lld) has no path on this card and could not be "
                             "copied unpadded (no device memory)\n", (long long) ldx, (long long) K);
        std::exit(1);
    }
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
    if (beta == 0.0f && (ldx == 0 || ldx == K) && native_mmq(X, ggml_type, W_blocks, Y, T, N, K, ldy)) return;
#endif
#if !defined(__HIPCC__) && defined(STRATA_EXPERIMENTAL_SM60)
    // STRATA_DENSE_F32=1 + STRATA_DENSE_F32_RESIDENT=1: the shapes f16() would run as cublasSgemm on FP32 copies (the
    // whole weight in the scratch: as below) read the weight's resident FP32 copy - the same dequantized FP16 values
    // widened, made once - instead of dequantizing and widening it every chunk.  No copy (no room): the FP16 call.
    if (beta == 0.0f && N * K <= scratch_elems_ && f32_resident_on()) {
        if (const int64_t min_t = dense_f32_min_t(N, K); min_t > 0 && T >= min_t && dense_f32_on()) {
            if (ldy <= 0) ldy = N;
            // STRATA_GEMM_ALGO6=2: the weight's transposed copy (no room: the untransposed route below, which
            // runs =1's call)
            if (gemm_algo6_mode() >= 2 && f32_algo6_t(N, K, T)) {
                const float* wt = shadow_get_in(g_shadow_t, W_blocks, N * K, (cudaStream_t) stream_, [&](float* dst) {
                    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
                    f16_transpose_w(scratch_, dst, N, K, (cudaStream_t) stream_);
                }, "STRATA_GEMM_ALGO6 (transposed FP32 copy)");
                if (wt != nullptr && sgemm_nt(X, wt, Y, T, N, K, ldy, beta)) return;
            }
            const float* wf = shadow_get(W_blocks, N * K, (cudaStream_t) stream_, [&](float* dst) {
                strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
                f16_to_f32(scratch_, dst, N * K, (cudaStream_t) stream_);
            });
            if (wf != nullptr && sgemm_f32w(X, wf, Y, T, N, K, ldy, beta)) return;
            strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
            no_f32_ = true;
            f16(X, scratch_, Y, T, N, K, ldy, beta);
            no_f32_ = false;
            return;
        }
    }
#endif
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        // the side stream's next dequant into scratch_ waits for these slices' reads and writes of it
        if (side_scratch_) (void) cudaEventRecord((cudaEvent_t) side_free_[0], (cudaStream_t) stream_);
        return;
    }
    if (side_scratch_) {
        // STRATA_DEQUANT_SIDE: the dequant on the side stream, into the scratch the compute stream read last but one.
        // free[cur] -> dequant -> ready[cur] -> GEMM -> free[cur] orders the dequant after the GEMM that last read
        // this scratch and the GEMM after the dequant that filled it, so every read and write of a scratch is where
        // the serial form had it - the same kernels, the same bits - and the dequant runs while the kernels queued
        // ahead of this GEMM do.
        uint16_t* const s = side_cur_ == 0 ? scratch_ : side_scratch_;
        (void) cudaStreamWaitEvent((cudaStream_t) side_stream_, (cudaEvent_t) side_free_[side_cur_], 0);
        strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, s, side_stream_);
        (void) cudaEventRecord((cudaEvent_t) side_ready_[side_cur_], (cudaStream_t) side_stream_);
        (void) cudaStreamWaitEvent((cudaStream_t) stream_, (cudaEvent_t) side_ready_[side_cur_], 0);
        f16(X, s, Y, T, N, K, ldy, beta);
        (void) cudaEventRecord((cudaEvent_t) side_free_[side_cur_], (cudaStream_t) stream_);
        side_cur_ ^= 1;
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
