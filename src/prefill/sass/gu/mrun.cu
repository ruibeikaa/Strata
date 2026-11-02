// mrun.cu - runs micro kernels  void k(float* out, int iters)  of a cubin on one PH402 die and prints the HFMA2 rate.
//   mrun <cubin> name:ctas_per_sm:threads:iters:hfma2_per_iter [...]
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define CKD(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* m_ = nullptr; cuGetErrorString(r_, &m_); std::fprintf(stderr, "driver error %s at %d: %s\n", #x, __LINE__, m_ ? m_ : "?"); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    if (argc < 3) return 2;
    cudaSetDevice(0); cudaFree(0);
    cudaDeviceProp P{}; cudaGetDeviceProperties(&P, 0);
    std::printf("device 0 = %s  cc %d.%d  %d SMs\n", P.name, P.major, P.minor, P.multiProcessorCount);
    if (P.major != 6 || std::strstr(P.name, "PH402") == nullptr) { std::fprintf(stderr, "NOT a PH402 die: refusing to run\n"); return 3; }
    CUmodule mod; CKD(cuModuleLoad(&mod, argv[1]));
    CUdeviceptr dO; CKD(cuMemAlloc(&dO, 64 << 20));
    for (int a = 2; a < argc; ++a) {
        char name[128]; int cps, thr, iters; double hper;
        if (std::sscanf(argv[a], "%127[^:]:%d:%d:%d:%lf", name, &cps, &thr, &iters, &hper) != 5) { std::fprintf(stderr, "bad spec %s\n", argv[a]); return 2; }
        CUfunction fn; CKD(cuModuleGetFunction(&fn, mod, name));
        int regs = 0, smem = 0; CKD(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn)); CKD(cuFuncGetAttribute(&smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn));
        int occ = 0; CKD(cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, thr, 0));
        const int grid = P.multiProcessorCount * cps * 8;
        if (std::getenv("SHOW")) CKD(cuMemsetD32(dO, 0xDEADBEEFu, 1024));   // 8 waves
        int it = iters; void* params[] = {&dO, &it};
        auto launch = [&] { CKD(cuLaunchKernel(fn, grid, 1, 1, thr, 1, 1, 0, 0, params, 0)); };
        launch(); launch(); cudaDeviceSynchronize();
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        std::vector<float> ms;
        for (int r = 0; r < 5; ++r) { cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1); float m; cudaEventElapsedTime(&m, e0, e1); ms.push_back(m); }
        std::sort(ms.begin(), ms.end());
        const double macs = (double) grid * thr * (double) iters * hper * 2.0;
        const double g = macs / (ms[2] * 1e-3) / 1e9;
        if (std::getenv("SHOW")) { unsigned h[8]; CKD(cuMemcpyDtoH(h, dO, sizeof h)); std::printf("   out[0..3] = %08x %08x %08x %08x\n", h[0], h[1], h[2], h[3]); }
        std::printf("%-26s regs %3d smem %5d occ %d/SM(asked %d) | %8.3f ms | %8.1f G MAC/s = %5.1f%% of 6451\n", name, regs, smem, occ, cps, ms[2], g, 100.0 * g / 6451.2);
    }
    return 0;
}
