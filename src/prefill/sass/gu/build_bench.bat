@echo off
rem build_bench.bat: gu_bench.cu (REF = ../../moe_ws_hf.cuh as compiled by the engine, MOD = gu_hand.cubin).  Run on a PH402 die only:
rem   gu_bench gu_hand.cubin <fam 0..3> <tier 16|32> <ntok> <experts> [reps] [wild] [tail] [full] [noref]     (see the head of gu_bench.cu)
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" -O3 -arch=sm_60 -use_fast_math -Wno-deprecated-gpu-targets -std=c++20 -I %~dp0.. -I %~dp0..\..\..\..\include -Xcompiler "/openmp /O2 /EHsc" -o gu_bench.exe gu_bench.cu -lcuda
