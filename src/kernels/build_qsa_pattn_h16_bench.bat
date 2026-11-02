@echo off
rem build_qsa_pattn_h16_bench.bat [outdir]: qsa_pattn_h16_bench.exe (includes qsa_pattn.cu; sm_60 only), ptxas -v log in <outdir>\ptxas.log
setlocal
set OUT=%~1
if "%OUT%"=="" set OUT=%~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0..\..
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" -O3 -arch=sm_60 -Wno-deprecated-gpu-targets -std=c++20 -DSTRATA_EXPERIMENTAL_SM60=1 %NVCC_EXTRA% -Xptxas -v -I include -o "%OUT%\qsa_pattn_h16_bench.exe" src\kernels\qsa_pattn_h16_bench.cu > "%OUT%\ptxas.log" 2>&1
echo nvcc exit %ERRORLEVEL%
