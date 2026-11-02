@echo off
rem build_hand.bat [kernel names]: regenerates the hand-scheduled HFMA2 GEMM cubin of gemm_sass.cuh and embeds it.
rem   mk_containers.py -> containers.cu (empty kernels with the real signatures) -> nvcc -cubin -> CuAssembler (cloudcores/CuAssembler)
rem   disassembles it -> gen_hfma.py replaces each kernel's instruction stream -> CuAssembler assembles -> hand.cubin ->
rem   embed_cubin.py -> gemm_sass_cubin.inc (what gemm_sass.cuh includes).
rem CUASSEMBLER: its checkout (default: the campaign copy).  Needs CUDA 12.x's nvcc + MSVC, Python 3 (with the CuAssembler requirements).
setlocal
if "%CUASSEMBLER%"=="" set CUASSEMBLER=%~dp0cuassembler
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%
set PYTHONPATH=%CUASSEMBLER%
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
python mk_containers.py || exit /b 1
nvcc -cubin -O3 -arch=sm_60 -Wno-deprecated-gpu-targets -o containers.cubin containers.cu || exit /b 1
python -W ignore %CUASSEMBLER%\bin\cuasm.py containers.cubin -o containers.cuasm -f cuasm-log-dis -q > nul 2>&1 || exit /b 2
python gen_hfma.py containers.cuasm hand.cuasm %* || exit /b 3
python -W ignore %CUASSEMBLER%\bin\cuasm.py hand.cuasm -o hand.cubin -f cuasm-log-asm -q > asm.out 2>&1 || (type asm.out & exit /b 4)
python embed_cubin.py hand.cubin gemm_sass_cubin.inc || exit /b 5
echo BUILD_HAND_OK
