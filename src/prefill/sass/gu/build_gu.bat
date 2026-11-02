@echo off
rem build_gu.bat: gen_gu.py's containers -> nvcc -cubin -> CuAssembler disassembly -> hand-scheduled gate/up kernels -> gu_hand.cubin
rem (NOT embedded in the engine: see gen_gu.py's head).  CUASSEMBLER: its checkout (default: the campaign copy).
setlocal
if "%CUASSEMBLER%"=="" set CUASSEMBLER=%~dp0cuassembler
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%
set PYTHONPATH=%CUASSEMBLER%
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
python gen_gu.py containers || exit /b 1
nvcc -cubin -O3 -arch=sm_60 -Wno-deprecated-gpu-targets -o containers_gu.cubin containers_gu.cu || exit /b 1
python -W ignore %CUASSEMBLER%\bin\cuasm.py containers_gu.cubin -o containers_gu.cuasm -f gu-log-dis -q > nul 2>&1 || exit /b 2
python gen_gu.py gen containers_gu.cuasm gu_hand.cuasm || exit /b 3
python -W ignore %CUASSEMBLER%\bin\cuasm.py gu_hand.cuasm -o gu_hand.cubin -f gu-log-asm -q > gu_asm.out 2>&1 || (type gu_asm.out & exit /b 4)
echo BUILD_GU_OK
