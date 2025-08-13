@echo off
setlocal

call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

REM 1. 设置路径
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set TRT_ROOT=D:\TensorRT-8.6.1.6
set MSVC_PATH="C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"

set PATH=%CUDA_PATH%\bin;%MSVC_PATH%;%PATH%

REM 2. 编译 CUDA 内核
nvcc -c ghostconv_kernel.cu -o ghostconv_kernel.obj -Xcompiler "/MD /EHsc" -I"%TRT_ROOT%\include"
nvcc -c ghostconv_kernel_fused.cu -o ghostconv_kernel_fused.obj -Xcompiler "/MD /EHsc" -I"%TRT_ROOT%\include"

REM 3. 编译 C++ 插件
cl /c GhostConvPlugin.cpp /I"%TRT_ROOT%\include" /I"%CUDA_PATH%\include" /MD /EHsc
cl /c GhostConvPluginCreator.cpp /I"%TRT_ROOT%\include" /I"%CUDA_PATH%\include" /MD /EHsc

REM 4. 链接生成 DLL
link /DLL /OUT:GhostConvPlugin.dll ghostconv_kernel.obj ghostconv_kernel_fused.obj GhostConvPlugin.obj GhostConvPluginCreator.obj ^
    /LIBPATH:"%TRT_ROOT%\lib" nvinfer.lib nvonnxparser.lib cudart.lib

echo 编译完成：GhostConvPlugin.dll
pause
