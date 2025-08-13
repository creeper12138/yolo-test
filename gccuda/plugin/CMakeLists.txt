cmake_minimum_required(VERSION 3.20)
project(ghostconv_trt LANGUAGES CXX CUDA)

# 版本与标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)

# TensorRT
set(TRT_ROOT "D:/TensorRT-8.6.1.6" CACHE PATH "TensorRT root directory")
find_path(TRT_INCLUDE_DIR NvInfer.h HINTS ${TRT_ROOT}/include)
find_library(TRT_LIB_nvinfer nvinfer HINTS ${TRT_ROOT}/lib ${TRT_ROOT}/lib/x64)
if (NOT TRT_INCLUDE_DIR OR NOT TRT_LIB_nvinfer)
  message(FATAL_ERROR "TensorRT not found. Set -DTRT_ROOT to a folder containing include/ and lib/")
endif()

# CUDA 工具包（给 cl.exe 提供 cuda_runtime*.h、并用 CUDA::cudart 链接）
find_package(CUDAToolkit REQUIRED)

# 源文件（按你当前 4 个）
set(SOURCES
  ghostconv_api_adapter.cpp
  GhostConvPlugin.cpp
  ghostconv_kernel_fused.cu
)

# 生成共享库（单库双用途）
add_library(ghostconv_trt SHARED ${SOURCES})

# 编译/导出设置
target_compile_definitions(ghostconv_trt PRIVATE GC_BUILD_SHARED)
target_compile_options(ghostconv_trt PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)

# 头文件路径（顺序现在在 add_library 之后）
target_include_directories(ghostconv_trt PRIVATE
  ${TRT_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# 目标 GPU 架构（按需改）
set_target_properties(ghostconv_trt PROPERTIES CUDA_ARCHITECTURES "75;86")

# 链接（使用导入目标更稳）
target_link_libraries(ghostconv_trt PRIVATE
  ${TRT_LIB_nvinfer}
  CUDA::cudart
)
