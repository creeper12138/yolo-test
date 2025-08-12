// ghostconv_kernel.h
#pragma once
#include <cuda_runtime.h>

#ifdef _WIN32
  #define DLL_EXPORT __declspec(dllexport)
#else
  #define DLL_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Stage 1: 主卷积 3×3
///   input:  [N, C_in,  H,  W ]
///   w1:     [C_mid, C_in, k, k]
///   tmp:    [N, C_mid, H_out, W_out]
DLL_EXPORT void launch_primary_conv(
    const float* input,
    const float* w1,
    float*       tmp,
    int          N,
    int          C_in,
    int          H,
    int          W,
    int          C_mid,
    int          k,       // 主卷积核大小
    int          stride,
    int          pad,
    int          H_out,
    int          W_out,
    cudaStream_t stream);

/// Stage 2: 深度可分离卷积 5×5
///   tmp:    [N, C_mid, H_out, W_out]
///   w2:     [C_mid,   1,  5, 5]
///   tmp2:   [N, C_mid, H_out, W_out]
DLL_EXPORT void launch_depthwise_conv(
    const float* tmp,
    const float* w2,
    float*       tmp2,
    int          N,
    int          C_mid,
    int          H_out,
    int          W_out,
    cudaStream_t stream);

/// Stage 3: 拼接 tmp/tmp2 → output
///   tmp:    [N, C_mid,   H_out, W_out]
///   tmp2:   [N, C_mid,   H_out, W_out]
///   output: [N, 2*C_mid, H_out, W_out]
DLL_EXPORT void launch_concat(
    const float* tmp,
    const float* tmp2,
    float*       output,
    int          N,
    int          C_mid,
    int          H_out,
    int          W_out,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
