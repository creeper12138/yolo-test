// ghostconv_kernel.h  —— 内部核声明（float 专用）
#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// 与你 fused 实现一致的原型（顺序/类型保持不变）
void launch_ghostconv(
    const float* input, const float* w1, const float* w2, float* output,
    int N, int C_in, int H, int W, int C_mid,
    int k1, int pad1, int stride, int H_out, int W_out, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
