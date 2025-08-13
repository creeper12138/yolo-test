#include "ghostconv_api.h"

// 这是 .cu 里已有的 float 版 fused 实现（保留原名）
extern "C" void launch_ghostconv(
    const float* input, const float* w1, const float* w2, float* output,
    int N, int C_in, int H, int W, int C_mid,
    int k1, int pad1, int stride, int H_out, int W_out, cudaStream_t stream);

// 形状工具
GC_API void ghostconv_get_output_shape(int H, int W, int k, int stride, int pad, int* Ho, int* Wo) {
    if (Ho) *Ho = (H + 2*pad - k) / stride + 1;
    if (Wo) *Wo = (W + 2*pad - k) / stride + 1;
}

// 你当前内核不需要外部 workspace，先返回 0 即可
GC_API size_t ghostconv_workspace_size(
    int, int, int, int, int, int, int, int, int, int, gc_dtype_t) {
    return 0;
}

// 对外统一入口（改名）：内部转调 float 版内核
GC_API bool ghostconv_launch(
    const void* x, void* y, const void* w1, const void* w2,
    int N, int C_in, int H, int W,
    int C_mid, int k, int stride, int pad,
    int H_out, int W_out,
    gc_dtype_t dtype,
    void*, size_t, cudaStream_t stream)
{
    if (dtype != GC_DTYPE_F32) {
        // 目前只支持 FP32
        return false;
    }
    launch_ghostconv(
        static_cast<const float*>(x),
        static_cast<const float*>(w1),
        static_cast<const float*>(w2),
        static_cast<float*>(y),
        N, C_in, H, W, C_mid,
        k, pad, stride, H_out, W_out, stream
    );
    return true;
}
