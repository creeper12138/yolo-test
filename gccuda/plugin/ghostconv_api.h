#pragma once
#include <cuda_runtime.h>
#include <stddef.h>
#include "ghostconv_exports.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GC_DTYPE_F32 = 0,
    GC_DTYPE_F16 = 1
} gc_dtype_t;

GC_API void ghostconv_get_output_shape(
    int H, int W, int k, int stride, int pad,
    int* H_out, int* W_out
);

GC_API size_t ghostconv_workspace_size(
    int N, int C_in, int H, int W,
    int C_mid, int k, int stride, int pad,
    int H_out, int W_out,
    gc_dtype_t dtype
);

// 注意：对外导出的统一入口改名为 ghostconv_launch
GC_API bool ghostconv_launch(
    const void* x, void* y,
    const void* w1, const void* w2,
    int N, int C_in, int H, int W,
    int C_mid, int k, int stride, int pad,
    int H_out, int W_out,
    gc_dtype_t dtype,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
