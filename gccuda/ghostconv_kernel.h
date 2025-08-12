#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void launch_ghconv(
    const float* input,
    const float* weight,
    float*       output,
    int          N,
    int          C_in,
    int          H,
    int          W,
    int          C_out,
    int          kH,
    int          kW,
    int          H_out,
    int          W_out,
    int          stride,
    int          pad,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
