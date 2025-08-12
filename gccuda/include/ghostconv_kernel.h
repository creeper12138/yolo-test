#ifndef GHOSTCONV_KERNEL_H
#define GHOSTCONV_KERNEL_H

#include <cuda_runtime.h>

void launch_ghconv(
	const float* input, const float* weight, float* output,
	int N, int C_in, int H, int W,
	int C_out, int kH, int kW,
	int H_out, int W_out,
	int stride, int pad,
	cudaStream_t stream = 0);

#endif
