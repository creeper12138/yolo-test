// ghostconv_kernel_split.cu

#include "ghostconv_kernel.h"
#include <cuda_runtime.h>

#define CHECK_CUDA(err) do { if ((err) != cudaSuccess) return; } while(0)

constexpr int TILE_H = 8;
constexpr int TILE_W = 8;
constexpr int K2     = 5;
constexpr int PAD2   = K2/2;

// ---------------------------------------------
// kernel 实现：同之前的 shared‐memory 版本
// ---------------------------------------------
__global__ void primary_conv_tiled(
    const float* __restrict__ input,
    const float* __restrict__ w1,
    float*       __restrict__ tmp,
    int N, int C_in, int H, int W,
    int C_mid, int k, int stride, int pad,
    int H_out, int W_out)
{
    extern __shared__ float sh_w1[];
    int cm = blockIdx.y, n = blockIdx.z;
    int th = threadIdx.y, tw = threadIdx.x;
    int tid = th*blockDim.x + tw, nthreads = blockDim.x*blockDim.y;

    // load w1[channel=cm] → shared
    int sz_w1 = C_in * k * k;
    for (int idx = tid; idx < sz_w1; idx += nthreads)
        sh_w1[idx] = w1[cm*sz_w1 + idx];
    __syncthreads();

    // compute output tile
    int tile_i = blockIdx.x * TILE_H;
    int h = tile_i + th, w = blockIdx.x * TILE_W + tw;
    if (h < H_out && w < W_out) {
        float sum = 0;
        for (int ci = 0; ci < C_in; ++ci)
            for (int yy = 0; yy < k; ++yy) {
                int ih = h*stride + yy - pad;
                if (ih<0||ih>=H) continue;
                for (int xx = 0; xx < k; ++xx) {
                    int iw = w*stride + xx - pad;
                    if (iw<0||iw>=W) continue;
                    int in_idx = ((n*C_in+ci)*H+ih)*W + iw;
                    int wt_idx = ci*(k*k) + yy*k + xx;
                    sum += input[in_idx] * sh_w1[wt_idx];
                }
            }
        tmp[((n*C_mid+cm)*H_out + h)*W_out + w] = sum;
    }
}

__global__ void cheap_dwconv_tiled(
    const float* __restrict__ tmp,
    const float* __restrict__ w2,
    float*       __restrict__ tmp2,
    int N, int C_mid, int H_out, int W_out)
{
    extern __shared__ float sh_w2[];
    int cm = blockIdx.y, n = blockIdx.z;
    int th = threadIdx.y, tw = threadIdx.x;
    int tid = th*blockDim.x + tw, nthreads = blockDim.x*blockDim.y;

    // load w2[channel=cm] → shared
    int sz_w2 = K2 * K2;
    for (int idx = tid; idx < sz_w2; idx += nthreads)
        sh_w2[idx] = w2[cm*sz_w2 + idx];
    __syncthreads();

    // compute depthwise tile
    int tile_i = blockIdx.x * TILE_H;
    int h = tile_i + th, w = blockIdx.x * TILE_W + tw;
    if (h < H_out && w < W_out) {
        float sum = 0;
        for (int yy = 0; yy < K2; ++yy) {
            int ih = h + yy - PAD2;
            if (ih<0||ih>=H_out) continue;
            for (int xx = 0; xx < K2; ++xx) {
                int iw = w + xx - PAD2;
                if (iw<0||iw>=W_out) continue;
                int in_idx = ((n*C_mid+cm)*H_out + ih)*W_out + iw;
                int wt_idx = yy*K2 + xx;
                sum += tmp[in_idx] * sh_w2[wt_idx];
            }
        }
        tmp2[((n*C_mid+cm)*H_out + h)*W_out + w] = sum;
    }
}

__global__ void concat_feature(
    const float* __restrict__ tmp,
    const float* __restrict__ tmp2,
    float*       __restrict__ out,
    int N, int C_mid, int H_out, int W_out)
{
    int cm2 = blockIdx.y, n = blockIdx.z;
    int th = threadIdx.y, tw = threadIdx.x;
    int tile_i = blockIdx.x * TILE_H;
    int h = tile_i + th, w = blockIdx.x * TILE_W + tw;
    if (h < H_out && w < W_out) {
        int base = ((n*(2*C_mid)+cm2)*H_out + h)*W_out + w;
        if (cm2 < C_mid)
            out[base] = tmp[((n*C_mid+cm2)*H_out + h)*W_out + w];
        else {
            int c2 = cm2 - C_mid;
            out[base] = tmp2[((n*C_mid+c2)*H_out + h)*W_out + w];
        }
    }
}

// ---------------------------------------------
// 三个接口的实现
// ---------------------------------------------
extern "C" void launch_primary_conv(
    const float* input, const float* w1, float* tmp,
    int N,int C_in,int H,int W,
    int C_mid,int k,int stride,int pad,
    int H_out,int W_out,
    cudaStream_t stream)
{
    dim3 block(TILE_W,TILE_H);
    int grid_x = (H_out + TILE_H - 1)/TILE_H;
    dim3 grid(grid_x, C_mid, N);
    size_t shm = size_t(C_in)*k*k*sizeof(float);
    primary_conv_tiled<<<grid,block,shm,stream>>>(
      input,w1,tmp,
      N,C_in,H,W,
      C_mid,k,stride,pad,
      H_out,W_out);
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void launch_depthwise_conv(
    const float* tmp, const float* w2, float* tmp2,
    int N,int C_mid,int H_out,int W_out,
    cudaStream_t stream)
{
    dim3 block(TILE_W,TILE_H);
    int grid_x = (H_out + TILE_H - 1)/TILE_H;
    dim3 grid(grid_x, C_mid, N);
    size_t shm = size_t(K2)*K2*sizeof(float);
    cheap_dwconv_tiled<<<grid,block,shm,stream>>>(
      tmp,w2,tmp2,
      N,C_mid,H_out,W_out);
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void launch_concat(
    const float* tmp,const float* tmp2, float* out,
    int N,int C_mid,int H_out,int W_out,
    cudaStream_t stream)
{
    dim3 block(TILE_W,TILE_H);
    int grid_x = (H_out + TILE_H - 1)/TILE_H;
    dim3 grid(grid_x, 2*C_mid, N);
    concat_feature<<<grid,block,0,stream>>>(
      tmp,tmp2,out,
      N,C_mid,H_out,W_out);
    CHECK_CUDA(cudaGetLastError());
}
