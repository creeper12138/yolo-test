// ghostconv_kernel_fused.cu

#include "ghostconv_kernel.h"
#include <cuda_runtime.h>
#define CHECK_CUDA(err) do { if((err)!=cudaSuccess) return; } while(0)

/// Tile 参数：每个 block 负责 TILE_H×TILE_W 的输出
constexpr int TILE_H = 8;
constexpr int TILE_W = 8;
/// depthwise 核大小
constexpr int K2   = 5;
constexpr int PAD2 = K2/2;

__global__ void ghostconv_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    float*       __restrict__ out,
    int N, int C_in, int H, int W,
    int C_mid, int k1, int pad1, int stride,
    int H_out, int W_out)
{
    // 计算当前 block 处理的批 + 通道
    int block_idx  = blockIdx.z;
    int cm = block_idx % C_mid;        // output channel idx
    int n  = block_idx / C_mid;        // batch idx

    // 计算 tile 在输出空间的起点 (h0, w0)
    int tile_i = blockIdx.x * TILE_H;
    int tile_j = blockIdx.y * TILE_W;

    // 线程在 block 内的位置 (0…TILE_H), (0…TILE_W)
    int th = threadIdx.y;
    int tw = threadIdx.x;

    // 共享内存布局： [ w1 | w2 | in_patch | prim_patch ]
    extern __shared__ float shm[];
    // 1) w1 大小: C_in * k1 * k1
    int sz_w1     = C_in * k1 * k1;
    float* sh_w1  = shm;
    // 2) w2 大小: K2*K2
    int sz_w2     = K2 * K2;
    float* sh_w2  = sh_w1 + sz_w1;
    // 3) 输入 patch 大小: C_in * (TILE_H*stride + k1-1) * (TILE_W*stride + k1-1)
    int patch_h   = TILE_H*stride + (k1 - 1);
    int patch_w   = TILE_W*stride + (k1 - 1);
    int sz_in     = C_in * patch_h * patch_w;
    float* sh_in  = sh_w2 + sz_w2;
    // 4) prim_patch 大小: (TILE_H + 2*PAD2) * (TILE_W + 2*PAD2)
    int ext_h     = TILE_H + 2*PAD2;
    int ext_w     = TILE_W + 2*PAD2;
    int sz_prim   = ext_h * ext_w;
    float* sh_prim= sh_in + sz_in;

    int tid = th * blockDim.x + tw;
    int nthreads = blockDim.x * blockDim.y;

    // --------------------------------------------------------
    // 加载 w1[channel=cm] 和 w2[channel=cm] 到 shared
    // --------------------------------------------------------
    for (int idx = tid; idx < sz_w1; idx += nthreads) {
        sh_w1[idx] = w1[cm * sz_w1 + idx];
    }
    for (int idx = tid; idx < sz_w2; idx += nthreads) {
        sh_w2[idx] = w2[cm * sz_w2 + idx];
    }
    __syncthreads();

    // --------------------------------------------------------
    // 加载输入 patch 到 shared (含 k1-1 的边界)
    // --------------------------------------------------------
    for (int idx = tid; idx < sz_in; idx += nthreads) {
        // 把 idx 展开为 (ci, ph, pw)
        int tmp = idx;
        int pw  = tmp % patch_w; tmp /= patch_w;
        int ph  = tmp % patch_h; tmp /= patch_h;
        int ci  = tmp;
        // 对应全局坐标
        int h_in = tile_i*stride + ph - pad1;
        int w_in = tile_j*stride + pw - pad1;
        float v = 0.0f;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            int in_idx = ((n*C_in + ci)*H + h_in)*W + w_in;
            v = input[in_idx];
        }
        sh_in[idx] = v;
    }
    __syncthreads();

    // --------------------------------------------------------
    // 计算 primary_conv（3×3）并写到 sh_prim 中心区域
    // --------------------------------------------------------
    // 先清零整个 sh_prim
    for (int idx = tid; idx < sz_prim; idx += nthreads) {
        sh_prim[idx] = 0.0f;
    }
    __syncthreads();

    // 每个线程只算自己对应的输出 prim(at local coords th,tw)
    if (th < TILE_H && tw < TILE_W) {
        int h_out = tile_i + th;
        int w_out = tile_j + tw;
        if (h_out < H_out && w_out < W_out) {
            float sum = 0.0f;
            // 遍历输入通道和 k1×k1
            for (int ci = 0; ci < C_in; ++ci) {
                for (int yy = 0; yy < k1; ++yy) {
                    for (int xx = 0; xx < k1; ++xx) {
                        // 在 shared in 中的位置
                        int ph = th*stride + yy;
                        int pw = tw*stride + xx;
                        float iv = sh_in[(ci*patch_h + ph)*patch_w + pw];
                        float kv = sh_w1[ci*(k1*k1) + yy*k1 + xx];
                        sum += iv * kv;
                    }
                }
            }
            // 存到 sh_prim 中心区域，加上 PAD2
            int ph2 = th + PAD2;
            int pw2 = tw + PAD2;
            sh_prim[ph2 * ext_w + pw2] = sum;
        }
    }
    __syncthreads();

    // --------------------------------------------------------
    // 计算 depthwise 5×5 并直接写两部分到 output
    // --------------------------------------------------------
    if (th < TILE_H && tw < TILE_W) {
        int h_out = tile_i + th;
        int w_out = tile_j + tw;
        if (h_out < H_out && w_out < W_out) {
            // 读取 prim 从 sh_prim 及其邻域
            float prim = sh_prim[(th+PAD2)*ext_w + (tw+PAD2)];
            float sum2 = 0.0f;
            for (int yy = 0; yy < K2; ++yy) {
                for (int xx = 0; xx < K2; ++xx) {
                    float iv = sh_prim[(th+yy)*(ext_w) + (tw+xx)];
                    float kv = sh_w2[yy*K2 + xx];
                    sum2 += iv * kv;
                }
            }
            // 写回输出：前半通道写 prim，后半写 depthwise
            int base = ((n*(2*C_mid) + cm)*H_out + h_out)*W_out + w_out;
            out[base]                   = prim;
            out[base + C_mid*H_out*W_out] = sum2;
        }
    }
}

// Host launch
extern "C" void launch_ghostconv(
    const float* input,
    const float* w1,
    const float* w2,
    float*       output,
    int          N,
    int          C_in,
    int          H,
    int          W,
    int          C_mid,
    int          k1,
    int          pad1,
    int          stride,
    int          H_out,
    int          W_out,
    cudaStream_t stream)
{
    // Grid & Block 布局：x→tile高, y→tile宽, z→batch*channels
    int grid_x = (H_out + TILE_H - 1) / TILE_H;
    int grid_y = (W_out + TILE_W - 1) / TILE_W;
    int grid_z = N * C_mid;
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(TILE_W, TILE_H);

    // 计算 shared memory 大小
    int patch_h   = TILE_H*stride + (k1-1);
    int patch_w   = TILE_W*stride + (k1-1);
    size_t sz_w1  = size_t(C_in)*k1*k1;
    size_t sz_w2  = size_t(K2)*K2;
    size_t sz_in  = size_t(C_in)*patch_h*patch_w;
    size_t sz_prim= size_t(TILE_H+2*PAD2)*(TILE_W+2*PAD2);
    size_t shm_sz = (sz_w1 + sz_w2 + sz_in + sz_prim) * sizeof(float);

    ghostconv_fused_kernel<<<grid, block, shm_sz, stream>>>(
      input, w1, w2, output,
      N, C_in, H, W,
      C_mid, k1, pad1, stride,
      H_out, W_out);

    CHECK_CUDA(cudaGetLastError());
}
