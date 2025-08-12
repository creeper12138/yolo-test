#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>  // getCurrentCUDAStream()
#include "ghostconv_kernel.h"

torch::Tensor ghostconv_forward(torch::Tensor input, torch::Tensor weight) {
    input  = input.contiguous();
    weight = weight.contiguous();
    TORCH_CHECK(input.is_cuda(),  "Input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32,  "float32 only");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "float32 only");

    int N     = input.size(0);
    int C_in  = input.size(1);
    int H     = input.size(2);
    int W     = input.size(3);
    int C_out = weight.size(0);
    int kH    = weight.size(2);
    int kW    = weight.size(3);
    int stride = 1;
    int pad    = kH / 2;
    int H_out = (H + 2*pad - kH) / stride + 1;
    int W_out = (W + 2*pad - kW) / stride + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_ghconv(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H, W,
        C_out, kH, kW,
        H_out, W_out,
        stride, pad,
        stream);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ghostconv_forward", &ghostconv_forward, "GhostConv forward (CUDA)");
}
