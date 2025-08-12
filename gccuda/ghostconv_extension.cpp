#include <torch/extension.h>
#include <vector>
#include "ghostconv_kernel.h"

torch::Tensor ghostconv_forward(torch::Tensor input, torch::Tensor weight) {
    // Ensure correct device and types
    input = input.contiguous();
    weight = weight.contiguous();
    TORCH_CHECK(input.device().is_cuda(), "Input must be CUDA");
    TORCH_CHECK(weight.device().is_cuda(), "Weight must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "Only float32 supported");

    int N = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int C_out = weight.size(0);
    int kH = weight.size(2);
    int kW = weight.size(3);
    int stride = 1;
    int pad = kH / 2;  // adjust if needed

    int H_out = (H + 2 * pad - kH) / stride + 1;
    int W_out = (W + 2 * pad - kW) / stride + 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    // Launch CUDA kernel
    launch_ghconv(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H, W,
        C_out, kH, kW,
        H_out, W_out,
        stride, pad,
        at::cuda::getCurrentCUDAStream());

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ghostconv_forward", &ghostconv_forward, "GhostConv forward (CUDA)");
}
