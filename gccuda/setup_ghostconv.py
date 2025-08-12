from torch.utils.cpp_extension import load

ghostconv = load(
    name="ghostconv",
    sources=["ghostconv_extension.cpp", "ghostconv_kernel.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

# Example usage
if __name__ == "__main__":
    import torch
    x = torch.randn(1, 3, 5, 5, device="cuda")
    w = torch.randn(2, 3, 3, 3, device="cuda")
    out = ghostconv.ghostconv_forward(x, w)
    print("Output shape:", out.shape)
