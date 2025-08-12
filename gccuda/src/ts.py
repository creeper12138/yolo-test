import ctypes, time
import torch

# ─── 1) ctypes+DLL 版 GhostConv ──────────────────────────
lib = ctypes.CDLL(r"D:\NSYS\gccuda\src\ghostconv2.dll")
lib.launch_ghostconv.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p
]
lib.launch_ghostconv.restype = None

def ghostconv_ctypes(x, w1, w2, y, device_stream):
    N, C_in, H, W = x.shape
    C_mid = w1.shape[0]
    k = w1.shape[2]
    stride, pad = 1, k//2
    H_out = (H + 2*pad - k)//stride + 1
    W_out = (W + 2*pad - k)//stride + 1

    lib.launch_ghostconv(
        x.data_ptr(), w1.data_ptr(), w2.data_ptr(),
        y.data_ptr(),
        N, C_in, H, W,
        C_mid, k, stride, pad,
        H_out, W_out,
        ctypes.c_void_p(0)
    )

# ─── 2) PyTorch/cuDNN 版 GhostConv ───────────────────────
class PyTorchGhostConv(torch.nn.Module):
    def __init__(self, C_in, C_mid):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(C_in, C_mid, 3, padding=1).cuda()
        self.conv2 = torch.nn.Conv2d(C_mid, C_mid, 5, padding=2, groups=C_mid).cuda()
    def forward(self, x):
        y = self.conv1(x)
        z = self.conv2(y)
        return torch.cat((y, z), dim=1)

# ─── 3) PyTorch/CPU 版 GhostConv ────────────────────────
class PyTorchGhostConvCPU(torch.nn.Module):
    def __init__(self, C_in, C_mid):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(C_in, C_mid, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(C_mid, C_mid, 5, padding=2, groups=C_mid)
    def forward(self, x):
        y = self.conv1(x)
        z = self.conv2(y)
        return torch.cat((y, z), dim=1)

# ─── 4) 准备数据 ────────────────────────────────────────
# 测试大小
N, C_in, H, W = 8, 3, 224, 224
C_mid = 64

# GPU 数据
x_gpu  = torch.randn(N, C_in, H, W, device="cuda")
w1_gpu = torch.randn(C_mid, C_in, 3, 3, device="cuda")
w2_gpu = torch.randn(C_mid, 1,    5, 5, device="cuda")
y_gpu1 = torch.empty(N, 2*C_mid, H, W, device="cuda")
y_gpu2 = torch.empty_like(y_gpu1)


# 实例化模型
pt_gpu = PyTorchGhostConv(C_in, C_mid).eval()
pt_cpu = PyTorchGhostConvCPU(C_in, C_mid).eval()

# warm-up
for _ in range(5):
    ghostconv_ctypes(x_gpu, w1_gpu, w2_gpu, y_gpu1, None)
    pt_gpu(x_gpu)

torch.cuda.synchronize()

# ─── 5) 基准测试 ────────────────────────────────────────
iters = 50

# GPU ctypes+DLL
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(iters):
    ghostconv_ctypes(x_gpu, w1_gpu, w2_gpu, y_gpu1, None)
end.record()
torch.cuda.synchronize()
time_ctypes = start.elapsed_time(end) / iters

# GPU PyTorch
start.record()
for _ in range(iters):
    y_gpu2 = pt_gpu(x_gpu)
end.record()
torch.cuda.synchronize()
time_ptgpu = start.elapsed_time(end) / iters



# ─── 6) 打印结果 ────────────────────────────────────────
print(f"GPU ctypes+DLL  GhostConv avg: {time_ctypes:6.2f} ms")
print(f"GPU PyTorch/cuDNN GhostConv avg: {time_ptgpu:6.2f} ms")
