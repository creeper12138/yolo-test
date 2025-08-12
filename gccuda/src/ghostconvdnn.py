import ctypes
import torch

# ─── 1) PyTorch GhostConv 定义 ────────────────────────────────────
class PyTorchGhostConv(torch.nn.Module):
    def __init__(self, C_in, C_mid):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(C_in, C_mid, 3, padding=1).cuda()
        self.conv2 = torch.nn.Conv2d(C_mid, C_mid, 5, padding=2, groups=C_mid).cuda()
    def forward(self, x):
        y = self.conv1(x)
        z = self.conv2(y)
        return torch.cat((y, z), dim=1)

# ─── 2) Custom DLL 接口加载 ────────────────────────────────────────
dll = ctypes.CDLL(r"D:\NSYS\gccuda\src\ghostconv3.dll")

# Stage 1: primary_conv
dll.launch_primary_conv.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # input, w1, tmp
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # N, C_in, H, W
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # C_mid, k, stride, pad
    ctypes.c_int, ctypes.c_int,                          # H_out, W_out
    ctypes.c_void_p                                      # stream
]
dll.launch_primary_conv.restype = None

# Stage 2: depthwise_conv
dll.launch_depthwise_conv.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # tmp, w2, tmp2
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # N, C_mid, H_out, W_out
    ctypes.c_void_p                                      # stream
]
dll.launch_depthwise_conv.restype = None

# Stage 3: concat
dll.launch_concat.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # tmp, tmp2, out
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # N, C_mid, H_out, W_out
    ctypes.c_void_p                                      # stream
]
dll.launch_concat.restype = None

def custom_full(x, w1, w2, tmp, tmp2, out, stream_handle):
    # 一次完整前向：3×3 → 5×5 depthwise → concat
    dll.launch_primary_conv(
        x.data_ptr(), w1.data_ptr(), tmp.data_ptr(),
        N, C_in, H, W,
        C_mid, k, stride, pad,
        H_out, W_out,
        ctypes.c_void_p(stream_handle)
    )
    dll.launch_depthwise_conv(
        tmp.data_ptr(), w2.data_ptr(), tmp2.data_ptr(),
        N, C_mid, H_out, W_out,
        ctypes.c_void_p(stream_handle)
    )
    dll.launch_concat(
        tmp.data_ptr(), tmp2.data_ptr(), out.data_ptr(),
        N, C_mid, H_out, W_out,
        ctypes.c_void_p(stream_handle)
    )

# ─── 3) 基准设置 ─────────────────────────────────────────────
device = "cuda"
# 参数
N, C_in, H, W = 8, 3, 224, 224
C_mid = 64
k, stride, pad = 3, 1, 1
H_out = (H + 2*pad - k)//stride + 1
W_out = (W + 2*pad - k)//stride + 1

# 数据
x    = torch.randn(N, C_in, H, W, device=device)
ptgc = PyTorchGhostConv(C_in, C_mid).eval()
w1   = torch.randn(C_mid, C_in, k, k, device=device)
w2   = torch.randn(C_mid, 1,    5, 5, device=device)
tmp  = torch.empty(N, C_mid, H_out, W_out, device=device)
tmp2 = torch.empty_like(tmp)
out  = torch.empty(N, 2*C_mid, H_out, W_out, device=device)

# 拿到当前 PyTorch CUDA Stream 的原生句柄
stream_handle = torch.cuda.current_stream().cuda_stream

# ─── 4) Warm-up ───────────────────────────────────────────────
for _ in range(20):
    _ = ptgc(x)
    custom_full(x, w1, w2, tmp, tmp2, out, stream_handle)
torch.cuda.synchronize()

# ─── 5) PyTorch/cuDNN 测时 ───────────────────────────────────
iters = 100
evt_s = torch.cuda.Event(enable_timing=True)
evt_e = torch.cuda.Event(enable_timing=True)

evt_s.record()
for _ in range(iters):
    _ = ptgc(x)
evt_e.record()
torch.cuda.synchronize()
pt_time = evt_s.elapsed_time(evt_e) / iters

# ─── 6) Custom DLL 测时 ───────────────────────────────────────
evt_s.record()
for _ in range(iters):
    custom_full(x, w1, w2, tmp, tmp2, out, stream_handle)
evt_e.record()
torch.cuda.synchronize()
zdy_time = evt_s.elapsed_time(evt_e) / iters

# ─── 7) 打印结果 ─────────────────────────────────────────────
print("=== PyTorch/cuDNN GhostConv avg (ms) ===")
print(f"{pt_time:.3f}")

print("\n=== Custom DLL GhostConv avg (ms) ===")
print(f"{zdy_time:.3f}")
