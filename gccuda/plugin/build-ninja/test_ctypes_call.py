import ctypes as C
import numpy as np
import os, sys

# 1) 加载 DLL（保证 TRT 和 CUDA 的 dll 路径在 PATH）
os.add_dll_directory(r"D:\TensorRT-8.6.1.6\lib")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin")
dll = C.CDLL(r"D:\NSYS\gccuda\plugin\build-ninja\ghostconv_trt.dll")  # ← 改成你的路径

# 2) 绑定函数原型
# enum gc_dtype_t { GC_DTYPE_F32=0, GC_DTYPE_F16=1 };
GC_DTYPE_F32 = 0

dll.ghostconv_get_output_shape.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                                           C.POINTER(C.c_int), C.POINTER(C.c_int)]
dll.ghostconv_get_output_shape.restype  = None

dll.ghostconv_launch.argtypes = [
    C.c_void_p, C.c_void_p,   # x, y
    C.c_void_p, C.c_void_p,   # w1, w2
    C.c_int, C.c_int, C.c_int, C.c_int,  # N, C_in, H, W
    C.c_int, C.c_int, C.c_int, C.c_int,  # C_mid, k, stride, pad
    C.c_int, C.c_int,                      # H_out, W_out
    C.c_int,                               # dtype
    C.c_void_p, C.c_size_t,                # workspace ptr, bytes (可为NULL/0)
    C.c_void_p                             # cudaStream_t (0=默认流)
]
dll.ghostconv_launch.restype = C.c_bool

# 3) 组装一次调用
N, C_in, H, W = 1, 16, 32, 32
k, s, p = 3, 1, 1
C_mid = 24       # C_out = 48
K2 = 5           # 你 DW 核的固定大小

H_out = C.c_int()
W_out = C.c_int()
dll.ghostconv_get_output_shape(H, W, k, s, p, C.byref(H_out), C.byref(W_out))
H_out, W_out = H_out.value, W_out.value
print("H_out, W_out =", H_out, W_out)

# 输入/权重/输出（FP32）
x  = np.random.randn(N, C_in, H, W).astype(np.float32)
w1 = np.random.randn(C_mid, C_in, k, k).astype(np.float32)
w2 = np.random.randn(C_mid, 1, K2, K2).astype(np.float32)
y  = np.empty((N, 2*C_mid, H_out, W_out), dtype=np.float32)

ok = dll.ghostconv_launch(
    x.ctypes.data_as(C.c_void_p),
    y.ctypes.data_as(C.c_void_p),
    w1.ctypes.data_as(C.c_void_p),
    w2.ctypes.data_as(C.c_void_p),
    N, C_in, H, W,
    C_mid, k, s, p,
    H_out, W_out,
    GC_DTYPE_F32,
    C.c_void_p(0), 0,
    C.c_void_p(0)  # 默认流
)
print("launch ok:", ok)
print("y stats: mean=%.6f std=%.6f min=%.6f max=%.6f" %
      (float(y.mean()), float(y.std()), float(y.min()), float(y.max())))
