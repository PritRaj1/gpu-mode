import os
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
device = "cuda"

matmul_cuda = load(
    name="matmul_cuda_profile",
    sources=[os.path.join(HERE, "kernel.cu")],
    extra_cuda_cflags=["-O3", "--ptxas-options=-v"],  # verbose
    verbose=True,
)

if __name__ == "__main__":
    M, K, N = 256, 256, 256
    torch.manual_seed(0)
    A = torch.randn(M, K, device=device, dtype=torch.float16).contiguous()
    B = torch.randn(K, N, device=device, dtype=torch.float16).contiguous()
    C = torch.zeros(M, N, device=device, dtype=torch.float16).contiguous()

    print(f"Profiling mamtul_v2 for shape ({M}, {K}, {N})...")

    # Warnmup to force JIT + ramp-up
    for _ in range(10):
        _ = matmul_cuda.forward(A, B, C)
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("v1_coalesced_flat_mm"):
        C_cuda = matmul_cuda.forward(A, B, C)
        torch.cuda.synchronize()

    print("Profile complete.")
