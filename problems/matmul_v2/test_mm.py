import pytest
import torch
import os
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
device = "cuda"

matmul_cuda = load(
    name="matmul_cuda",
    sources=[os.path.join(HERE, "kernel.cu")],
    verbose=True,
)

@pytest.mark.parametrize("M,K,N", [
    (64, 64, 64),
    (128, 256, 128),
    (256, 256, 256),
    (512, 512, 512),
])
def test_shapes(M, K, N):
    torch.manual_seed(0)
    A = torch.randn(M, K, device=device, dtype=torch.float32).contiguous()
    B = torch.randn(K, N, device=device, dtype=torch.float32).contiguous()
    C = torch.zeros(M, N, device=device, dtype=torch.float32).contiguous()

    C_cuda = matmul_cuda.forward(A, B, C)
    C_ref = A @ B

    diff = (C_cuda - C_ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    print(f"\nShape ({M},{K},{N}) -> max error: {max_err:.6f}, mean error: {mean_err:.6f}")
    assert max_err < 1e-2
