import pytest
import torch
import os
from torch.utils.cpp_extension import load

# Reference: https://siboehm.com/articles/22/CUDA-MMM

HERE = os.path.dirname(os.path.abspath(__file__))
device = "cuda"

matmul_cuda = load(
    name="matmul_cuda",
    sources=[os.path.join(HERE, "kernel.cu")],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)


@pytest.mark.parametrize(
    "M,K,N",
    [
        (64, 64, 64),
        (128, 256, 128),
        (256, 256, 256),
    ],
)
def test_shapes(M, K, N):
    torch.manual_seed(0)
    A = torch.randn(M, K, device=device, dtype=torch.float16).contiguous()
    B = torch.randn(K, N, device=device, dtype=torch.float16).contiguous()
    C = torch.zeros(M, N, device=device, dtype=torch.float16).contiguous()

    C_cuda = matmul_cuda.forward(A, B, C)
    C_ref = A @ B

    diff = (C_cuda - C_ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    print(
        f"\nShape ({M},{K},{N}) -> max error: {max_err:.6f}, mean error: {mean_err:.6f}"
    )

    assert torch.allclose(C_cuda, C_ref, rtol=1e-1, atol=1e-1), (
        f"Numerical check failed: max abs discrepancy found: {max_err:.6f}"
    )
