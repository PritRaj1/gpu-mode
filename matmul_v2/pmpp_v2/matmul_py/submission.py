#!POPCORN leaderboard matmul_v2
#!POPCORN gpu A100

from task import input_t, output_t
import pathlib
from torch.utils.cpp_extension import load_inline

cuda_path = pathlib.Path(__file__).parent / "kernel.cu"
cuda_source = cuda_path.read_text()

cuda_module = load_inline(
    name="matmul_extension",
    cpp_sources="",
    cuda_sources=cuda_source,
    functions=["forward"],
    with_cuda=True,
    extra_cflags=["-O3"],
)


def custom_kernel(data: input_t) -> output_t:
    """
    Implementation of 2D matmul using PyTorch.
    Args:
        data: Tuple of (input tensor, kernel tensor)
        spec: Matmul specifications
    Returns:
        Output tensor after matmul
    """
    a, b, c = data
    # c[...] = a @ b

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if not c.is_contiguous():
        c = c.contiguous()

    cuda_module.forward(a, b, c)
    return c
