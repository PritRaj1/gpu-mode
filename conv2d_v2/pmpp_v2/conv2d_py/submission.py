#!POPCORN leaderboard conv2d_v2
#!POPCORN gpu A100

from task import input_t, output_t
import torch
# import torch.nn.functional as F
import pathlib
from torch.utils.cpp_extension import load_inline

cuda_path = pathlib.Path(__file__).parent / "kernel.cu"
cuda_source = cuda_path.read_text()

cuda_module = load_inline(
    name="conv2d_extension",
    cpp_sources="",
    cuda_sources=cuda_source,
    functions=["forward"],
    with_cuda=True,
    extra_cflags=["-O3"]
)

def custom_kernel(data: input_t) -> output_t:
    """
    Implementation of 2D convolution using PyTorch with no padding and no striding.
    Args:
        data: Tuple of (input tensor, kernel tensor)
        spec: Convolution specifications
    Returns:
        Output tensor after convolution
    """
    input_tensor, kernel, output = data
    # output[...] = F.conv2d(input_tensor, kernel, stride=1, padding=0)
 
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    if not kernel.is_contiguous():
        kernel = kernel.contiguous()
    if not output.is_contiguous():
        output = output.contiguous()

    cuda_module.forward(input_tensor, kernel, output)
    return output
