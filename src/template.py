from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CUDA_SRC = __CUDA_SOURCE__

cuda_module = load_inline(
    name="kernel",
    cpp_sources="",
    cuda_sources=CUDA_SRC,
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
)

def custom_kernel(data: input_t) -> output_t:
    a, b, c = data
    return cuda_module.forward(a, b, c)
