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
    A, B, C = data

    if not A.is_contiguous(): A = A.contiguous()
    if not B.is_contiguous(): B = B.contiguous()
    if not C.is_contiguous(): C = C.contiguous()

    default_stream = torch.cuda.default_stream()
    with torch.cuda.stream(default_stream):
        cuda_module.forward(A, B, C)
        torch.cuda.synchronize()

    return C
