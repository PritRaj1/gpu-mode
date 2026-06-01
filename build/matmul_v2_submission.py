#!POPCORN leaderboard matmul_v2
#!POPCORN gpu A100


from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

// Reference: https://siboehm.com/articles/22/CUDA-MMM

#define TILE_SIZE 32

__global__ void matmul_naive_kernel(
    const float* __restrict__ A,  // __restrict__ -> pointer is sole means of
                                  // accessing A
    const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
  // Address = block_idx * block_size + thread_idx
  const int row = blockIdx.y * blockDim.y + threadIdx.y;  // Vertical (Y)
  const int col = blockIdx.x * blockDim.x + threadIdx.x;  // Horiz (X)

  if (row < M && col < N) {
    float acc = 0.0;
    for (int i = 0; i < K; ++i) {
      acc += A[row * K + i] * B[i * N + col];  // Row-major indexing
    }
    C[row * N + col] = acc;
  }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);

  dim3 threads(TILE_SIZE, TILE_SIZE, 1);
  dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y,
              1);

  // Launch the naive kernel on the default execution stream
  matmul_naive_kernel<<<blocks, threads>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Matrix Multiplication Forward (CUDA)");
}

"""

cuda_module = load_inline(
    name="kernel",
    cpp_sources="",
    cuda_sources=CUDA_SRC,
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
)


def custom_kernel(data: input_t) -> output_t:
    A, B, C = data

    torch.cuda.synchronize()

    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    cuda_module.forward(A, B, C)
    torch.cuda.synchronize()

    return C
