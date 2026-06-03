#!POPCORN leaderboard matmul_v2
#!POPCORN gpu A100

from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define TILE_SIZE 16

/*
__restrict__ hint: this pointer is the sole pointer that accesses A
*/
__global__ void matmul_naive_kernel(const __half* __restrict__ a,
                                    const __half* __restrict__ b,
                                    __half* __restrict__ c, int m, int n,
                                    int k, float alpha, float beta) {
  // address = block_idx * block_size + thread_idx
  const int row = blockIdx.y * blockDim.y + threadIdx.y;  // vertical (y)
  const int col = blockIdx.x * blockDim.x + threadIdx.x;  // horiz (x)

  if (row < m && col < n) {
    float dotprod = 0.0f;

    for (int i = 0; i < k; ++i) {
      float a_val = __half2float(a[row * k + i]);
      float b_val = __half2float(b[i * n + col]);
      dotprod += a_val * b_val;
    }
    
    float c_val = __half2float(c[row * n + col]);
    float fma_out = alpha * dotprod + beta * c_val;
    c[row * n + col] = __float2half(fma_out);
  }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);
  float alpha = 1.0f;
  float beta = 0.0f;

  dim3 threads(TILE_SIZE, TILE_SIZE, 1);
  dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y,
              1);

  matmul_naive_kernel<<<blocks, threads>>>(
      reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(C.data_ptr<at::Half>()), M, N, K, alpha, beta);

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

    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    cuda_module.forward(A, B, C)
    return C
