#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define TILE_SIZE 32

/*
__restrict__ hint: this pointer is the sole pointer that accesses A
*/
__global__ void matmul_naive_kernel(const __half* __restrict__ A,
                                    const __half* __restrict__ B,
                                    __half* __restrict__ C, int M, int N,
                                    int K) {
  // Address = block_idx * block_size + thread_idx
  const int row = blockIdx.y * blockDim.y + threadIdx.y;  // Vertical (Y)
  const int col = blockIdx.x * blockDim.x + threadIdx.x;  // Horiz (X)

  if (row < M && col < N) {
    __half acc = __float2half(0.0f);
    for (int i = 0; i < K; ++i) {
      acc = __hadd(
          acc, __hmul(A[row * K + i], B[i * N + col]));  // Row-major indexing
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

  matmul_naive_kernel<<<blocks, threads>>>(
      reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(C.data_ptr<at::Half>()), M, N, K);

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Matrix Multiplication Forward (CUDA)");
}
