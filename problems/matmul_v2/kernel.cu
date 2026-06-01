#include <torch/extension.h>
#include <cuda_runtime.h>

// Reference: https://siboehm.com/articles/22/CUDA-MMM

#define TILE_SIZE 32

__global__ void matmul_naive_kernel(
    const float* __restrict__ A, // __restrict__ -> pointer is sole means of accessing A
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
    ) {
  // Address = block_idx * block_size + thread_idx
  const int row = blockIdx.y * blockDim.y + threadIdx.y; // Vertical (Y)
  const int col = blockIdx.x * blockDim.x + threadIdx.x; // Horiz (X)

  if (row < M && col < N) {
    float acc = 0.0;
    for (int i = 0; i < K; ++i) {
      acc += A[row * K + i] * B[i * N + col]; // Row-major indexing
    }
    C[row * N + col] = acc;
  }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    dim3 threads(TILE_SIZE, TILE_SIZE, 1);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y,
        1
    );

    // Launch the naive kernel on the default execution stream
    matmul_naive_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiplication Forward (CUDA)");
}
