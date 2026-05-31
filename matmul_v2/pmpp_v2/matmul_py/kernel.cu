#include <cuda_runtime.h>
#include <torch/extension.h>

// Reference: https://siboehm.com/articles/22/CUDA-MMM

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &matmul_cuda_forward, "Matmul Forward (CUDA)");
