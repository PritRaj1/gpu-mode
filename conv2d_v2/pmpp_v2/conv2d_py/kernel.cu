#include <cuda_runtime.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv2d_cuda_forward, "Conv2D Spatial Forward (CUDA)");
}
