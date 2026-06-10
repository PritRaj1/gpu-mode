# gpu-mode

Submission for [GPU MODE](https://www.gpumode.com) practice problems.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- [Popcorn](https://github.com/gpu-mode/popcorn-cli)
- [Nsight systems](https://developer.nvidia.com/nsight-systems/get-started)

## NVIDIA Kernel Module Permissions

Linux distros ship with the security flag `RestrictProfilingToAdminUsers=1` enabled. For profiling within an isolated Python env, you can instruct the kernel module to allow user-space register sampling:

```bash
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee /etc/modprobe.d/nvidia-profiler.conf
# Then reboot
```

## Links

- [Matmul step-through](https://siboehm.com/articles/22/CUDA-MMM)
- [Nsight profiling](https://arikpoz.github.io/posts/2025-05-25-speed-up-pytorch-training-by-3x-with-nvidia-nsight-and-pytorch-2-tricks/)
- [Using ncu-ui](https://www.youtube.com/watch?v=04dJ-aePYpE)
- [Discord invite](https://discord.com/invite/gpumode)
- [GPU Glossary](https://modal.com/gpu-glossary)
