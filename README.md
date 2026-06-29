# gpu-mode

Submission utils for [GPU MODE](https://www.gpumode.com) practice problems.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- [Popcorn](https://github.com/gpu-mode/popcorn-cli)
- [Nsight systems](https://developer.nvidia.com/nsight-systems/get-started)

## Install

```bash
uv sync
```

## Quick start

Run with `uv run gpu-mode`:

```bash
Usage: gpu-mode [OPTIONS] COMMAND [ARGS]...

  Utils for GPU Mode challenges.

Options:
  --help  Show this message and exit.

Commands:
  build    Bundle C++ kernel with submission.py template.
  clean    Clear build artifacts and wipe local PyTorch JIT caches.
  compile  JIT compile C++ kernel to check for compilation errors.
```

For example, to build the python submission script for the `matmul_v2`

```bash
uv run gpu-mode build matmul_v2
```

Then to submit/test using GPU MODE's popcorn CLI:

```bash
popcorn submit build/matmul_v2_submission.py
```

## NVIDIA Kernel Module Permissions

Linux distros ship with the security flag `RestrictProfilingToAdminUsers=1` enabled. For profiling within an isolated uv venv, you can instruct the kernel module to allow user-space register sampling:

```bash
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee /etc/modprobe.d/nvidia-profiler.conf
# Then reboot
```

## Links

- [Matmul step-through](https://siboehm.com/articles/22/CUDA-MMM)
- [Nsight profiling](https://puzzles.modular.com/puzzle_30/nvidia_profiling_basics.html)
- [Using ncu-ui](https://www.youtube.com/watch?v=04dJ-aePYpE)
- [Discord invite](https://discord.com/invite/gpumode)
- [GPU Glossary](https://modal.com/gpu-glossary)
