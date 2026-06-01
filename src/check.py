from torch.utils.cpp_extension import load_inline
from pathlib import Path
import sys

problem = sys.argv[1]

kernel = Path(f"problems/{problem}/kernel.cu").read_text()

mod = load_inline(
    name="test_kernel",
    cpp_sources="",
    cuda_sources=kernel,
    with_cuda=True,
    verbose=True,
)

print("✔ compile OK")
