from pathlib import Path
import sys

problem = sys.argv[1]

kernel_path = Path(f"problems/{problem}/kernel.cu")
template_path = Path("src/template.py")

kernel = kernel_path.read_text()
template = template_path.read_text()

submission = template.replace("__CUDA_SOURCE__", f'r"""\n{kernel}\n"""')

out_path = Path(f"build/{problem}_submission.py")
out_path.write_text(submission)

print(f"Built {out_path}")
