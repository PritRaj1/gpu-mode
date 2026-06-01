import torch
import sys
import importlib.util
from pathlib import Path


def load_submission(path):
    spec = importlib.util.spec_from_file_location("submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.custom_kernel


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <problem_name>")
        sys.exit(1)

    problem = sys.argv[1]
    submission_path = Path(f"build/{problem}_submission.py")

    if not submission_path.exists():
        raise FileNotFoundError(submission_path)

    kernel_fn = load_submission(submission_path)

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.randn(4, 4, device=device)
    B = torch.randn(4, 4, device=device)
    C = torch.zeros(4, 4, device=device)

    try:
        out = kernel_fn((A, B, C))
        torch.cuda.synchronize()

        print("✔ Kernel executed")
        print("Output shape:", out.shape)
        print("Output sample:\n", out)

        if torch.isfinite(out).all():
            print("✔ Output finite")
        else:
            print("x Output contains NaNs/Infs")

    except Exception:
        print("x Kernel crashed")
        raise


if __name__ == "__main__":
    main()
