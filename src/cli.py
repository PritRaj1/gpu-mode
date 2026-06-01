from pathlib import Path
import click
from torch.utils.cpp_extension import load_inline

@click.group()
def main():
    """Utils for GPU Mode challenges."""
    pass

@main.command()
@click.argument("problem")
@click.option("--gpu", default="A100", help="Target popcorn accelerator.")
def build(problem, gpu):
    """Bundle C++ kernel with submission.py template."""
    kernel_path = Path(f"problems/{problem}/kernel.cu")
    template_path = Path("src/template.py")

    if not kernel_path.exists():
        raise click.ClickException(f"Kernel source file missing at: {kernel_path}")
    if not template_path.exists():
        raise click.ClickException(f"Template wrapper missing at: {template_path}")

    kernel = kernel_path.read_text()
    template = template_path.read_text()

    headers = f"#!POPCORN leaderboard {problem}\n#!POPCORN gpu {gpu}\n\n"
    submission = template.replace("__CUDA_SOURCE__", f'r"""\n{kernel}\n"""')

    out_path = Path(f"build/{problem}_submission.py")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(headers + submission)

    click.secho(f"✨ Successfully bundled -> {out_path} [{gpu}]", fg="green")

@main.command()
@click.argument("problem")
def compile(problem):
    """JIT compile C++ kernel to check for compilation errors."""
    kernel_path = Path(f"problems/{problem}/kernel.cu")
    
    if not kernel_path.exists():
        raise click.ClickException(f"Kernel source file missing at: {kernel_path}")
        
    kernel = kernel_path.read_text()

    click.echo(f"Compiling {problem} kernel locally via load_inline...")
    load_inline(
        name=f"test_compile_{problem}",
        cpp_sources="",
        cuda_sources=kernel,
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
    click.secho("✔ Compilation OK!", fg="green")

if __name__ == "__main__":
    main()
