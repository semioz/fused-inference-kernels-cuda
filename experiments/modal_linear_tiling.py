import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("linear-tiling-comparison")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .add_local_file(str(ROOT / "model.cuh"), "/work/model.cuh")
    .add_local_file(str(ROOT / "part_6_linear_layers_and_fused_mlp_ops/016_linear_kernel.cu"),
                    "/work/part_6_linear_layers_and_fused_mlp_ops/016_linear_kernel.cu")
    .add_local_file(str(ROOT / "experiments/linear_tiling_bench.cu"),
                    "/work/experiments/linear_tiling_bench.cu")
)


@app.function(gpu="B200", image=image, timeout=600)
def benchmark():
    subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=True)
    subprocess.run(
        ["nvcc", "-O3", "-std=c++17", "-arch=sm_100", "-o", "/work/linear_bench",
         "/work/experiments/linear_tiling_bench.cu",
         "/work/part_6_linear_layers_and_fused_mlp_ops/016_linear_kernel.cu"],
        check=True,
    )
    subprocess.run(["/work/linear_bench"], check=True)


@app.local_entrypoint()
def main():
    benchmark.remote()
