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
def benchmark(profile: bool = False):
    subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=True)
    subprocess.run(
        ["nvcc", "-O3", "-std=c++17", "-arch=sm_100", "-o", "/work/linear_bench",
         "/work/experiments/linear_tiling_bench.cu",
         "/work/part_6_linear_layers_and_fused_mlp_ops/016_linear_kernel.cu"],
        check=True,
    )
    if not profile:
        subprocess.run(["/work/linear_bench"], check=True)
        return

    metrics = ("Duration ", "Memory Throughput ", "DRAM Throughput ",
               "L1/TEX Cache Throughput ", "L2 Cache Throughput ",
               "Compute (SM) Throughput ", "Grid Size ", "Waves Per SM ",
               "Theoretical Occupancy ", "Achieved Occupancy ")
    for kernel in ("linear_kernel", "linear_tiled_kernel"):
        for m, skip in ((256, 2), (4096, 35)):
            # Two small correctness cases; M=256 adds 1 check, 2 warmups, 30 timed launches.
            result = subprocess.run(
                ["ncu", "--clock-control", "none", "--set", "basic",
                 "--kernel-name", f"regex:^{kernel}$", "--launch-skip", str(skip),
                 "--launch-count", "1", "/work/linear_bench"],
                capture_output=True, text=True,
            )
            if result.returncode or "==PROF== Profiling" not in result.stdout:
                raise RuntimeError(result.stdout + result.stderr)
            print(f"{kernel} M={m}", flush=True)
            for line in result.stdout.splitlines():
                if line.strip().startswith(metrics):
                    print(line.strip(), flush=True)


@app.local_entrypoint()
def main(profile: bool = False):
    benchmark.remote(profile)
