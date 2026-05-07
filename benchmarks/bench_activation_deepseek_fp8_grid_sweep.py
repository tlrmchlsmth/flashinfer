"""
Grid sweep for activationDeepSeekKernelV2 (FP8 E4M3 silu+mul + per-block quantize).

Measures kernel latency across production EP32 DeepSeek-R1 shapes.
The kernel runs between FC1 and FC2 in the MoE: it takes FP8 input from GEMM1,
applies SwiGLU activation in f32, then quantizes output to FP8 with per-128-element
block scales.

Grid structure: (numScaleBlocks, gridY, 1) with 128 threads/block.
  - X = innerDim/2/128 (fixed by K, one block per 128-element output scale block)
  - Y = number of token rows (sweep dimension)

Uses CUDA graphs with 200 replays per capture to eliminate launch overhead.

Usage:
    python bench_activation_deepseek_fp8_grid_sweep.py --output-dir /path/to/results
"""

import argparse
import csv
import json
import os
import datetime
import functools
import numpy as np
import torch

from flashinfer.testing.utils import bench_gpu_time
from flashinfer.jit.fp8_activation_benchmark import gen_fp8_activation_benchmark_module

CUDA_GRAPH_ITERS = 200

# Production DeepSeek-R1 with EP_size=32:
# K = innerDim passed to activation kernel
# Each token routes to 6-10 experts; after EP32 + padding:
#   real_rows = actual tokens on this rank
#   padded_rows = after EP alignment padding (totalNumPaddedTokens)
#
# (real_rows, padded_rows, inner_dim)
PRODUCTION_SCENARIOS = [
    # Low load: 16-32 concurrent requests
    (96, 3072, 2048),
    (192, 3072, 2048),
    # Medium load
    (384, 3072, 2048),
    (512, 8192, 2048),
    # High load
    (768, 8192, 2048),
    (1024, 32768, 2048),
]


def get_sm_count():
    return torch.cuda.get_device_properties(0).multi_processor_count


def get_gpu_name():
    return torch.cuda.get_device_name(0)


def get_module():
    return gen_fp8_activation_benchmark_module().build_and_load()


def create_test_inputs(padded_rows, real_rows, inner_dim, device="cuda"):
    """Create inputs matching activationDeepSeekKernelV2 data layout.

    Input layout (from GEMM1 output):
      input: [padded_rows, inner_dim] fp8_e4m3
      input_scales: [numInputScaleBlocks * padded_rows] float32, column-major
        where numInputScaleBlocks = inner_dim / 128

    Output layout:
      output: [padded_rows, inner_dim/2] fp8_e4m3
      output_scales: [numOutputScaleBlocks * padded_rows] float32, column-major
        where numOutputScaleBlocks = inner_dim / 2 / 128

    total_padded_tokens: [1] int32 on device
    """
    half_dim = inner_dim // 2

    # Generate bf16 random data, quantize to fp8 with scales
    x_bf16 = torch.randn(padded_rows, inner_dim, dtype=torch.bfloat16, device=device)

    # Compute per-128-block scales for input (column-major storage)
    num_input_scale_blocks = inner_dim // 128
    input_scales_list = []
    for sb in range(num_input_scale_blocks):
        col_start = sb * 128
        col_end = col_start + 128
        block_data = x_bf16[:, col_start:col_end].float()
        block_amax = block_data.abs().amax(dim=1).clamp(min=1e-12)
        scale = block_amax / 448.0
        input_scales_list.append(scale)

    # Pack scales in column-major: scales[scaleBlock * padded_rows + row]
    input_scales = torch.stack(input_scales_list, dim=0).contiguous()  # [numBlocks, padded_rows]
    input_scales_flat = input_scales.reshape(-1)  # column-major: block-major order

    # Quantize input to fp8: input[row, col] = round(x_bf16[row, col] / scale[row, col/128])
    x_fp8 = torch.empty(padded_rows, inner_dim, dtype=torch.float8_e4m3fn, device=device)
    for sb in range(num_input_scale_blocks):
        col_start = sb * 128
        col_end = col_start + 128
        scale = input_scales[sb].unsqueeze(1)  # [padded_rows, 1]
        x_fp8[:, col_start:col_end] = (x_bf16[:, col_start:col_end].float() / scale).to(
            torch.float8_e4m3fn
        )

    # Output tensors
    num_output_scale_blocks = half_dim // 128
    output = torch.empty(padded_rows, half_dim, dtype=torch.float8_e4m3fn, device=device)
    output_scales = torch.empty(
        num_output_scale_blocks * padded_rows, dtype=torch.float32, device=device
    )

    # Device tensor for totalNumPaddedTokens
    total_padded = torch.tensor([real_rows], dtype=torch.int32, device=device)

    return {
        "input": x_fp8,
        "input_scales": input_scales_flat,
        "output": output,
        "output_scales": output_scales,
        "total_padded": total_padded,
        "inner_dim": inner_dim,
        # For correctness checking: dequantized f32 input
        "x_bf16": x_bf16,
        "per_block_input_scales": input_scales,  # [numBlocks, padded_rows]
    }


def run_kernel(module, inputs, grid_y=-1):
    if grid_y > 0:
        module.activation_deepseek_fp8_v2_tuned(
            inputs["input"],
            inputs["input_scales"],
            inputs["output"],
            inputs["output_scales"],
            inputs["total_padded"],
            inputs["inner_dim"],
            grid_y,
        )
    else:
        module.activation_deepseek_fp8_v2(
            inputs["input"],
            inputs["input_scales"],
            inputs["output"],
            inputs["output_scales"],
            inputs["total_padded"],
            inputs["inner_dim"],
        )


def compute_reference(inputs, real_rows):
    """Compute reference silu+mul output in f32 for correctness checking."""
    inner_dim = inputs["inner_dim"]
    half_dim = inner_dim // 2
    per_block_scales = inputs["per_block_input_scales"]  # [numBlocks, padded_rows]
    x_fp8 = inputs["input"]

    # Dequantize: for each block, x_f32 = fp8_val * scale
    x_f32 = torch.zeros(real_rows, inner_dim, dtype=torch.float32, device=x_fp8.device)
    num_blocks = inner_dim // 128
    for sb in range(num_blocks):
        col_start = sb * 128
        col_end = col_start + 128
        scale = per_block_scales[sb, :real_rows].unsqueeze(1)  # [real_rows, 1]
        x_f32[:, col_start:col_end] = x_fp8[:real_rows, col_start:col_end].float() * scale

    # SwiGLU: silu(x2) * x1
    x1 = x_f32[:, :half_dim]
    x2 = x_f32[:, half_dim:]
    ref_out = torch.nn.functional.silu(x2) * x1
    return ref_out


def check_correctness(module, inputs, real_rows, grid_y=-1):
    """Verify kernel output matches reference within FP8 quantization tolerance."""
    run_kernel(module, inputs, grid_y)
    torch.cuda.synchronize()

    ref_f32 = compute_reference(inputs, real_rows)

    inner_dim = inputs["inner_dim"]
    half_dim = inner_dim // 2
    num_out_blocks = half_dim // 128
    padded_rows = inputs["input"].shape[0]

    # Dequantize kernel output to f32
    kernel_out_fp8 = inputs["output"][:real_rows]
    out_scales_flat = inputs["output_scales"]

    kernel_out_f32 = torch.zeros(real_rows, half_dim, dtype=torch.float32, device=kernel_out_fp8.device)
    for sb in range(num_out_blocks):
        col_start = sb * 128
        col_end = col_start + 128
        # Column-major: scale at index [sb * padded_rows + row]
        scale = out_scales_flat[sb * padded_rows : sb * padded_rows + real_rows].unsqueeze(1)
        kernel_out_f32[:, col_start:col_end] = kernel_out_fp8[:, col_start:col_end].float() * scale

    # Allow tolerance for fp8 quantization (2 steps of quantization: input + output)
    atol = ref_f32.abs().max().item() * 0.05  # 5% of max value
    rtol = 0.1
    close = torch.allclose(kernel_out_f32, ref_f32, atol=atol, rtol=rtol)

    if not close:
        diff = (kernel_out_f32 - ref_f32).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        ref_max = ref_f32.abs().max().item()
        print(f"    MISMATCH: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} "
              f"ref_max={ref_max:.6f} (grid_y={grid_y})")
    return close


def time_it(fn, dry_run_iters, repeat_iters):
    times = bench_gpu_time(
        fn,
        enable_cupti=True,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        cold_l2_cache=True,
        use_cuda_graph=True,
        num_iters_within_graph=CUDA_GRAPH_ITERS,
    )
    return float(np.median(times)), float(np.std(times))


def get_grid_y_candidates(padded_rows, inner_dim, sm_count):
    """Generate candidate gridY values to sweep.

    gridY controls how many token rows each CTA processes:
    - gridY = padded_rows: 1 row per CTA (max parallelism, many idle CTAs if real < padded)
    - gridY = real_rows: match actual work
    - gridY < real_rows: fewer CTAs, each handles multiple rows via loop
    """
    half_dim = inner_dim // 2
    num_scale_blocks = half_dim // 128  # gridX (fixed)
    candidates = set()

    # Key points
    candidates.add(1)
    candidates.add(min(padded_rows, 8192))  # production default

    # Multiples of SM count (wave-filling)
    for mult in [1, 2, 4, 8, 16, 32, 64, 128]:
        val = mult * sm_count // num_scale_blocks
        if 1 <= val <= min(padded_rows, 8192):
            candidates.add(val)

    # Powers of 2
    y = 1
    while y <= min(padded_rows, 8192):
        candidates.add(y)
        y *= 2

    # Near the real row counts
    for real in [96, 192, 384, 512, 768, 1024]:
        for offset in [-32, -16, 0, 16, 32]:
            val = real + offset
            if 1 <= val <= min(padded_rows, 8192):
                candidates.add(val)

    return sorted(candidates)


def run_sweep(args):
    print(f"GPU: {get_gpu_name()}")
    sm_count = get_sm_count()
    print(f"SM count: {sm_count}")
    print(f"CUDA graph replays per measurement: {CUDA_GRAPH_ITERS}")

    module = get_module()
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    for real_rows, padded_rows, inner_dim in PRODUCTION_SCENARIOS:
        half_dim = inner_dim // 2
        num_scale_blocks = half_dim // 128

        print(f"\n{'='*78}")
        print(
            f"  real={real_rows}  padded={padded_rows}  "
            f"({padded_rows // real_rows}x padding)  "
            f"innerDim={inner_dim}  gridX={num_scale_blocks}"
        )
        print(f"{'='*78}")

        # --- Baseline: no padding (padded = real) ---
        inputs_real = create_test_inputs(real_rows, real_rows, inner_dim)
        real_ms, real_std = time_it(
            functools.partial(run_kernel, module, inputs_real),
            args.dry_run_iters,
            args.repeat_iters,
        )
        print(
            f"  No padding  (padded={real_rows:>5}, total={real_rows:>5}): "
            f"{real_ms:.4f} ms"
        )

        # --- Padded with real totalPadded (production case) ---
        inputs_padded = create_test_inputs(padded_rows, real_rows, inner_dim)
        padded_ms, padded_std = time_it(
            functools.partial(run_kernel, module, inputs_padded),
            args.dry_run_iters,
            args.repeat_iters,
        )
        overhead = padded_ms / real_ms if real_ms > 0 else 0
        print(
            f"  Padded+real  (padded={padded_rows:>5}, total={real_rows:>5}): "
            f"{padded_ms:.4f} ms  ({overhead:.2f}x vs no-pad)"
        )

        # --- Padded with full compute (totalPadded = padded_rows) ---
        inputs_full = create_test_inputs(padded_rows, padded_rows, inner_dim)
        full_ms, full_std = time_it(
            functools.partial(run_kernel, module, inputs_full),
            args.dry_run_iters,
            args.repeat_iters,
        )
        skip_speedup = full_ms / padded_ms if padded_ms > 0 else 0
        print(
            f"  Full compute (padded={padded_rows:>5}, total={padded_rows:>5}): "
            f"{full_ms:.4f} ms  (early-exit saves {skip_speedup:.2f}x)"
        )

        # --- Correctness check ---
        correct = check_correctness(module, inputs_padded, real_rows)
        print(f"  Correctness (vs f32 reference): {'PASS' if correct else 'FAIL'}")

        # --- Grid Y sweep ---
        print(f"\n  Grid Y sweep (padded+real case):")
        grid_y_candidates = get_grid_y_candidates(padded_rows, inner_dim, sm_count)
        print(f"  {len(grid_y_candidates)} gridY values to test")
        print(
            f"  {'gridY':>8} {'ms':>9} {'vs_heur':>8} {'vs_nopad':>9} {'ok':>4}"
        )
        print(f"  {'-'*8} {'-'*9} {'-'*8} {'-'*9} {'-'*4}")

        sweep_results = []
        best = {"grid_y": -1, "median_ms": float("inf")}
        n_fail = 0

        for grid_y in grid_y_candidates:
            ms, std = time_it(
                functools.partial(run_kernel, module, inputs_padded, grid_y),
                args.dry_run_iters,
                args.repeat_iters,
            )
            vs_heur = padded_ms / ms if ms > 0 else 0
            vs_nopad = ms / real_ms if real_ms > 0 else 0

            cfg_ok = check_correctness(module, inputs_padded, real_rows, grid_y)
            if not cfg_ok:
                n_fail += 1

            sweep_results.append(
                {
                    "grid_y": grid_y,
                    "median_ms": ms,
                    "std_ms": std,
                    "correct": cfg_ok,
                }
            )

            if ms < best["median_ms"]:
                best = {"grid_y": grid_y, "median_ms": ms}

            ok_str = "Y" if cfg_ok else "FAIL"
            print(
                f"  {grid_y:>8} {ms:>9.4f} {vs_heur:>7.3f}x "
                f"{vs_nopad:>8.2f}x {ok_str:>4}"
            )

        if n_fail > 0:
            print(
                f"\n  WARNING: {n_fail}/{len(grid_y_candidates)} configs "
                f"produced incorrect output!"
            )

        best_vs_nopad = best["median_ms"] / real_ms if real_ms > 0 else 0
        best_vs_heur = padded_ms / best["median_ms"] if best["median_ms"] > 0 else 0
        print(
            f"\n  Best: gridY={best['grid_y']} "
            f"{best['median_ms']:.4f}ms "
            f"({best_vs_heur:.3f}x vs heuristic, "
            f"{best_vs_nopad:.2f}x vs no-pad)"
        )

        scenario_result = {
            "real_rows": real_rows,
            "padded_rows": padded_rows,
            "inner_dim": inner_dim,
            "no_padding_ms": real_ms,
            "padded_real_ms": padded_ms,
            "full_compute_ms": full_ms,
            "early_exit_speedup": skip_speedup,
            "padding_overhead": overhead,
            "correctness": correct,
            "sweep_failures": n_fail,
            "best_grid_y": best["grid_y"],
            "best_ms": best["median_ms"],
            "best_vs_heuristic": best_vs_heur,
            "best_vs_nopad": best_vs_nopad,
            "sweep_results": sweep_results,
        }

        out_file = os.path.join(
            args.output_dir,
            f"sweep_real{real_rows}_pad{padded_rows}_k{inner_dim}.json",
        )
        with open(out_file, "w") as f:
            json.dump(scenario_result, f, indent=2)
        all_results.append(scenario_result)

    # Summary
    print(f"\n{'='*78}")
    print("SUMMARY")
    print(f"{'='*78}")
    print(
        f"  {'real':>6} {'padded':>7} {'innerD':>6} {'no_pad':>8} "
        f"{'pad+real':>9} {'full':>8} {'best_ms':>8} {'best_gY':>8} "
        f"{'vs_nopad':>9}"
    )
    print(
        f"  {'-'*6} {'-'*7} {'-'*6} {'-'*8} {'-'*9} {'-'*8} "
        f"{'-'*8} {'-'*8} {'-'*9}"
    )
    for r in all_results:
        print(
            f"  {r['real_rows']:>6} {r['padded_rows']:>7} "
            f"{r['inner_dim']:>6} "
            f"{r['no_padding_ms']:>8.4f} {r['padded_real_ms']:>9.4f} "
            f"{r['full_compute_ms']:>8.4f} {r['best_ms']:>8.4f} "
            f"{r['best_grid_y']:>8} "
            f"{r['best_vs_nopad']:>8.2f}x"
        )

    csv_file = os.path.join(args.output_dir, "summary.csv")
    with open(csv_file, "w", newline="") as f:
        fieldnames = [
            "real_rows",
            "padded_rows",
            "inner_dim",
            "no_padding_ms",
            "padded_real_ms",
            "full_compute_ms",
            "early_exit_speedup",
            "best_ms",
            "best_grid_y",
            "best_vs_heuristic",
            "best_vs_nopad",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nSaved: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Grid sweep for activationDeepSeekKernelV2 (FP8 silu+mul+quantize)"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeat-iters", type=int, default=50)
    parser.add_argument("--dry-run-iters", type=int, default=10)
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
