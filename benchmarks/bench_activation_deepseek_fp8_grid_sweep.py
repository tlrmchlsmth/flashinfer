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
import functools
import numpy as np
import torch

from flashinfer.testing.utils import bench_gpu_time
from flashinfer.jit.fp8_activation_benchmark import gen_fp8_activation_benchmark_module

CUDA_GRAPH_ITERS = 200

# Production DeepSeek-R1 with EP_size=32:
# Per-expert intermediate_size=2048, innerDim = 2 * 2048 = 4096 for SwiGLU
# halfDim=2048, numScaleBlocks=16
# Each token routes to 6-10 experts; after EP32 + padding:
#   real_rows = actual totalNumPaddedTokens on this rank (read from device)
#   padded_rows = maxPermutedPaddedCount (workspace allocation, used for grid sizing)
#
# (real_rows, padded_rows, inner_dim)
PRODUCTION_SCENARIOS = [
    # Low load: 16-32 concurrent requests
    (96, 3072, 4096),
    (192, 3072, 4096),
    # Medium load
    (384, 3072, 4096),
    (512, 8192, 4096),
    # High load
    (768, 8192, 4096),
    (1024, 32768, 4096),
]


def get_sm_count():
    return torch.cuda.get_device_properties(0).multi_processor_count


def get_gpu_name():
    return torch.cuda.get_device_name(0)


def get_module():
    return gen_fp8_activation_benchmark_module().build_and_load()


def create_test_inputs(padded_rows, real_rows, inner_dim, device="cuda"):
    """Create inputs matching activationDeepSeekKernelV2 data layout.

    The kernel reads sfStride = totalNumPaddedTokens[0] = real_rows.
    Scale tensors must be laid out with stride = real_rows (not padded_rows).

    Scale access pattern: scales[row + sfStride * scaleBlockIdx]
      = scales[scaleBlockIdx * real_rows + row]

    Input layout:
      input: [padded_rows, inner_dim] fp8_e4m3 (workspace-sized, only first real_rows used)
      input_scales: 1D float32 tensor, laid out with stride = real_rows

    Output layout:
      output: [padded_rows, inner_dim/2] fp8_e4m3
      output_scales: 1D float32 tensor

    total_padded_tokens: [1] int32 on device = real_rows
    """
    half_dim = inner_dim // 2

    # Generate bf16 random data, quantize to fp8 with scales
    x_bf16 = torch.randn(padded_rows, inner_dim, dtype=torch.bfloat16, device=device)

    # Compute per-128-block scales for the first real_rows rows
    num_input_scale_blocks = inner_dim // 128
    # scales_2d[sb, row] = scale for scaleBlock sb, row row
    scales_2d = torch.zeros(num_input_scale_blocks, padded_rows, dtype=torch.float32, device=device)
    for sb in range(num_input_scale_blocks):
        col_start = sb * 128
        col_end = col_start + 128
        block_data = x_bf16[:real_rows, col_start:col_end].float()
        block_amax = block_data.abs().amax(dim=1).clamp(min=1e-12)
        scales_2d[sb, :real_rows] = block_amax / 448.0

    # Pack scales with stride = real_rows: flat[sb * real_rows + row]
    total_scale_alloc = num_input_scale_blocks * padded_rows
    input_scales_flat = torch.zeros(total_scale_alloc, dtype=torch.float32, device=device)
    for sb in range(num_input_scale_blocks):
        input_scales_flat[sb * real_rows : sb * real_rows + real_rows] = scales_2d[sb, :real_rows]

    # Quantize input to fp8 (only first real_rows rows need correct quantization)
    x_fp8 = torch.empty(padded_rows, inner_dim, dtype=torch.float8_e4m3fn, device=device)
    for sb in range(num_input_scale_blocks):
        col_start = sb * 128
        col_end = col_start + 128
        scale = scales_2d[sb, :real_rows].unsqueeze(1)  # [real_rows, 1]
        x_fp8[:real_rows, col_start:col_end] = (
            x_bf16[:real_rows, col_start:col_end].float() / scale
        ).to(torch.float8_e4m3fn)

    # Output tensors
    num_output_scale_blocks = half_dim // 128
    output = torch.empty(padded_rows, half_dim, dtype=torch.float8_e4m3fn, device=device)
    output_scales = torch.zeros(
        num_output_scale_blocks * padded_rows, dtype=torch.float32, device=device
    )

    # Device tensor for totalNumPaddedTokens
    total_padded = torch.tensor([real_rows], dtype=torch.int32, device=device)

    # V1 production mapping: topK=8, EP32 -> ~1/4 of expanded entries are real
    # numTokens such that numTokens * (topK / num_experts_per_rank) ~ real_rows
    # With 256 experts, EP32 -> 8 local experts, topK=8 -> ~8/256*8 = 0.25 real per token
    # But simpler: numTokens * topK total entries, real_rows of them are real
    v1_top_k = 8
    # In production: numTokens = total tokens across all requests on this rank
    # Each token routes to topK=8 experts out of 256, this rank handles 8 experts
    # Expected real entries: numTokens * topK * (local_experts / total_experts)
    #                      = numTokens * 8 * (8/256) = numTokens * 0.25
    # So numTokens = real_rows / 0.25 = real_rows * 4
    v1_num_tokens = real_rows * 4
    v1_expanded = torch.full(
        (v1_num_tokens * v1_top_k,), -1, dtype=torch.int32, device=device
    )
    # Scatter real_rows valid entries randomly across the expanded array
    valid_positions = torch.randperm(v1_num_tokens * v1_top_k, device=device)[:real_rows]
    v1_expanded[valid_positions] = torch.arange(real_rows, dtype=torch.int32, device=device)

    return {
        "input": x_fp8,
        "input_scales": input_scales_flat,
        "output": output,
        "output_scales": output_scales,
        "total_padded": total_padded,
        "expanded_idx": v1_expanded,
        "v1_num_tokens": v1_num_tokens,
        "v1_top_k": v1_top_k,
        "inner_dim": inner_dim,
        "real_rows": real_rows,
        "padded_rows": padded_rows,
        # For correctness checking
        "x_bf16": x_bf16,
        "scales_2d": scales_2d,  # [numBlocks, padded_rows]
    }


def run_kernel(module, inputs, grid_y=-1, version="v2"):
    fn_base = f"activation_deepseek_fp8_{version}"
    if version == "v1":
        module.activation_deepseek_fp8_v1(
            inputs["input"],
            inputs["input_scales"],
            inputs["output"],
            inputs["output_scales"],
            inputs["total_padded"],
            inputs["expanded_idx"],
            inputs["inner_dim"],
            inputs["v1_num_tokens"],
            inputs["v1_top_k"],
        )
    elif grid_y > 0:
        getattr(module, fn_base + "_tuned")(
            inputs["input"],
            inputs["input_scales"],
            inputs["output"],
            inputs["output_scales"],
            inputs["total_padded"],
            inputs["inner_dim"],
            grid_y,
        )
    else:
        getattr(module, fn_base)(
            inputs["input"],
            inputs["input_scales"],
            inputs["output"],
            inputs["output_scales"],
            inputs["total_padded"],
            inputs["inner_dim"],
        )


def compute_reference(inputs):
    """Compute reference silu+mul output in f32 for correctness checking."""
    inner_dim = inputs["inner_dim"]
    half_dim = inner_dim // 2
    real_rows = inputs["real_rows"]
    scales_2d = inputs["scales_2d"]  # [numBlocks, padded_rows]
    x_fp8 = inputs["input"]

    # Dequantize: for each block, x_f32 = fp8_val * scale
    x_f32 = torch.zeros(real_rows, inner_dim, dtype=torch.float32, device=x_fp8.device)
    num_blocks = inner_dim // 128
    for sb in range(num_blocks):
        col_start = sb * 128
        col_end = col_start + 128
        scale = scales_2d[sb, :real_rows].unsqueeze(1)  # [real_rows, 1]
        x_f32[:, col_start:col_end] = x_fp8[:real_rows, col_start:col_end].float() * scale

    # SwiGLU: silu(x2) * x1
    x1 = x_f32[:, :half_dim]
    x2 = x_f32[:, half_dim:]
    ref_out = torch.nn.functional.silu(x2) * x1
    return ref_out


def check_correctness(module, inputs, grid_y=-1, version="v2"):
    """Verify kernel output matches reference within FP8 quantization tolerance."""
    run_kernel(module, inputs, grid_y, version)
    torch.cuda.synchronize()

    ref_f32 = compute_reference(inputs)

    inner_dim = inputs["inner_dim"]
    half_dim = inner_dim // 2
    real_rows = inputs["real_rows"]
    num_out_blocks = half_dim // 128

    # Dequantize kernel output to f32
    # Output scales use stride = real_rows (sfStride = totalPadded = real_rows)
    kernel_out_fp8 = inputs["output"][:real_rows]
    out_scales_flat = inputs["output_scales"]

    kernel_out_f32 = torch.zeros(real_rows, half_dim, dtype=torch.float32, device=kernel_out_fp8.device)
    for sb in range(num_out_blocks):
        col_start = sb * 128
        col_end = col_start + 128
        # Scale at index [sb * real_rows + row] (stride = real_rows)
        scale = out_scales_flat[sb * real_rows : sb * real_rows + real_rows].unsqueeze(1)
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

        # --- V1 production baseline (no padding, expandedIdx indirection) ---
        inputs_real = create_test_inputs(real_rows, real_rows, inner_dim)
        v1_ms, _ = time_it(
            functools.partial(run_kernel, module, inputs_real, -1, "v1"),
            args.dry_run_iters,
            args.repeat_iters,
        )
        print(
            f"  V1 prod     (rows={real_rows:>5}, topK=1, no padding): "
            f"{v1_ms:.4f} ms"
        )

        # --- V2 padded baseline (activationDeepSeekKernelV2 with prod grid heuristic) ---
        inputs_padded = create_test_inputs(padded_rows, real_rows, inner_dim)
        padded_ms, _ = time_it(
            functools.partial(run_kernel, module, inputs_padded, -1, "v2"),
            args.dry_run_iters,
            args.repeat_iters,
        )
        print(
            f"  V2 padded   (padded={padded_rows:>5}, total={real_rows:>5}): "
            f"{padded_ms:.4f} ms  ({padded_ms / v1_ms:.2f}x vs V1)"
        )

        # --- V4 no-pad baseline (apples-to-apples for pad overhead) ---
        v4_real_ms, _ = time_it(
            functools.partial(run_kernel, module, inputs_real, -1, "v4"),
            args.dry_run_iters,
            args.repeat_iters,
        )
        print(
            f"  V4 no-pad   (padded={real_rows:>5}, total={real_rows:>5}): "
            f"{v4_real_ms:.4f} ms"
        )

        # --- Correctness checks ---
        for ver in ["v2", "v3", "v4", "v5"]:
            ok = check_correctness(module, inputs_padded, version=ver)
            print(f"  {ver.upper()} correctness: {'PASS' if ok else 'FAIL'}")

        # --- V3/V4 default grid comparison ---
        version_results = {}
        for ver in ["v3", "v4", "v5"]:
            ms, std = time_it(
                functools.partial(run_kernel, module, inputs_padded, -1, ver),
                args.dry_run_iters,
                args.repeat_iters,
            )
            vs_v2 = padded_ms / ms if ms > 0 else 0
            vs_nopad = ms / v4_real_ms if v4_real_ms > 0 else 0
            version_results[ver] = {"default_ms": ms}
            print(
                f"  {ver.upper()} default    (padded={padded_rows:>5}, total={real_rows:>5}): "
                f"{ms:.4f} ms  ({vs_v2:.2f}x vs V2 heur, "
                f"{vs_nopad:.2f}x vs no-pad)"
            )

        # --- Grid Y sweep for V3 and V4 ---
        grid_y_candidates = get_grid_y_candidates(padded_rows, inner_dim, sm_count)

        for ver in ["v3", "v4", "v5"]:
            print(f"\n  {ver.upper()} Grid Y sweep ({len(grid_y_candidates)} values):")
            print(
                f"  {'gridY':>8} {'ms':>9} {'vs_v2h':>8} {'vs_nopad':>9} {'ok':>4}"
            )
            print(f"  {'-'*8} {'-'*9} {'-'*8} {'-'*9} {'-'*4}")

            sweep = []
            best_ver = {"grid_y": -1, "median_ms": float("inf")}

            for grid_y in grid_y_candidates:
                ms, std = time_it(
                    functools.partial(run_kernel, module, inputs_padded, grid_y, ver),
                    args.dry_run_iters,
                    args.repeat_iters,
                )
                vs_heur = padded_ms / ms if ms > 0 else 0
                vs_nopad = ms / v4_real_ms if v4_real_ms > 0 else 0

                cfg_ok = check_correctness(module, inputs_padded, grid_y, ver)
                sweep.append({"grid_y": grid_y, "median_ms": ms, "std_ms": std, "correct": cfg_ok})
                if ms < best_ver["median_ms"]:
                    best_ver = {"grid_y": grid_y, "median_ms": ms}

                ok_str = "Y" if cfg_ok else "FAIL"
                print(f"  {grid_y:>8} {ms:>9.4f} {vs_heur:>7.3f}x {vs_nopad:>8.2f}x {ok_str:>4}")

            best_vs_nopad = best_ver["median_ms"] / v4_real_ms if v4_real_ms > 0 else 0
            best_vs_heur = padded_ms / best_ver["median_ms"] if best_ver["median_ms"] > 0 else 0
            print(
                f"\n  Best {ver.upper()}: gridY={best_ver['grid_y']} "
                f"{best_ver['median_ms']:.4f}ms "
                f"({best_vs_heur:.3f}x vs V2 heur, {best_vs_nopad:.2f}x vs no-pad)"
            )
            version_results[ver]["best_grid_y"] = best_ver["grid_y"]
            version_results[ver]["best_ms"] = best_ver["median_ms"]
            version_results[ver]["sweep"] = sweep

        scenario_result = {
            "real_rows": real_rows,
            "padded_rows": padded_rows,
            "inner_dim": inner_dim,
            "v1_prod_ms": v1_ms,
            "v2_padded_ms": padded_ms,
            "v4_nopad_ms": v4_real_ms,
        }
        for ver in ["v3", "v4", "v5"]:
            vr = version_results[ver]
            scenario_result[f"{ver}_default_ms"] = vr["default_ms"]
            scenario_result[f"{ver}_best_grid_y"] = vr["best_grid_y"]
            scenario_result[f"{ver}_best_ms"] = vr["best_ms"]

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
        f"  {'real':>6} {'pad':>6} "
        f"{'V1_prod':>8} {'V2_pad':>8} "
        f"{'V4_np':>8} {'V4_best':>8} {'pad%':>6} "
        f"{'vs_V1':>7}"
    )
    print(
        f"  {'-'*6} {'-'*6} "
        f"{'-'*8} {'-'*8} "
        f"{'-'*8} {'-'*8} {'-'*6} "
        f"{'-'*7}"
    )
    for r in all_results:
        v4_best = min(r.get("v4_best_ms", float("inf")), r.get("v5_best_ms", float("inf")))
        v4_pad_pct = (v4_best / r["v4_nopad_ms"] - 1) * 100 if r["v4_nopad_ms"] > 0 else 0
        vs_v1 = r["v1_prod_ms"] / v4_best if v4_best > 0 else 0
        print(
            f"  {r['real_rows']:>6} {r['padded_rows']:>6} "
            f"{r['v1_prod_ms']:>8.4f} {r['v2_padded_ms']:>8.4f} "
            f"{r['v4_nopad_ms']:>8.4f} {v4_best:>8.4f} {v4_pad_pct:>5.0f}% "
            f"{vs_v1:>6.1f}x"
        )

    csv_file = os.path.join(args.output_dir, "summary.csv")
    with open(csv_file, "w", newline="") as f:
        fieldnames = [
            "real_rows", "padded_rows", "inner_dim",
            "v1_prod_ms", "v2_padded_ms", "v4_nopad_ms",
        ]
        for ver in ["v3", "v4", "v5"]:
            fieldnames += [f"{ver}_default_ms", f"{ver}_best_ms", f"{ver}_best_grid_y"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
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
