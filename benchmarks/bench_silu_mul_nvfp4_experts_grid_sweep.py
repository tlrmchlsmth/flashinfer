"""
Grid/block sweep for silu_and_mul_scaled_nvfp4_experts_quantize kernel.

Measures kernel latency across a matrix of (grid_size, block_size) for each
production-representative token count, targeting EP_size=32 DeepSeek-R1 shapes.

Usage:
    # Full sweep (all production shapes)
    python bench_silu_mul_nvfp4_experts_grid_sweep.py --output-dir /path/to/results

    # Quick test with specific shapes
    python bench_silu_mul_nvfp4_experts_grid_sweep.py --output-dir ./results --m-topk 3072 32768

    # With correctness verification
    python bench_silu_mul_nvfp4_experts_grid_sweep.py --output-dir ./results --verify
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
from flashinfer.quantization.fp4_quantization import (
    gen_fp4_quantization_sm100_module,
    gen_fp4_quantization_sm110_module,
    gen_fp4_quantization_sm120_module,
)
from flashinfer.utils import is_sm100a_supported

FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
FLOAT4_E2M1_MAX = 6.0

# DeepSeek-R1 with EP_size=32
DEFAULT_K = 2048
DEFAULT_N_EXPERTS = 8
CVT_FP16_TO_FP4_ELTS_PER_THREAD = 16  # Blackwell (SM100+)

PRODUCTION_M_TOPK = [96, 192, 384, 768, 1536, 3072, 6144, 12288, 24576, 32768]


def get_sm_count():
    props = torch.cuda.get_device_properties(0)
    return props.multi_processor_count


def get_gpu_name():
    return torch.cuda.get_device_name(0)


def get_module():
    """Get the raw JIT module with TVM-FFI functions (not the SimpleNamespace wrapper)."""
    major, minor = torch.cuda.get_device_capability()
    cc = major * 10 + minor
    if cc >= 120:
        return gen_fp4_quantization_sm120_module().build_and_load()
    elif cc >= 110:
        return gen_fp4_quantization_sm110_module().build_and_load()
    elif cc >= 100:
        return gen_fp4_quantization_sm100_module().build_and_load()
    else:
        raise RuntimeError(f"SM{cc} does not support FP4 quantization (need SM100+)")


def round_up(x, multiple):
    return (x + multiple - 1) // multiple * multiple


def create_test_inputs(m_topk, k, n_experts, device="cuda", dtype=torch.bfloat16):
    k_by_2 = k * 2
    sf_vec_size = 16
    tokens_per_expert = m_topk // n_experts

    x = torch.randn(n_experts, tokens_per_expert, k_by_2, dtype=dtype, device=device)
    mask = torch.full((n_experts,), tokens_per_expert, dtype=torch.int32, device=device)

    ref_silu = torch.nn.functional.silu(x[..., :k]) * x[..., k:]
    tensor_amax = ref_silu.abs().amax(dim=(1, 2)).to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    scale_k = k // sf_vec_size
    padded_k = round_up(scale_k, 4)
    padded_k_int32 = padded_k // 4
    padded_m = round_up(tokens_per_expert, 128)

    output = torch.empty(n_experts, tokens_per_expert, k // 2, device=device, dtype=torch.uint8)
    output_scales = torch.empty(n_experts, padded_m, padded_k_int32, device=device, dtype=torch.int32)

    return {
        "input_flat": x.view(m_topk, k_by_2),
        "output_flat": output.view(m_topk, k // 2),
        "output_scales_flat": output_scales.view(n_experts * padded_m, padded_k_int32),
        "global_scale": global_scale,
        "mask": mask,
    }


def run_kernel(module, inputs, grid_size=-1, block_size=-1):
    module.silu_and_mul_scaled_nvfp4_experts_quantize_tuned(
        inputs["output_flat"],
        inputs["output_scales_flat"],
        inputs["input_flat"],
        inputs["global_scale"],
        inputs["mask"],
        True,
        grid_size,
        block_size,
    )


def run_kernel_baseline(module, inputs):
    module.silu_and_mul_scaled_nvfp4_experts_quantize(
        inputs["output_flat"],
        inputs["output_scales_flat"],
        inputs["input_flat"],
        inputs["global_scale"],
        inputs["mask"],
        True,
    )


def time_kernel(module, inputs, grid_size, block_size, dry_run_iters, repeat_iters):
    fn = functools.partial(run_kernel, module, inputs, grid_size, block_size)
    times = bench_gpu_time(
        fn,
        enable_cupti=True,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        cold_l2_cache=True,
        use_cuda_graph=False,
    )
    return float(np.median(times)), float(np.std(times))


def time_kernel_baseline(module, inputs, dry_run_iters, repeat_iters):
    fn = functools.partial(run_kernel_baseline, module, inputs)
    times = bench_gpu_time(
        fn,
        enable_cupti=True,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        cold_l2_cache=True,
        use_cuda_graph=False,
    )
    return float(np.median(times)), float(np.std(times))


def verify_output(module, inputs, grid_size, block_size, reference_output):
    run_kernel(module, inputs, grid_size, block_size)
    torch.cuda.synchronize()
    return torch.equal(inputs["output_flat"], reference_output)


def get_grid_candidates(m_topk, k, n_experts, max_grid=8192):
    work_per_row = max(1, k // CVT_FP16_TO_FP4_ELTS_PER_THREAD)
    total_work = m_topk * work_per_row
    candidates = []
    g = n_experts
    while g <= min(total_work, max_grid):
        candidates.append(g)
        if g < 64:
            g += n_experts
        elif g < 256:
            g += n_experts * 2
        elif g < 1024:
            g += n_experts * 4
        else:
            g += n_experts * 8
    return candidates


def get_block_candidates(k):
    work_per_row = max(1, k // CVT_FP16_TO_FP4_ELTS_PER_THREAD)
    candidates = []
    for b in [64, 128, 256, 512]:
        if b <= work_per_row:
            candidates.append(b)
    return candidates if candidates else [64]


def run_sweep_for_shape(module, m_topk, k, n_experts, args):
    print(f"\n{'='*70}")
    print(f"  m_topk={m_topk}  K={k}  n_experts={n_experts}")
    print(f"{'='*70}")

    inputs = create_test_inputs(m_topk, k, n_experts)

    # Baseline (heuristic)
    print("  Timing heuristic baseline...", flush=True)
    baseline_ms, baseline_std = time_kernel_baseline(
        module, inputs, args.dry_run_iters, args.repeat_iters
    )
    print(f"  Heuristic: {baseline_ms:.4f} ms (std={baseline_std:.4f})")

    if args.verify:
        run_kernel_baseline(module, inputs)
        torch.cuda.synchronize()
        reference_output = inputs["output_flat"].clone()
    else:
        reference_output = None

    grid_candidates = get_grid_candidates(m_topk, k, n_experts)
    block_candidates = get_block_candidates(k)

    total_configs = len(grid_candidates) * len(block_candidates)
    print(f"  Sweeping {total_configs} configs: "
          f"{len(grid_candidates)} grids x {len(block_candidates)} blocks")

    col_g, col_b, col_t, col_std, col_su, col_c = 10, 10, 12, 10, 10, 8
    print(f"  {'grid':>{col_g}} {'block':>{col_b}} {'median_ms':>{col_t}} "
          f"{'std_ms':>{col_std}} {'speedup':>{col_su}}"
          + (f" {'correct':>{col_c}}" if args.verify else ""))
    print(f"  {'-'*col_g} {'-'*col_b} {'-'*col_t} {'-'*col_std} {'-'*col_su}"
          + (f" {'-'*col_c}" if args.verify else ""))

    results = []
    best = {"grid_size": -1, "block_size": -1, "median_ms": float("inf")}
    idx = 0

    for block_size in block_candidates:
        for grid_size in grid_candidates:
            idx += 1
            median_ms, std_ms = time_kernel(
                module, inputs, grid_size, block_size,
                args.dry_run_iters, args.repeat_iters,
            )
            speedup = baseline_ms / median_ms if median_ms > 0 else 0

            correct = None
            if args.verify:
                correct = verify_output(module, inputs, grid_size, block_size, reference_output)

            result = {
                "grid_size": grid_size,
                "block_size": block_size,
                "median_ms": median_ms,
                "std_ms": std_ms,
                "speedup_vs_heuristic": speedup,
            }
            if correct is not None:
                result["correct"] = correct

            results.append(result)

            if median_ms < best["median_ms"]:
                best = {"grid_size": grid_size, "block_size": block_size, "median_ms": median_ms}

            correct_str = f" {'OK' if correct else 'FAIL':>{col_c}}" if correct is not None else ""
            print(f"  {grid_size:>{col_g}} {block_size:>{col_b}} {median_ms:>{col_t}.4f} "
                  f"{std_ms:>{col_std}.4f} {speedup:>{col_su}.3f}x{correct_str}")

    best_speedup = baseline_ms / best["median_ms"] if best["median_ms"] > 0 else 0
    print(f"\n  Best: grid={best['grid_size']} block={best['block_size']} "
          f"{best['median_ms']:.4f}ms ({best_speedup:.3f}x vs heuristic)")

    return {
        "metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "gpu_name": get_gpu_name(),
            "sm_count": get_sm_count(),
            "k": k,
            "n_experts": n_experts,
            "m_topk": m_topk,
            "tokens_per_expert": m_topk // n_experts,
            "elts_per_thread": CVT_FP16_TO_FP4_ELTS_PER_THREAD,
            "dtype": "bfloat16",
            "dry_run_iters": args.dry_run_iters,
            "repeat_iters": args.repeat_iters,
        },
        "heuristic": {
            "median_ms": baseline_ms,
            "std_ms": baseline_std,
        },
        "results": results,
        "best": {
            "grid_size": best["grid_size"],
            "block_size": best["block_size"],
            "median_ms": best["median_ms"],
            "speedup_vs_heuristic": best_speedup,
        },
    }


def run_sweep(args):
    if not is_sm100a_supported(torch.device("cuda")):
        print("ERROR: This benchmark requires SM100+ (Blackwell)")
        return

    print(f"GPU: {get_gpu_name()}")
    print(f"SM count: {get_sm_count()}")
    print(f"K={args.k}, n_experts={args.n_experts}")

    module = get_module()

    m_topk_values = args.m_topk or PRODUCTION_M_TOPK
    # m_topk must be divisible by n_experts
    m_topk_values = [m for m in m_topk_values if m % args.n_experts == 0]

    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows = []

    for m_topk in m_topk_values:
        result = run_sweep_for_shape(module, m_topk, args.k, args.n_experts, args)

        out_file = os.path.join(args.output_dir, f"sweep_m{m_topk}_k{args.k}.json")
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_file}")

        summary_rows.append({
            "m_topk": m_topk,
            "best_grid": result["best"]["grid_size"],
            "best_block": result["best"]["block_size"],
            "best_ms": result["best"]["median_ms"],
            "heuristic_ms": result["heuristic"]["median_ms"],
            "speedup": result["best"]["speedup_vs_heuristic"],
        })

    # Write summary CSV
    csv_file = os.path.join(args.output_dir, "summary.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved: {csv_file}")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'m_topk':>8} {'best_grid':>10} {'best_block':>11} "
          f"{'best_ms':>9} {'heur_ms':>9} {'speedup':>8}")
    print(f"  {'-'*8} {'-'*10} {'-'*11} {'-'*9} {'-'*9} {'-'*8}")
    for row in summary_rows:
        print(f"  {row['m_topk']:>8} {row['best_grid']:>10} {row['best_block']:>11} "
              f"{row['best_ms']:>9.4f} {row['heuristic_ms']:>9.4f} {row['speedup']:>7.3f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Grid/block sweep for silu_and_mul_scaled_nvfp4_experts_quantize"
    )
    parser.add_argument("--output-dir", required=True, help="Directory for result JSON/CSV files")
    parser.add_argument("--m-topk", nargs="+", type=int, default=None,
                        help="m_topk values to test (default: production range 96-32768)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Hidden dim (default: 2048)")
    parser.add_argument("--n-experts", type=int, default=DEFAULT_N_EXPERTS,
                        help="Local experts per GPU (default: 8)")
    parser.add_argument("--repeat-iters", type=int, default=50, help="Timing iterations")
    parser.add_argument("--dry-run-iters", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--verify", action="store_true",
                        help="Verify correctness of each config against heuristic baseline")
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
