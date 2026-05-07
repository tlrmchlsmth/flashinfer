"""
Grid/block sweep for silu_and_mul_scaled_nvfp4_experts_quantize kernel.

Measures kernel latency across production EP32 DeepSeek-R1 shapes, including
the cost of padding (real vs padded token counts with mask-based early exit).

Uses CUDA graphs with multiple replays per graph to eliminate launch overhead.

Usage:
    python bench_silu_mul_nvfp4_experts_grid_sweep.py --output-dir /path/to/results
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

DEFAULT_K = 2048
DEFAULT_N_EXPERTS = 8
CVT_FP16_TO_FP4_ELTS_PER_THREAD = 16  # Blackwell (SM100+)
CUDA_GRAPH_ITERS = 200  # replays within each CUDA graph capture

# Production scenarios: (real_tokens, padded_tokens)
# EP_size=32, so worst-case padding = 32x
PRODUCTION_SCENARIOS = [
    (96, 3072),
    (192, 3072),
    (384, 3072),
    (512, 8192),
    (768, 8192),
    (1024, 32768),
]


def get_sm_count():
    return torch.cuda.get_device_properties(0).multi_processor_count


def get_gpu_name():
    return torch.cuda.get_device_name(0)


def get_module():
    """Get the raw JIT module with TVM-FFI functions."""
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


def create_test_inputs(m_topk, real_tokens, k, n_experts, device="cuda", dtype=torch.bfloat16):
    """Create inputs with m_topk padded rows but mask set to real_tokens."""
    k_by_2 = k * 2
    sf_vec_size = 16
    tokens_per_expert = m_topk // n_experts
    real_per_expert = real_tokens // n_experts

    x = torch.randn(n_experts, tokens_per_expert, k_by_2, dtype=dtype, device=device)
    mask = torch.full((n_experts,), real_per_expert, dtype=torch.int32, device=device)

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


def check_correctness_same_shape(module, inputs, n_experts, k, real_tokens, padded_tokens,
                                  grid_size=-1, block_size=-1):
    """Verify tuned kernel output matches heuristic baseline at the same shape."""
    tokens_per_expert = padded_tokens // n_experts
    real_per_expert = real_tokens // n_experts

    # Reference: heuristic baseline
    run_kernel_baseline(module, inputs)
    torch.cuda.synchronize()
    ref_out = inputs["output_flat"].clone()

    # Test: tuned kernel (or override)
    run_kernel(module, inputs, grid_size, block_size)
    torch.cuda.synchronize()
    test_out = inputs["output_flat"]

    # Compare only real (masked) rows per expert
    ok = True
    for e in range(n_experts):
        start = e * tokens_per_expert
        n = real_per_expert
        if not torch.equal(ref_out[start:start+n], test_out[start:start+n]):
            ok = False
            mismatches = (ref_out[start:start+n] != test_out[start:start+n]).sum().item()
            total = n * (k // 2)
            print(f"    MISMATCH expert {e}: {mismatches}/{total} bytes differ "
                  f"(grid={grid_size}, block={block_size})")
    return ok


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
    return [b for b in [64, 128, 256, 512] if b <= work_per_row] or [64]


def run_sweep(args):
    if not is_sm100a_supported(torch.device("cuda")):
        print("ERROR: This benchmark requires SM100+ (Blackwell)")
        return

    print(f"GPU: {get_gpu_name()}")
    print(f"SM count: {get_sm_count()}")
    print(f"K={args.k}, n_experts={args.n_experts}")
    print(f"CUDA graph replays per measurement: {CUDA_GRAPH_ITERS}")

    module = get_module()
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    for real_tokens, padded_tokens in PRODUCTION_SCENARIOS:
        print(f"\n{'='*78}")
        print(f"  real={real_tokens}  padded={padded_tokens}  "
              f"({padded_tokens // real_tokens}x padding)  "
              f"K={args.k}  n_experts={args.n_experts}")
        print(f"{'='*78}")

        # --- Baseline: no padding (m_topk = real, mask = real) ---
        inputs_real = create_test_inputs(real_tokens, real_tokens, args.k, args.n_experts)
        real_ms, real_std = time_it(
            functools.partial(run_kernel_baseline, module, inputs_real),
            args.dry_run_iters, args.repeat_iters,
        )
        print(f"  No padding  (m_topk={real_tokens:>5}, mask={real_tokens:>5}): "
              f"{real_ms:.4f} ms")

        # --- Padded with mask early-exit (m_topk = padded, mask = real) ---
        inputs_padded = create_test_inputs(padded_tokens, real_tokens, args.k, args.n_experts)
        padded_masked_ms, padded_masked_std = time_it(
            functools.partial(run_kernel_baseline, module, inputs_padded),
            args.dry_run_iters, args.repeat_iters,
        )
        mask_overhead = padded_masked_ms / real_ms if real_ms > 0 else 0
        print(f"  Padded+mask  (m_topk={padded_tokens:>5}, mask={real_tokens:>5}): "
              f"{padded_masked_ms:.4f} ms  ({mask_overhead:.2f}x vs no-pad)")

        # --- Padded without mask skip (m_topk = padded, mask = padded = full compute) ---
        inputs_full = create_test_inputs(padded_tokens, padded_tokens, args.k, args.n_experts)
        full_ms, full_std = time_it(
            functools.partial(run_kernel_baseline, module, inputs_full),
            args.dry_run_iters, args.repeat_iters,
        )
        skip_speedup = full_ms / padded_masked_ms if padded_masked_ms > 0 else 0
        print(f"  Full compute (m_topk={padded_tokens:>5}, mask={padded_tokens:>5}): "
              f"{full_ms:.4f} ms  (mask saves {skip_speedup:.2f}x)")

        # --- Correctness: new heuristic must match old baseline at same shape ---
        correct = check_correctness_same_shape(
            module, inputs_padded, args.n_experts, args.k,
            real_tokens, padded_tokens,
        )
        print(f"  Correctness (new heuristic vs baseline): {'PASS' if correct else 'FAIL'}")

        # --- Grid sweep on the padded+masked case ---
        print(f"\n  Grid sweep (padded+masked):")
        grid_candidates = get_grid_candidates(padded_tokens, args.k, args.n_experts)
        block_candidates = get_block_candidates(args.k)
        total_configs = len(grid_candidates) * len(block_candidates)
        print(f"  {total_configs} configs: {len(grid_candidates)} grids x {len(block_candidates)} blocks")

        # Compute reference output for correctness checks (heuristic at same shape)
        run_kernel_baseline(module, inputs_padded)
        torch.cuda.synchronize()
        ref_out = inputs_padded["output_flat"].clone()
        real_per_expert = real_tokens // args.n_experts
        tokens_per_expert_pad = padded_tokens // args.n_experts

        print(f"  {'grid':>8} {'block':>6} {'ms':>9} {'vs_heur':>8} {'vs_nopad':>9} {'ok':>4}")
        print(f"  {'-'*8} {'-'*6} {'-'*9} {'-'*8} {'-'*9} {'-'*4}")

        sweep_results = []
        best = {"grid_size": -1, "block_size": -1, "median_ms": float("inf")}
        n_fail = 0

        for block_size in block_candidates:
            for grid_size in grid_candidates:
                ms, std = time_it(
                    functools.partial(run_kernel, module, inputs_padded, grid_size, block_size),
                    args.dry_run_iters, args.repeat_iters,
                )
                vs_heur = padded_masked_ms / ms if ms > 0 else 0
                vs_nopad = ms / real_ms if real_ms > 0 else 0

                # Correctness: run once outside graph, compare real rows vs heuristic
                run_kernel(module, inputs_padded, grid_size, block_size)
                torch.cuda.synchronize()
                cfg_ok = True
                for e in range(args.n_experts):
                    s = e * tokens_per_expert_pad
                    n = real_per_expert
                    if not torch.equal(ref_out[s:s+n], inputs_padded["output_flat"][s:s+n]):
                        cfg_ok = False
                        break
                if not cfg_ok:
                    n_fail += 1

                sweep_results.append({
                    "grid_size": grid_size,
                    "block_size": block_size,
                    "median_ms": ms,
                    "std_ms": std,
                    "correct": cfg_ok,
                })

                if ms < best["median_ms"]:
                    best = {"grid_size": grid_size, "block_size": block_size, "median_ms": ms}

                ok_str = "Y" if cfg_ok else "FAIL"
                print(f"  {grid_size:>8} {block_size:>6} {ms:>9.4f} {vs_heur:>7.3f}x {vs_nopad:>8.2f}x {ok_str:>4}")

        if n_fail > 0:
            print(f"\n  WARNING: {n_fail}/{total_configs} configs produced incorrect output!")

        best_vs_nopad = best["median_ms"] / real_ms if real_ms > 0 else 0
        best_vs_heur = padded_masked_ms / best["median_ms"] if best["median_ms"] > 0 else 0
        print(f"\n  Best: grid={best['grid_size']} block={best['block_size']} "
              f"{best['median_ms']:.4f}ms "
              f"({best_vs_heur:.3f}x vs heuristic, {best_vs_nopad:.2f}x vs no-pad)")

        scenario_result = {
            "real_tokens": real_tokens,
            "padded_tokens": padded_tokens,
            "no_padding_ms": real_ms,
            "padded_masked_ms": padded_masked_ms,
            "full_compute_ms": full_ms,
            "mask_skip_speedup": skip_speedup,
            "padding_overhead": mask_overhead,
            "heuristic_correct": correct,
            "sweep_failures": n_fail,
            "best_grid": best["grid_size"],
            "best_block": best["block_size"],
            "best_ms": best["median_ms"],
            "best_vs_heuristic": best_vs_heur,
            "best_vs_nopad": best_vs_nopad,
            "sweep_results": sweep_results,
        }

        out_file = os.path.join(
            args.output_dir,
            f"sweep_real{real_tokens}_pad{padded_tokens}_k{args.k}.json",
        )
        with open(out_file, "w") as f:
            json.dump(scenario_result, f, indent=2)
        all_results.append(scenario_result)

    # Summary
    print(f"\n{'='*78}")
    print("SUMMARY")
    print(f"{'='*78}")
    print(f"  {'real':>6} {'padded':>7} {'no_pad':>8} {'pad+mask':>9} {'full':>8} "
          f"{'mask_skip':>10} {'best_ms':>8} {'best_grid':>10} {'vs_nopad':>9}")
    print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*9} {'-'*8} "
          f"{'-'*10} {'-'*8} {'-'*10} {'-'*9}")
    for r in all_results:
        print(f"  {r['real_tokens']:>6} {r['padded_tokens']:>7} "
              f"{r['no_padding_ms']:>8.4f} {r['padded_masked_ms']:>9.4f} "
              f"{r['full_compute_ms']:>8.4f} {r['mask_skip_speedup']:>9.2f}x "
              f"{r['best_ms']:>8.4f} {r['best_grid']:>10} "
              f"{r['best_vs_nopad']:>8.2f}x")

    csv_file = os.path.join(args.output_dir, "summary.csv")
    with open(csv_file, "w", newline="") as f:
        fieldnames = [
            "real_tokens", "padded_tokens", "no_padding_ms", "padded_masked_ms",
            "full_compute_ms", "mask_skip_speedup", "best_ms", "best_grid",
            "best_block", "best_vs_heuristic", "best_vs_nopad",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nSaved: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Grid/block sweep for silu_and_mul_scaled_nvfp4_experts_quantize"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--n-experts", type=int, default=DEFAULT_N_EXPERTS)
    parser.add_argument("--repeat-iters", type=int, default=50)
    parser.add_argument("--dry-run-iters", type=int, default=10)
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
