"""
Minimal script to launch activationDeepSeekKernelV3 once for ncu profiling.

Usage:
    ncu --set full -o profile_v3 python profile_activation_deepseek_fp8.py
"""

import argparse
import torch
from flashinfer.jit.fp8_activation_benchmark import gen_fp8_activation_benchmark_module


def create_test_inputs(padded_rows, real_rows, inner_dim, device="cuda"):
    half_dim = inner_dim // 2
    x_bf16 = torch.randn(padded_rows, inner_dim, dtype=torch.bfloat16, device=device)

    num_input_scale_blocks = inner_dim // 128
    scales_2d = torch.zeros(num_input_scale_blocks, padded_rows, dtype=torch.float32, device=device)
    for sb in range(num_input_scale_blocks):
        col_start = sb * 128
        col_end = col_start + 128
        block_data = x_bf16[:real_rows, col_start:col_end].float()
        block_amax = block_data.abs().amax(dim=1).clamp(min=1e-12)
        scales_2d[sb, :real_rows] = block_amax / 448.0

    total_scale_alloc = num_input_scale_blocks * padded_rows
    input_scales_flat = torch.zeros(total_scale_alloc, dtype=torch.float32, device=device)
    for sb in range(num_input_scale_blocks):
        input_scales_flat[sb * real_rows : sb * real_rows + real_rows] = scales_2d[sb, :real_rows]

    x_fp8 = torch.empty(padded_rows, inner_dim, dtype=torch.float8_e4m3fn, device=device)
    for sb in range(num_input_scale_blocks):
        col_start = sb * 128
        col_end = col_start + 128
        scale = scales_2d[sb, :real_rows].unsqueeze(1)
        x_fp8[:real_rows, col_start:col_end] = (
            x_bf16[:real_rows, col_start:col_end].float() / scale
        ).to(torch.float8_e4m3fn)

    num_output_scale_blocks = half_dim // 128
    output = torch.empty(padded_rows, half_dim, dtype=torch.float8_e4m3fn, device=device)
    output_scales = torch.zeros(
        num_output_scale_blocks * padded_rows, dtype=torch.float32, device=device
    )
    total_padded = torch.tensor([real_rows], dtype=torch.int32, device=device)

    return {
        "input": x_fp8,
        "input_scales": input_scales_flat,
        "output": output,
        "output_scales": output_scales,
        "total_padded": total_padded,
        "inner_dim": inner_dim,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-rows", type=int, default=1024)
    parser.add_argument("--padded-rows", type=int, default=32768)
    parser.add_argument("--inner-dim", type=int, default=2048)
    parser.add_argument("--version", choices=["v2", "v3", "v4", "v5"], default="v5")
    parser.add_argument("--grid-y", type=int, default=-1)
    args = parser.parse_args()

    module = gen_fp8_activation_benchmark_module().build_and_load()
    inputs = create_test_inputs(args.padded_rows, args.real_rows, args.inner_dim)

    # Warmup outside profiling range
    fn_base = f"activation_deepseek_fp8_{args.version}"
    fn = getattr(module, fn_base + "_tuned") if args.grid_y > 0 else getattr(module, fn_base)
    call_args = [
        inputs["input"], inputs["input_scales"],
        inputs["output"], inputs["output_scales"],
        inputs["total_padded"], inputs["inner_dim"],
    ]
    if args.grid_y > 0:
        call_args.append(args.grid_y)

    for _ in range(3):
        fn(*call_args)
    torch.cuda.synchronize()

    # Profiled invocation
    torch.cuda.nvtx.range_push(f"activation_{args.version}")
    fn(*call_args)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    print(f"Profiled {args.version} kernel: real={args.real_rows} padded={args.padded_rows} "
          f"innerDim={args.inner_dim} gridY={args.grid_y}")


if __name__ == "__main__":
    main()
