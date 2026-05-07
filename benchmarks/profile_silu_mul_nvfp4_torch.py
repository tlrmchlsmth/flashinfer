"""Profile silu_and_mul_scaled_nvfp4_experts_quantize with PyTorch profiler.

No elevated permissions needed (unlike ncu).

Usage:
    python profile_silu_mul_nvfp4_torch.py
"""

import torch
from torch.profiler import profile, ProfilerActivity
from flashinfer.quantization.fp4_quantization import (
    gen_fp4_quantization_sm100_module,
    gen_fp4_quantization_sm110_module,
    gen_fp4_quantization_sm120_module,
)

FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
FLOAT4_E2M1_MAX = 6.0
K = 2048
N_EXPERTS = 8
SF_VEC_SIZE = 16


def get_module():
    major, minor = torch.cuda.get_device_capability()
    cc = major * 10 + minor
    if cc >= 120:
        return gen_fp4_quantization_sm120_module().build_and_load()
    elif cc >= 110:
        return gen_fp4_quantization_sm110_module().build_and_load()
    else:
        return gen_fp4_quantization_sm100_module().build_and_load()


def round_up(x, m):
    return (x + m - 1) // m * m


def profile_shape(module, m_topk, real_tokens):
    tokens_per_expert = m_topk // N_EXPERTS
    real_per_expert = real_tokens // N_EXPERTS
    k_by_2 = K * 2

    x = torch.randn(N_EXPERTS, tokens_per_expert, k_by_2, dtype=torch.bfloat16, device="cuda")
    mask = torch.full((N_EXPERTS,), real_per_expert, dtype=torch.int32, device="cuda")

    ref_silu = torch.nn.functional.silu(x[..., :K]) * x[..., K:]
    tensor_amax = ref_silu.abs().amax(dim=(1, 2)).to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    scale_k = K // SF_VEC_SIZE
    padded_k = round_up(scale_k, 4)
    padded_k_int32 = padded_k // 4
    padded_m = round_up(tokens_per_expert, 128)
    output = torch.empty(N_EXPERTS, tokens_per_expert, K // 2, device="cuda", dtype=torch.uint8)
    output_scales = torch.empty(
        N_EXPERTS, padded_m, padded_k_int32, device="cuda", dtype=torch.int32
    )

    def run():
        module.silu_and_mul_scaled_nvfp4_experts_quantize(
            output.view(m_topk, K // 2),
            output_scales.view(N_EXPERTS * padded_m, padded_k_int32),
            x.view(m_topk, k_by_2),
            global_scale,
            mask,
            True,
        )

    # Warmup
    for _ in range(20):
        run()
    torch.cuda.synchronize()

    # Profile
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(50):
            run()
        torch.cuda.synchronize()

    print(f"\n=== m_topk={m_topk}, real={real_tokens} ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))

    # Also export chrome trace for detailed view
    import os
    output_dir = os.environ.get("OUTPUT_DIR", "/tmp")
    trace_path = os.path.join(output_dir, f"trace_m{m_topk}_r{real_tokens}.json")
    prof.export_chrome_trace(trace_path)
    print(f"Trace saved: {trace_path}")


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    module = get_module()

    scenarios = [
        (3072, 96),
        (3072, 384),
        (32768, 1024),
    ]

    for m_topk, real_tokens in scenarios:
        profile_shape(module, m_topk, real_tokens)


if __name__ == "__main__":
    main()
