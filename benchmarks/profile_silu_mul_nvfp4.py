"""Profile script for silu_and_mul_scaled_nvfp4_experts_quantize kernel.

Run with ncu:
    ncu --set full --kernel-name cvt_fp16_to_fp4_expert \
        --launch-skip 5 --launch-count 1 \
        -o profile_output \
        python profile_silu_mul_nvfp4.py

Or standalone (just runs the kernel for timing):
    python profile_silu_mul_nvfp4.py
"""

import torch
from flashinfer.quantization.fp4_quantization import (
    gen_fp4_quantization_sm100_module,
    gen_fp4_quantization_sm110_module,
    gen_fp4_quantization_sm120_module,
)

FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
FLOAT4_E2M1_MAX = 6.0

K = 2048
N_EXPERTS = 8
m_topk = 32768
real_tokens = 1024
tokens_per_expert = m_topk // N_EXPERTS
real_per_expert = real_tokens // N_EXPERTS
k_by_2 = K * 2
sf_vec_size = 16

major, minor = torch.cuda.get_device_capability()
cc = major * 10 + minor
if cc >= 120:
    module = gen_fp4_quantization_sm120_module().build_and_load()
elif cc >= 110:
    module = gen_fp4_quantization_sm110_module().build_and_load()
else:
    module = gen_fp4_quantization_sm100_module().build_and_load()

x = torch.randn(N_EXPERTS, tokens_per_expert, k_by_2, dtype=torch.bfloat16, device="cuda")
mask = torch.full((N_EXPERTS,), real_per_expert, dtype=torch.int32, device="cuda")

ref_silu = torch.nn.functional.silu(x[..., :K]) * x[..., K:]
tensor_amax = ref_silu.abs().amax(dim=(1, 2)).to(torch.float32)
global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

scale_k = K // sf_vec_size
padded_k = ((scale_k + 3) // 4) * 4
padded_k_int32 = padded_k // 4
padded_m = ((tokens_per_expert + 127) // 128) * 128
output = torch.empty(N_EXPERTS, tokens_per_expert, K // 2, device="cuda", dtype=torch.uint8)
output_scales = torch.empty(
    N_EXPERTS, padded_m, padded_k_int32, device="cuda", dtype=torch.int32
)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"m_topk={m_topk}, real={real_tokens}, K={K}, n_experts={N_EXPERTS}")
print(f"tokens_per_expert={tokens_per_expert}, real_per_expert={real_per_expert}")

for i in range(10):
    module.silu_and_mul_scaled_nvfp4_experts_quantize(
        output.view(m_topk, K // 2),
        output_scales.view(N_EXPERTS * padded_m, padded_k_int32),
        x.view(m_topk, k_by_2),
        global_scale,
        mask,
        True,
    )
torch.cuda.synchronize()
print("Done - 10 kernel invocations completed")
