"""
JIT module builder for the FP8 activation benchmark wrapper.

Compiles csrc/fp8_activation_benchmark.cu which contains
activationDeepSeekKernelV2 and TVM-FFI wrappers for benchmarking.
"""

from .core import gen_jit_spec, current_compilation_context
from . import env as jit_env


def gen_fp8_activation_benchmark_module():
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 12]
    )
    return gen_jit_spec(
        "fp8_activation_benchmark",
        [
            jit_env.FLASHINFER_CSRC_DIR / "fp8_activation_benchmark.cu",
        ],
        extra_cuda_cflags=nvcc_flags
        + [
            "-DENABLE_BF16",
            "-DENABLE_FP8",
        ],
        extra_cflags=[
            "-DENABLE_BF16",
            "-DENABLE_FP8",
        ],
    )
