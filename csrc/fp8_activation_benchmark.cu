/*
 * Standalone benchmark wrapper for activationDeepSeekKernelV2.
 *
 * Kernel source: csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu:344
 * This file contains an exact copy of the kernel plus TVM-FFI wrappers for
 * benchmarking with configurable grid dimensions.
 *
 * Data flow: FP8_E4M3 input + per-128-block scales -> f32 dequant -> silu(x2)*x1
 *            -> per-128-block max reduce -> FP8_E4M3 quantize + output scales
 */

#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

#include <cutlass/numeric_types.h>

#include "tvm_ffi_utils.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int ACTIVATION_THREADS_PER_CTA = 128;
constexpr int ELTS_PER_SCALE_BLOCK = 128;

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Exact copy of activationDeepSeekKernelV2 from
// csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu:344
// Keep in sync with the original when applying optimizations.
////////////////////////////////////////////////////////////////////////////////////////////////////

struct ActivationDeepSeekParams {
  cutlass::float_e4m3_t const* inPtr;
  cutlass::float_e4m3_t* outPtr;
  float* inDqSfsPtr;
  float* outDqSfsPtr;
  int32_t innerDim;
  int32_t const* totalNumPaddedTokens;
};

__global__ void __launch_bounds__(ACTIVATION_THREADS_PER_CTA)
    activationDeepSeekKernelV2(ActivationDeepSeekParams params) {
  using BlockReduce = cub::BlockReduce<float, ACTIVATION_THREADS_PER_CTA>;

  __shared__ float s_scaleOut;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  float constexpr E4m3MaxVal{448.f};
  int const totalPadded = params.totalNumPaddedTokens[0];
  int const sfStride = totalPadded;

  int const hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x;
  int const halfDim = params.innerDim / 2;
  if (hiddenIdx >= halfDim) return;

  // Hoist loop-invariant scale index bases
  int const scaleBlockIdx = hiddenIdx / 128;
  int64_t const scale1Base = (int64_t)sfStride * scaleBlockIdx;
  int64_t const scale2Base =
      (int64_t)sfStride * (scaleBlockIdx + halfDim / 128);

  for (int permutedRow = blockIdx.y; permutedRow < totalPadded;
       permutedRow += gridDim.y) {
    int64_t const rowOffset = (int64_t)permutedRow * params.innerDim;
    float scale1 = params.inDqSfsPtr[permutedRow + scale1Base];
    float scale2 = params.inDqSfsPtr[permutedRow + scale2Base];
    float x1 =
        scale1 * static_cast<float>(params.inPtr[rowOffset + hiddenIdx]);
    float x2 = scale2 * static_cast<float>(
                             params.inPtr[rowOffset + halfDim + hiddenIdx]);

    float out = silu_f(x2) * x1;
    float absOut = fabsf(out);

#if CUDA_VERSION >= 12090
    float aMax =
        BlockReduce(tempStorage).Reduce(absOut, cuda::maximum<>{});
#else
    float aMax = BlockReduce(tempStorage).Reduce(absOut, cub::Max{});
#endif

    if (threadIdx.x == 0) {
      float scaleOut = fmaxf(aMax / E4m3MaxVal, FLT_MIN);
      s_scaleOut = scaleOut;
      params.outDqSfsPtr[permutedRow + scale1Base] = scaleOut;
    }
    __syncthreads();

    int64_t const outIdx = (int64_t)permutedRow * halfDim + hiddenIdx;
    params.outPtr[outIdx] =
        static_cast<cutlass::float_e4m3_t>(out / s_scaleOut);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Launcher: computes production grid dimensions
////////////////////////////////////////////////////////////////////////////////////////////////////

static void launchActivationV2(
    cutlass::float_e4m3_t const* inPtr, cutlass::float_e4m3_t* outPtr,
    float* inDqSfsPtr, float* outDqSfsPtr, int32_t innerDim,
    int32_t const* totalNumPaddedTokens, int32_t maxPermutedPaddedCount,
    int32_t gridY_override, cudaStream_t stream) {
  int const outputDim = innerDim / 2;
  int const numScaleBlocks =
      (outputDim + ELTS_PER_SCALE_BLOCK - 1) / ELTS_PER_SCALE_BLOCK;
  int const gridSizeX = numScaleBlocks;

  int gridSizeY;
  if (gridY_override > 0) {
    gridSizeY = gridY_override;
  } else {
    gridSizeY = min(8192, max(1, maxPermutedPaddedCount));
  }

  dim3 grid(gridSizeX, gridSizeY, 1);

  ActivationDeepSeekParams params;
  params.inPtr = inPtr;
  params.outPtr = outPtr;
  params.inDqSfsPtr = inDqSfsPtr;
  params.outDqSfsPtr = outDqSfsPtr;
  params.innerDim = innerDim;
  params.totalNumPaddedTokens = totalNumPaddedTokens;

  activationDeepSeekKernelV2<<<grid, ACTIVATION_THREADS_PER_CTA, 0, stream>>>(
      params);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// TVM-FFI wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////

// activation_deepseek_fp8_v2(input, input_scales, output, output_scales,
//                            total_padded_tokens, inner_dim)
//
// input: [totalPadded, innerDim] fp8_e4m3
// input_scales: [numInputScaleBlocks * totalPadded] float32, column-major
//               numInputScaleBlocks = innerDim / 128
// output: [totalPadded, innerDim/2] fp8_e4m3
// output_scales: [numOutputScaleBlocks * totalPadded] float32, column-major
//                numOutputScaleBlocks = innerDim / 2 / 128
// total_padded_tokens: [1] int32 (device)
// inner_dim: int64 scalar

void activation_deepseek_fp8_v2(Tensor input, Tensor input_scales,
                                Tensor output, Tensor output_scales,
                                Tensor total_padded_tokens,
                                int64_t inner_dim) {
  CHECK_CUDA(input);
  CHECK_CUDA(input_scales);
  CHECK_CUDA(output);
  CHECK_CUDA(output_scales);
  CHECK_CUDA(total_padded_tokens);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(input_scales);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(output_scales);
  CHECK_CONTIGUOUS(total_padded_tokens);

  CHECK_INPUT_TYPE(input, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(input_scales, dl_float32);
  CHECK_INPUT_TYPE(output, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(output_scales, dl_float32);
  CHECK_INPUT_TYPE(total_padded_tokens, dl_int32);

  int32_t maxPermutedPaddedCount = static_cast<int32_t>(input.shape()[0]);
  cudaStream_t stream = get_stream(input.device());

  launchActivationV2(
      static_cast<cutlass::float_e4m3_t const*>(input.data_ptr()),
      static_cast<cutlass::float_e4m3_t*>(output.data_ptr()),
      static_cast<float*>(input_scales.data_ptr()),
      static_cast<float*>(output_scales.data_ptr()),
      static_cast<int32_t>(inner_dim),
      static_cast<int32_t const*>(total_padded_tokens.data_ptr()),
      maxPermutedPaddedCount, /*gridY_override=*/-1, stream);
}

// activation_deepseek_fp8_v2_tuned: same as above but with grid_y override
void activation_deepseek_fp8_v2_tuned(Tensor input, Tensor input_scales,
                                      Tensor output, Tensor output_scales,
                                      Tensor total_padded_tokens,
                                      int64_t inner_dim,
                                      int64_t grid_y_override) {
  CHECK_CUDA(input);
  CHECK_CUDA(input_scales);
  CHECK_CUDA(output);
  CHECK_CUDA(output_scales);
  CHECK_CUDA(total_padded_tokens);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(input_scales);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(output_scales);
  CHECK_CONTIGUOUS(total_padded_tokens);

  CHECK_INPUT_TYPE(input, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(input_scales, dl_float32);
  CHECK_INPUT_TYPE(output, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(output_scales, dl_float32);
  CHECK_INPUT_TYPE(total_padded_tokens, dl_int32);

  int32_t maxPermutedPaddedCount = static_cast<int32_t>(input.shape()[0]);
  cudaStream_t stream = get_stream(input.device());

  launchActivationV2(
      static_cast<cutlass::float_e4m3_t const*>(input.data_ptr()),
      static_cast<cutlass::float_e4m3_t*>(output.data_ptr()),
      static_cast<float*>(input_scales.data_ptr()),
      static_cast<float*>(output_scales.data_ptr()),
      static_cast<int32_t>(inner_dim),
      static_cast<int32_t const*>(total_padded_tokens.data_ptr()),
      maxPermutedPaddedCount,
      static_cast<int32_t>(grid_y_override), stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(activation_deepseek_fp8_v2,
                              activation_deepseek_fp8_v2);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(activation_deepseek_fp8_v2_tuned,
                              activation_deepseek_fp8_v2_tuned);
