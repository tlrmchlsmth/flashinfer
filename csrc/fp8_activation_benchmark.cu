/*
 * Benchmark wrapper for activationDeepSeekKernelV2 and optimized V3.
 *
 * V2: Original kernel (with hoisted loop invariants) — 128 threads, 1 elt/thread,
 *     cub::BlockReduce, shared memory + __syncthreads.
 *
 * V3: Vectorized kernel — 4 elts/thread, warp-level reduction via __shfl,
 *     one warp per 128-element scale block, no shared memory, no barriers.
 *     Inspired by cvt_fp16_to_fp4_expert in quantization.cuh.
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

constexpr int V2_THREADS_PER_CTA = 128;
constexpr int ELTS_PER_SCALE_BLOCK = 128;
constexpr int V3_ELTS_PER_THREAD = 4;
constexpr int V3_THREADS_PER_SCALE_BLOCK = ELTS_PER_SCALE_BLOCK / V3_ELTS_PER_THREAD;  // 32 = 1 warp
constexpr int V3_WARPS_PER_CTA = 8;
constexpr int V3_THREADS_PER_CTA = V3_WARPS_PER_CTA * 32;  // 256

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float warp_reduce_max(float val) {
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
  return val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// V2: Original kernel with hoisted loop invariants
////////////////////////////////////////////////////////////////////////////////////////////////////

struct ActivationDeepSeekParams {
  cutlass::float_e4m3_t const* inPtr;
  cutlass::float_e4m3_t* outPtr;
  float* inDqSfsPtr;
  float* outDqSfsPtr;
  int32_t innerDim;
  int32_t const* totalNumPaddedTokens;
};

__global__ void __launch_bounds__(V2_THREADS_PER_CTA)
    activationDeepSeekKernelV2(ActivationDeepSeekParams params) {
  using BlockReduce = cub::BlockReduce<float, V2_THREADS_PER_CTA>;

  __shared__ float s_scaleOut;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  float constexpr E4m3MaxVal{448.f};
  int const totalPadded = params.totalNumPaddedTokens[0];
  int const sfStride = totalPadded;

  int const hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x;
  int const halfDim = params.innerDim / 2;
  if (hiddenIdx >= halfDim) return;

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
// V3: Vectorized kernel — 4 elts/thread, warp-level reduction, no smem
//
// Thread mapping:
//   warpId = threadIdx.x / 32   → selects scale block within CTA
//   laneId = threadIdx.x % 32   → selects 4-element group within scale block
//   scaleBlock = blockIdx.x * V3_WARPS_PER_CTA + warpId
//
// Each warp independently handles one 128-element scale block:
//   32 lanes × 4 elements = 128 elements
//   Warp shuffle max reduction (no shared memory, no __syncthreads)
//   Lane 0 broadcasts output scale via __shfl
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(V3_THREADS_PER_CTA)
    activationDeepSeekKernelV3(ActivationDeepSeekParams params) {
  float constexpr E4m3MaxVal{448.f};
  int const totalPadded = params.totalNumPaddedTokens[0];
  int const sfStride = totalPadded;
  int const halfDim = params.innerDim / 2;
  int const numOutputScaleBlocks = halfDim / ELTS_PER_SCALE_BLOCK;

  int const warpId = threadIdx.x / 32;
  int const laneId = threadIdx.x % 32;
  int const scaleBlock = blockIdx.x * V3_WARPS_PER_CTA + warpId;

  if (scaleBlock >= numOutputScaleBlocks) return;

  int const elemBase = scaleBlock * ELTS_PER_SCALE_BLOCK + laneId * V3_ELTS_PER_THREAD;

  // Loop-invariant scale bases (column-major: scales[scaleBlock * sfStride + row])
  int64_t const scale1Base = (int64_t)sfStride * scaleBlock;
  int64_t const scale2Base = (int64_t)sfStride * (scaleBlock + numOutputScaleBlocks);

  using fp8_t = cutlass::float_e4m3_t;

  for (int permutedRow = blockIdx.y; permutedRow < totalPadded;
       permutedRow += gridDim.y) {
    // Load scales (broadcast within warp — all lanes read same address, L1 coalesces)
    float scale1 = params.inDqSfsPtr[permutedRow + scale1Base];
    float scale2 = params.inDqSfsPtr[permutedRow + scale2Base];

    // Vectorized load: 4 fp8 elements as uint32
    int64_t const x1Offset = (int64_t)permutedRow * params.innerDim + elemBase;
    int64_t const x2Offset = x1Offset + halfDim;

    uint32_t packed_x1 =
        *reinterpret_cast<uint32_t const*>(&params.inPtr[x1Offset]);
    uint32_t packed_x2 =
        *reinterpret_cast<uint32_t const*>(&params.inPtr[x2Offset]);

    // Unpack, dequantize, silu+mul, track max
    fp8_t x1_vals[4], x2_vals[4];
    memcpy(x1_vals, &packed_x1, 4);
    memcpy(x2_vals, &packed_x2, 4);

    float results[4];
    float localMax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float f1 = scale1 * static_cast<float>(x1_vals[i]);
      float f2 = scale2 * static_cast<float>(x2_vals[i]);
      results[i] = silu_f(f2) * f1;
      localMax = fmaxf(localMax, fabsf(results[i]));
    }

    // Warp-level max reduction (32 lanes → 1 max, 5 shuffle steps)
    float aMax = warp_reduce_max(localMax);

    // Lane 0 computes output scale
    float scaleOut;
    if (laneId == 0) {
      scaleOut = fmaxf(aMax / E4m3MaxVal, FLT_MIN);
      params.outDqSfsPtr[permutedRow + scale1Base] = scaleOut;
    }
    // Broadcast scale from lane 0 to all lanes
    scaleOut = __shfl_sync(0xffffffff, scaleOut, 0);

    // Quantize and vectorized store
    float invScale = 1.0f / scaleOut;
    fp8_t out_vals[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      out_vals[i] = static_cast<fp8_t>(results[i] * invScale);
    }
    uint32_t packed_out;
    memcpy(&packed_out, out_vals, 4);

    int64_t const outOffset =
        (int64_t)permutedRow * halfDim + elemBase;
    *reinterpret_cast<uint32_t*>(&params.outPtr[outOffset]) = packed_out;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Launchers
////////////////////////////////////////////////////////////////////////////////////////////////////

static void launchActivationV2(
    cutlass::float_e4m3_t const* inPtr, cutlass::float_e4m3_t* outPtr,
    float* inDqSfsPtr, float* outDqSfsPtr, int32_t innerDim,
    int32_t const* totalNumPaddedTokens, int32_t maxPermutedPaddedCount,
    int32_t gridY_override, cudaStream_t stream) {
  int const numScaleBlocks =
      (innerDim / 2 + ELTS_PER_SCALE_BLOCK - 1) / ELTS_PER_SCALE_BLOCK;

  int gridSizeY = gridY_override > 0
                      ? gridY_override
                      : min(8192, max(1, maxPermutedPaddedCount));

  dim3 grid(numScaleBlocks, gridSizeY, 1);

  ActivationDeepSeekParams params;
  params.inPtr = inPtr;
  params.outPtr = outPtr;
  params.inDqSfsPtr = inDqSfsPtr;
  params.outDqSfsPtr = outDqSfsPtr;
  params.innerDim = innerDim;
  params.totalNumPaddedTokens = totalNumPaddedTokens;

  activationDeepSeekKernelV2<<<grid, V2_THREADS_PER_CTA, 0, stream>>>(params);
}

static void launchActivationV3(
    cutlass::float_e4m3_t const* inPtr, cutlass::float_e4m3_t* outPtr,
    float* inDqSfsPtr, float* outDqSfsPtr, int32_t innerDim,
    int32_t const* totalNumPaddedTokens, int32_t maxPermutedPaddedCount,
    int32_t gridY_override, cudaStream_t stream) {
  int const numScaleBlocks =
      (innerDim / 2 + ELTS_PER_SCALE_BLOCK - 1) / ELTS_PER_SCALE_BLOCK;
  int const gridSizeX =
      (numScaleBlocks + V3_WARPS_PER_CTA - 1) / V3_WARPS_PER_CTA;

  int device{-1};
  cudaGetDevice(&device);
  int numSms = 0;
  cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);

  int gridSizeY = gridY_override > 0
                      ? gridY_override
                      : min(numSms, max(1, maxPermutedPaddedCount));

  dim3 grid(gridSizeX, gridSizeY, 1);

  ActivationDeepSeekParams params;
  params.inPtr = inPtr;
  params.outPtr = outPtr;
  params.inDqSfsPtr = inDqSfsPtr;
  params.outDqSfsPtr = outDqSfsPtr;
  params.innerDim = innerDim;
  params.totalNumPaddedTokens = totalNumPaddedTokens;

  activationDeepSeekKernelV3<<<grid, V3_THREADS_PER_CTA, 0, stream>>>(params);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// TVM-FFI wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////

// Shared validation
static void checkInputs(Tensor input, Tensor input_scales, Tensor output,
                         Tensor output_scales, Tensor total_padded_tokens) {
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
}

// V2 baseline
void activation_deepseek_fp8_v2(Tensor input, Tensor input_scales,
                                Tensor output, Tensor output_scales,
                                Tensor total_padded_tokens,
                                int64_t inner_dim) {
  checkInputs(input, input_scales, output, output_scales, total_padded_tokens);
  launchActivationV2(
      static_cast<cutlass::float_e4m3_t const*>(input.data_ptr()),
      static_cast<cutlass::float_e4m3_t*>(output.data_ptr()),
      static_cast<float*>(input_scales.data_ptr()),
      static_cast<float*>(output_scales.data_ptr()),
      static_cast<int32_t>(inner_dim),
      static_cast<int32_t const*>(total_padded_tokens.data_ptr()),
      static_cast<int32_t>(input.shape()[0]),
      /*gridY_override=*/-1,
      get_stream(input.device()));
}

// V2 with grid override
void activation_deepseek_fp8_v2_tuned(Tensor input, Tensor input_scales,
                                      Tensor output, Tensor output_scales,
                                      Tensor total_padded_tokens,
                                      int64_t inner_dim,
                                      int64_t grid_y_override) {
  checkInputs(input, input_scales, output, output_scales, total_padded_tokens);
  launchActivationV2(
      static_cast<cutlass::float_e4m3_t const*>(input.data_ptr()),
      static_cast<cutlass::float_e4m3_t*>(output.data_ptr()),
      static_cast<float*>(input_scales.data_ptr()),
      static_cast<float*>(output_scales.data_ptr()),
      static_cast<int32_t>(inner_dim),
      static_cast<int32_t const*>(total_padded_tokens.data_ptr()),
      static_cast<int32_t>(input.shape()[0]),
      static_cast<int32_t>(grid_y_override),
      get_stream(input.device()));
}

// V3 optimized (default grid)
void activation_deepseek_fp8_v3(Tensor input, Tensor input_scales,
                                Tensor output, Tensor output_scales,
                                Tensor total_padded_tokens,
                                int64_t inner_dim) {
  checkInputs(input, input_scales, output, output_scales, total_padded_tokens);
  launchActivationV3(
      static_cast<cutlass::float_e4m3_t const*>(input.data_ptr()),
      static_cast<cutlass::float_e4m3_t*>(output.data_ptr()),
      static_cast<float*>(input_scales.data_ptr()),
      static_cast<float*>(output_scales.data_ptr()),
      static_cast<int32_t>(inner_dim),
      static_cast<int32_t const*>(total_padded_tokens.data_ptr()),
      static_cast<int32_t>(input.shape()[0]),
      /*gridY_override=*/-1,
      get_stream(input.device()));
}

// V3 with grid override
void activation_deepseek_fp8_v3_tuned(Tensor input, Tensor input_scales,
                                      Tensor output, Tensor output_scales,
                                      Tensor total_padded_tokens,
                                      int64_t inner_dim,
                                      int64_t grid_y_override) {
  checkInputs(input, input_scales, output, output_scales, total_padded_tokens);
  launchActivationV3(
      static_cast<cutlass::float_e4m3_t const*>(input.data_ptr()),
      static_cast<cutlass::float_e4m3_t*>(output.data_ptr()),
      static_cast<float*>(input_scales.data_ptr()),
      static_cast<float*>(output_scales.data_ptr()),
      static_cast<int32_t>(inner_dim),
      static_cast<int32_t const*>(total_padded_tokens.data_ptr()),
      static_cast<int32_t>(input.shape()[0]),
      static_cast<int32_t>(grid_y_override),
      get_stream(input.device()));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(activation_deepseek_fp8_v2,
                              activation_deepseek_fp8_v2);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(activation_deepseek_fp8_v2_tuned,
                              activation_deepseek_fp8_v2_tuned);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(activation_deepseek_fp8_v3,
                              activation_deepseek_fp8_v3);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(activation_deepseek_fp8_v3_tuned,
                              activation_deepseek_fp8_v3_tuned);
