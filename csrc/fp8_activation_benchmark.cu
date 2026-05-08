/*
 * Benchmark wrapper for activationDeepSeekKernelV2/V3/V4.
 *
 * V2: Original kernel (hoisted invariants) — 128 threads, 1 elt/thread,
 *     cub::BlockReduce, shared memory + __syncthreads.
 *
 * V3: Vectorized — 4 elts/thread, warp-level reduction, 8 warps/CTA (256 threads).
 *     One warp per scale block. No shared memory, no barriers.
 *
 * V4: Occupancy-tuned V3 — 4 warps/CTA (128 threads) instead of 8.
 *     2x more blocks → better SM utilization. Scale loads via lane-0 + shfl broadcast.
 *     ncu showed V3 at 0.61 waves / 52% occupancy; V4 targets 1.0+ waves / 80%+.
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
constexpr int ELTS_PER_THREAD = 4;
constexpr int THREADS_PER_SCALE_BLOCK = ELTS_PER_SCALE_BLOCK / ELTS_PER_THREAD;  // 32 = 1 warp

////////////////////////////////////////////////////////////////////////////////////////////////////

// 2 SFU ops: MUFU.EX2 + MUFU.RCP
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

// 1 SFU op: MUFU.TANH + 1 FFMA
__device__ __forceinline__ float silu_tanh(float x) {
  return x * (0.5f + 0.5f * __tanhf(x * 0.5f));
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
  return val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// V2/V3/V4/V5 params (padded layout, no indirection)
struct ActivationDeepSeekParams {
  cutlass::float_e4m3_t const* inPtr;
  cutlass::float_e4m3_t* outPtr;
  float* inDqSfsPtr;
  float* outDqSfsPtr;
  int32_t innerDim;
  int32_t const* totalNumPaddedTokens;
};

// V1 params (production kernel with expandedIdx indirection)
struct ActivationDeepSeekParamsV1 {
  cutlass::float_e4m3_t const* inPtr;
  cutlass::float_e4m3_t* outPtr;
  float* inDqSfsPtr;
  float* outDqSfsPtr;
  int32_t innerDim;
  int32_t numTokens;
  int32_t topK;
  int32_t* expandedIdxToPermutedIdx;
  int32_t const* totalNumPaddedTokens;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// V1: Production kernel (activationDeepSeekKernel) — uses expandedIdx indirection,
//     loops over numTokens * topK, no padding awareness.
//     Copy from csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu:202
//     Instantiated with NumTokensPerCta=1.
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(V2_THREADS_PER_CTA)
    activationDeepSeekKernelV1(ActivationDeepSeekParamsV1 params) {
  using BlockReduce = cub::BlockReduce<float, V2_THREADS_PER_CTA>;

  __shared__ float s_scaleOut;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  float constexpr E4m3MaxVal{448.f};
  int const totalNumPaddedTokens = params.totalNumPaddedTokens[0];

  using fp8_t = cutlass::float_e4m3_t;

  for (int k = blockIdx.z; k < params.topK; k += gridDim.z) {
    for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens;
         tokenIdx += gridDim.y) {
      for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x;
           hiddenIdx < params.innerDim / 2;
           hiddenIdx += blockDim.x * gridDim.x) {
        int const expandedIdx = tokenIdx * params.topK + k;
        int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
        if (permutedIdx == -1) continue;

        int64_t const baseIdx =
            (int64_t)permutedIdx * params.innerDim + hiddenIdx;
        int64_t const scale1Idx =
            (int64_t)permutedIdx +
            (int64_t)totalNumPaddedTokens * (hiddenIdx / 128);
        int64_t const scale2Idx =
            (int64_t)permutedIdx +
            (int64_t)totalNumPaddedTokens *
                ((hiddenIdx / 128) + (params.innerDim / 2 / 128));

        float scale1 = params.inDqSfsPtr[scale1Idx];
        float scale2 = params.inDqSfsPtr[scale2Idx];
        float x1 = scale1 * static_cast<float>(params.inPtr[baseIdx]);
        float x2 = scale2 * static_cast<float>(
                                 params.inPtr[baseIdx + params.innerDim / 2]);

        float out = silu_f(x2) * x1;
        float absOut = fabsf(out);

#if CUDA_VERSION >= 12090
        float aMax =
            BlockReduce(tempStorage).Reduce(absOut, cuda::maximum<>{});
#else
        float aMax = BlockReduce(tempStorage).Reduce(absOut, cub::Max{});
#endif

        if (threadIdx.x == 0) {
          float scaleOut =
              fmaxf(aMax / E4m3MaxVal, FLT_MIN);
          s_scaleOut = scaleOut;
          int64_t const scaleOutIdx =
              (int64_t)permutedIdx +
              (int64_t)totalNumPaddedTokens * (hiddenIdx / 128);
          params.outDqSfsPtr[scaleOutIdx] = scaleOut;
        }
        __syncthreads();

        int64_t const outIdx =
            (int64_t)permutedIdx * (params.innerDim / 2) + hiddenIdx;
        params.outPtr[outIdx] = static_cast<fp8_t>(out / s_scaleOut);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// V2: Original kernel with hoisted loop invariants (padded layout)
////////////////////////////////////////////////////////////////////////////////////////////////////

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
// Vectorized kernel template — parameterized by warps-per-CTA
//
// Thread mapping:
//   warpId = threadIdx.x / 32   -> selects scale block within CTA
//   laneId = threadIdx.x % 32   -> selects 4-element group within scale block
//   scaleBlock = blockIdx.x * WarpsPerCta + warpId
//
// Each warp independently handles one 128-element scale block.
// Scale loads via lane-0 + __shfl broadcast (avoids uncoalesced sector waste).
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int WarpsPerCta, bool UseTanhSilu = false>
__global__ void __launch_bounds__(WarpsPerCta * 32)
    activationDeepSeekKernelVec(ActivationDeepSeekParams params) {
  float constexpr E4m3MaxVal{448.f};
  int const totalPadded = params.totalNumPaddedTokens[0];
  int const sfStride = totalPadded;
  int const halfDim = params.innerDim / 2;
  int const numOutputScaleBlocks = halfDim / ELTS_PER_SCALE_BLOCK;

  int const warpId = threadIdx.x / 32;
  int const laneId = threadIdx.x % 32;
  int const scaleBlock = blockIdx.x * WarpsPerCta + warpId;

  if (scaleBlock >= numOutputScaleBlocks) return;

  int const elemBase =
      scaleBlock * ELTS_PER_SCALE_BLOCK + laneId * ELTS_PER_THREAD;

  // Loop-invariant scale bases
  int64_t const scale1Base = (int64_t)sfStride * scaleBlock;
  int64_t const scale2Base =
      (int64_t)sfStride * (scaleBlock + numOutputScaleBlocks);

  using fp8_t = cutlass::float_e4m3_t;

  for (int permutedRow = blockIdx.y; permutedRow < totalPadded;
       permutedRow += gridDim.y) {
    // Lane-0 loads scales, broadcast via shfl (avoids 32 redundant sector loads)
    float scale1, scale2;
    if (laneId == 0) {
      scale1 = params.inDqSfsPtr[permutedRow + scale1Base];
      scale2 = params.inDqSfsPtr[permutedRow + scale2Base];
    }
    scale1 = __shfl_sync(0xffffffff, scale1, 0);
    scale2 = __shfl_sync(0xffffffff, scale2, 0);

    // Vectorized load: 4 fp8 elements as uint32
    int64_t const x1Offset =
        (int64_t)permutedRow * params.innerDim + elemBase;

    uint32_t packed_x1 =
        *reinterpret_cast<uint32_t const*>(&params.inPtr[x1Offset]);
    uint32_t packed_x2 =
        *reinterpret_cast<uint32_t const*>(&params.inPtr[x1Offset + halfDim]);

    // Unpack, dequantize, silu+mul, track max
    fp8_t x1_vals[4], x2_vals[4];
    memcpy(x1_vals, &packed_x1, 4);
    memcpy(x2_vals, &packed_x2, 4);

    float localMax = 0.0f;
    float results[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float f1 = scale1 * static_cast<float>(x1_vals[i]);
      float f2 = scale2 * static_cast<float>(x2_vals[i]);
      if constexpr (UseTanhSilu) {
        results[i] = silu_tanh(f2) * f1;
      } else {
        results[i] = silu_f(f2) * f1;
      }
      localMax = fmaxf(localMax, fabsf(results[i]));
    }

    // Warp-level max reduction
    float aMax = warp_reduce_max(localMax);

    // Lane 0 computes + stores output scale, broadcast to all lanes
    float scaleOut;
    if (laneId == 0) {
      scaleOut = fmaxf(aMax / E4m3MaxVal, FLT_MIN);
      params.outDqSfsPtr[permutedRow + scale1Base] = scaleOut;
    }
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

    int64_t const outOffset = (int64_t)permutedRow * halfDim + elemBase;
    *reinterpret_cast<uint32_t*>(&params.outPtr[outOffset]) = packed_out;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Launchers
////////////////////////////////////////////////////////////////////////////////////////////////////

static void launchActivationV1(
    ActivationDeepSeekParamsV1& params, cudaStream_t stream) {
  int const numScaleBlocks =
      (params.innerDim / 2 + ELTS_PER_SCALE_BLOCK - 1) / ELTS_PER_SCALE_BLOCK;

  int device{-1};
  cudaGetDevice(&device);
  int numSms = 0;
  cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);

  // Production grid heuristic from run()
  auto numCtas = numScaleBlocks * params.numTokens * params.topK;
  int numTokensPerCta = 1;
  if (numCtas > numSms * 32) {
    numTokensPerCta = 4;
  } else if (numCtas > numSms * 4) {
    numTokensPerCta = 2;
  }
  int const gridSizeY = min(8192,
      (params.numTokens + numTokensPerCta - 1) / numTokensPerCta);

  dim3 grid(numScaleBlocks, gridSizeY, params.topK);
  activationDeepSeekKernelV1<<<grid, V2_THREADS_PER_CTA, 0, stream>>>(params);
}

static void launchActivationV2(
    ActivationDeepSeekParams& params, int32_t maxPermutedPaddedCount,
    int32_t gridY_override, cudaStream_t stream) {
  int const numScaleBlocks =
      (params.innerDim / 2 + ELTS_PER_SCALE_BLOCK - 1) / ELTS_PER_SCALE_BLOCK;

  int gridSizeY = gridY_override > 0
                      ? gridY_override
                      : min(8192, max(1, maxPermutedPaddedCount));

  dim3 grid(numScaleBlocks, gridSizeY, 1);
  activationDeepSeekKernelV2<<<grid, V2_THREADS_PER_CTA, 0, stream>>>(params);
}

template <int WarpsPerCta, bool UseTanhSilu = false>
static void launchActivationVec(
    ActivationDeepSeekParams& params, int32_t maxPermutedPaddedCount,
    int32_t gridY_override, cudaStream_t stream) {
  int const numScaleBlocks =
      (params.innerDim / 2 + ELTS_PER_SCALE_BLOCK - 1) / ELTS_PER_SCALE_BLOCK;
  int const gridSizeX =
      (numScaleBlocks + WarpsPerCta - 1) / WarpsPerCta;

  int device{-1};
  cudaGetDevice(&device);
  int numSms = 0;
  cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);

  int gridSizeY = gridY_override > 0
                      ? gridY_override
                      : min(numSms, max(1, maxPermutedPaddedCount));

  dim3 grid(gridSizeX, gridSizeY, 1);
  activationDeepSeekKernelVec<WarpsPerCta, UseTanhSilu>
      <<<grid, WarpsPerCta * 32, 0, stream>>>(params);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// TVM-FFI wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////

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

static ActivationDeepSeekParams makeParams(
    Tensor input, Tensor input_scales, Tensor output,
    Tensor output_scales, Tensor total_padded_tokens, int64_t inner_dim) {
  ActivationDeepSeekParams p;
  p.inPtr = static_cast<cutlass::float_e4m3_t const*>(input.data_ptr());
  p.outPtr = static_cast<cutlass::float_e4m3_t*>(output.data_ptr());
  p.inDqSfsPtr = static_cast<float*>(input_scales.data_ptr());
  p.outDqSfsPtr = static_cast<float*>(output_scales.data_ptr());
  p.innerDim = static_cast<int32_t>(inner_dim);
  p.totalNumPaddedTokens =
      static_cast<int32_t const*>(total_padded_tokens.data_ptr());
  return p;
}

// V1: production kernel with expandedIdx indirection
// extra args: expanded_idx_to_permuted_idx (int32 [numTokens * topK]),
//             num_tokens (int64), top_k (int64)
void activation_deepseek_fp8_v1(Tensor input, Tensor input_scales,
                                Tensor output, Tensor output_scales,
                                Tensor total_padded_tokens,
                                Tensor expanded_idx_to_permuted_idx,
                                int64_t inner_dim,
                                int64_t num_tokens,
                                int64_t top_k) {
  checkInputs(input, input_scales, output, output_scales, total_padded_tokens);
  CHECK_CUDA(expanded_idx_to_permuted_idx);
  CHECK_CONTIGUOUS(expanded_idx_to_permuted_idx);
  CHECK_INPUT_TYPE(expanded_idx_to_permuted_idx, dl_int32);

  ActivationDeepSeekParamsV1 p;
  p.inPtr = static_cast<cutlass::float_e4m3_t const*>(input.data_ptr());
  p.outPtr = static_cast<cutlass::float_e4m3_t*>(output.data_ptr());
  p.inDqSfsPtr = static_cast<float*>(input_scales.data_ptr());
  p.outDqSfsPtr = static_cast<float*>(output_scales.data_ptr());
  p.innerDim = static_cast<int32_t>(inner_dim);
  p.numTokens = static_cast<int32_t>(num_tokens);
  p.topK = static_cast<int32_t>(top_k);
  p.expandedIdxToPermutedIdx =
      static_cast<int32_t*>(expanded_idx_to_permuted_idx.data_ptr());
  p.totalNumPaddedTokens =
      static_cast<int32_t const*>(total_padded_tokens.data_ptr());
  launchActivationV1(p, get_stream(input.device()));
}

// V2 baseline
void activation_deepseek_fp8_v2(Tensor input, Tensor input_scales,
                                Tensor output, Tensor output_scales,
                                Tensor total_padded_tokens,
                                int64_t inner_dim) {
  checkInputs(input, input_scales, output, output_scales, total_padded_tokens);
  auto params = makeParams(input, input_scales, output, output_scales,
                            total_padded_tokens, inner_dim);
  launchActivationV2(params, static_cast<int32_t>(input.shape()[0]),
                     -1, get_stream(input.device()));
}

void activation_deepseek_fp8_v2_tuned(Tensor input, Tensor input_scales,
                                      Tensor output, Tensor output_scales,
                                      Tensor total_padded_tokens,
                                      int64_t inner_dim,
                                      int64_t grid_y_override) {
  checkInputs(input, input_scales, output, output_scales, total_padded_tokens);
  auto params = makeParams(input, input_scales, output, output_scales,
                            total_padded_tokens, inner_dim);
  launchActivationV2(params, static_cast<int32_t>(input.shape()[0]),
                     static_cast<int32_t>(grid_y_override),
                     get_stream(input.device()));
}

// V3: 8 warps/CTA (256 threads)
void activation_deepseek_fp8_v3(Tensor input, Tensor input_scales,
                                Tensor output, Tensor output_scales,
                                Tensor total_padded_tokens,
                                int64_t inner_dim) {
  checkInputs(input, input_scales, output, output_scales, total_padded_tokens);
  auto params = makeParams(input, input_scales, output, output_scales,
                            total_padded_tokens, inner_dim);
  launchActivationVec<8>(params, static_cast<int32_t>(input.shape()[0]),
                         -1, get_stream(input.device()));
}

void activation_deepseek_fp8_v3_tuned(Tensor input, Tensor input_scales,
                                      Tensor output, Tensor output_scales,
                                      Tensor total_padded_tokens,
                                      int64_t inner_dim,
                                      int64_t grid_y_override) {
  checkInputs(input, input_scales, output, output_scales, total_padded_tokens);
  auto params = makeParams(input, input_scales, output, output_scales,
                            total_padded_tokens, inner_dim);
  launchActivationVec<8>(params, static_cast<int32_t>(input.shape()[0]),
                         static_cast<int32_t>(grid_y_override),
                         get_stream(input.device()));
}

// V4: 4 warps/CTA (128 threads) — 2x more blocks for better occupancy
void activation_deepseek_fp8_v4(Tensor input, Tensor input_scales,
                                Tensor output, Tensor output_scales,
                                Tensor total_padded_tokens,
                                int64_t inner_dim) {
  checkInputs(input, input_scales, output, output_scales, total_padded_tokens);
  auto params = makeParams(input, input_scales, output, output_scales,
                            total_padded_tokens, inner_dim);
  launchActivationVec<4>(params, static_cast<int32_t>(input.shape()[0]),
                         -1, get_stream(input.device()));
}

void activation_deepseek_fp8_v4_tuned(Tensor input, Tensor input_scales,
                                      Tensor output, Tensor output_scales,
                                      Tensor total_padded_tokens,
                                      int64_t inner_dim,
                                      int64_t grid_y_override) {
  checkInputs(input, input_scales, output, output_scales, total_padded_tokens);
  auto params = makeParams(input, input_scales, output, output_scales,
                            total_padded_tokens, inner_dim);
  launchActivationVec<4>(params, static_cast<int32_t>(input.shape()[0]),
                         static_cast<int32_t>(grid_y_override),
                         get_stream(input.device()));
}

// V5: tanh-based silu (1 SFU op instead of 2), 4 warps/CTA
void activation_deepseek_fp8_v5(Tensor input, Tensor input_scales,
                                Tensor output, Tensor output_scales,
                                Tensor total_padded_tokens,
                                int64_t inner_dim) {
  checkInputs(input, input_scales, output, output_scales, total_padded_tokens);
  auto params = makeParams(input, input_scales, output, output_scales,
                            total_padded_tokens, inner_dim);
  launchActivationVec<4, true>(params, static_cast<int32_t>(input.shape()[0]),
                               -1, get_stream(input.device()));
}

void activation_deepseek_fp8_v5_tuned(Tensor input, Tensor input_scales,
                                      Tensor output, Tensor output_scales,
                                      Tensor total_padded_tokens,
                                      int64_t inner_dim,
                                      int64_t grid_y_override) {
  checkInputs(input, input_scales, output, output_scales, total_padded_tokens);
  auto params = makeParams(input, input_scales, output, output_scales,
                            total_padded_tokens, inner_dim);
  launchActivationVec<4, true>(params, static_cast<int32_t>(input.shape()[0]),
                               static_cast<int32_t>(grid_y_override),
                               get_stream(input.device()));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(activation_deepseek_fp8_v1,
                              activation_deepseek_fp8_v1);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(activation_deepseek_fp8_v2,
                              activation_deepseek_fp8_v2);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(activation_deepseek_fp8_v2_tuned,
                              activation_deepseek_fp8_v2_tuned);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(activation_deepseek_fp8_v3,
                              activation_deepseek_fp8_v3);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(activation_deepseek_fp8_v3_tuned,
                              activation_deepseek_fp8_v3_tuned);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(activation_deepseek_fp8_v4,
                              activation_deepseek_fp8_v4);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(activation_deepseek_fp8_v4_tuned,
                              activation_deepseek_fp8_v4_tuned);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(activation_deepseek_fp8_v5,
                              activation_deepseek_fp8_v5);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(activation_deepseek_fp8_v5_tuned,
                              activation_deepseek_fp8_v5_tuned);
