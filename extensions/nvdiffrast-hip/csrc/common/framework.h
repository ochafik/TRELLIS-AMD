// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

// Framework-specific macros to enable code sharing.

//------------------------------------------------------------------------
// PyTorch.

#ifdef NVDR_TORCH
#include <torch/extension.h>

// HIP/CUDA context headers - only include when compiling with device compiler
// because they include raw HIP/CUDA headers that MSVC can't handle
#if defined(__HIP_PLATFORM_AMD__) && defined(__HIPCC__)
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPStream.h>
#elif defined(__CUDACC__)
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#if !defined(__CUDACC__) && !defined(__HIPCC__)
#include <pybind11/numpy.h>
#endif

#define NVDR_CHECK(COND, ERR)                                                  \
  do {                                                                         \
    TORCH_CHECK(COND, ERR)                                                     \
  } while (0)

// NVDR_CHECK_CUDA_ERROR requires CUDA/HIP types, only define when compiling with device compiler
#if defined(__HIP_PLATFORM_AMD__) && defined(__HIPCC__)
#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL)                                       \
  do {                                                                         \
    hipError_t err = CUDA_CALL;                                                \
    TORCH_CHECK(!err, "Cuda error: ", hipGetLastError(), "[", #CUDA_CALL,      \
                ";]");                                                         \
  } while (0)
#elif defined(__CUDACC__)
#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL)                                       \
  do {                                                                         \
    cudaError_t err = CUDA_CALL;                                               \
    TORCH_CHECK(!err, "Cuda error: ", cudaGetLastError(), "[", #CUDA_CALL,     \
                ";]");                                                         \
  } while (0)
#endif

#endif // NVDR_TORCH

//------------------------------------------------------------------------
