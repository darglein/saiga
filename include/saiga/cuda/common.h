/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

//remove all CUDA_SYNC_CHECK_ERROR and CUDA_ASSERTS
//for gcc add cppflag: -DCUDA_NDEBUG
#if !defined(CUDA_NDEBUG) 
#if !defined(CUDA_DEBUG)
#define CUDA_DEBUG
#endif
#else
#undef CUDA_DEBUG
#endif


#ifdef __CUDACC__
#	define HD __host__ __device__
#	define IS_CUDA
#	if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
#		define ON_DEVICE
#	endif
#else
#	define HD
#   if !defined(__launch_bounds__)
#       define __launch_bounds__
#   endif
#endif


// constants defined as functions, because cuda device code
// can access constexpr functions but not constexpr globals
//HD SAIGA_CONSTEXPR inline float PI() { return 3.1415926535897932f; }
//HD SAIGA_CONSTEXPR inline float TWOPI() { return 2 * PI(); }
//HD SAIGA_CONSTEXPR inline float INV_PI() { return 1.f / PI(); }
//HD SAIGA_CONSTEXPR inline float INV_TWOPI() { return 1.f / TWOPI(); }


#define WARP_SIZE 32

#define L1_CACHE_LINE_SIZE 128
#define L2_CACHE_LINE_SIZE 32
