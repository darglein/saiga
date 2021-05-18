/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#define SAIGA_ARRAY_VIEW_THRUST

#include "saiga/config.h"
#include "saiga/core/math/imath.h"
#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/cuda/cuda.h"
#include "saiga/cuda/cudaTimer.h"
#include "saiga/cuda/thrust_helper.h"



namespace Saiga
{
namespace CUDA
{
template <typename T1, typename T2>
HD SAIGA_CONSTEXPR T1 getBlockCount(T1 problemSize, T2 threadCount)
{
    return (problemSize + (threadCount - T2(1))) / (threadCount);
}
}  // namespace CUDA
}  // namespace Saiga


#define THREAD_BLOCK(_problemSize, _threadCount) Saiga::CUDA::getBlockCount(_problemSize, _threadCount), _threadCount
