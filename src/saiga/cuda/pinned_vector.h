/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/thrust_helper.h"
#include <thrust/system/cuda/experimental/pinned_allocator.h>

namespace Saiga {
namespace thrust{

template<typename T>
using pinned_vector=::thrust::host_vector<T, ::thrust::cuda::experimental::pinned_allocator<T> >;


}
}
