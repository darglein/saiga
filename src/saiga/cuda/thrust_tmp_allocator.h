/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/cuda.h"

#include <iostream>

#include "thrust_helper.h"
#include <thrust/device_vector.h>
#include <thrust/mr/memory_resource.h>

namespace Saiga
{
// Some thrust functions, for example "sort" and "exclusive_scan" use temorary memory for their computations.
// This object can be passed as first argument so that memory is allocated only once.
//
// Example:
//   memory_resource thrust_tmp_memory;
//   thrust::exclusive_scan(
//        thrust::cuda::par(thrust_tmp_memory), d_cells_counts.begin(), d_cells_counts.end(),
//                           d_cells.begin(), 0);
class SAIGA_CUDA_API memory_resource : public thrust::mr::memory_resource<>
{
   public:
    typedef char value_type;
    typedef void* pointer;
    memory_resource();
    void* do_allocate(std::size_t num_bytes, std::size_t alignment);
    void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment);

   private:
    thrust::device_vector<char> mem;
    bool locked = false;
};

}  // namespace Saiga
