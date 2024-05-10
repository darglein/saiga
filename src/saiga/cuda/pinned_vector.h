/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include <thrust/host_vector.h>
// #include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/system/cuda/memory_resource.h>

namespace Saiga
{
/**
 * A host vector with pinnend memory (page locked). This allows faster host-device memory transfers.
 * The usage is identical to thrust::host_vector:
 *
 * Saiga::pinned_vector<T> h_data(N);
 * thrust::device_vector<T> d_data = h_data;
 */

using mr = thrust::system::cuda::universal_host_pinned_memory_resource;
template <typename T>
using pinned_allocator = thrust::mr::stateless_resource_allocator<T, mr >;

template <typename T>
using pinned_vector = ::thrust::host_vector<T, pinned_allocator<T> >;


}  // namespace Saiga
