#pragma once

#include <thrust/version.h>

#if THRUST_VERSION < 100700
#error "Thrust v1.7.0 or newer is required"
#endif

#include <thrust/detail/config.h>
#include <thrust/detail/config/host_device.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/detail/detail/launch_calculator.h>


namespace CUDA {
/**
 * Use this function to compute the optimal number of blocks to start for a given kernel and block size.
 * With that trick you can start exactly as many threads as needed for 100% occupancy and no more.
 *
 * Example Usage:
 * const int blockSize = 256;
 * auto numBlocks = CUDA::max_active_blocks(deviceReduceBlockAtomicKernel,blockSize,0);
 * deviceReduceBlockAtomicKernel<<<numBlocks,blockSize>>>(v,res);
 */
template <typename KernelFunction>
size_t max_active_blocks(KernelFunction kernel, const size_t CTA_SIZE, const size_t dynamic_smem_bytes = 0)
{
  using namespace thrust::system::cuda::detail;
  function_attributes_t attributes = function_attributes(kernel);
  device_properties_t properties = device_properties();
  return properties.multiProcessorCount * cuda_launch_config_detail::max_active_blocks_per_multiprocessor(properties, attributes, CTA_SIZE, dynamic_smem_bytes);
}

}
