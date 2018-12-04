/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/imageProcessing/image.h"

namespace Saiga
{
namespace CUDA
{
void resizeDeviceVector(thrust::device_vector<unsigned char>& v, int size)
{
    v.resize(size);
}

void copyDeviceVector(const thrust::device_vector<unsigned char>& src, thrust::device_vector<unsigned char>& dst)
{
    dst = src;
}

}  // namespace CUDA
}  // namespace Saiga
