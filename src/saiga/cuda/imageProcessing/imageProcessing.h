/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/imageProcessing/image.h"
#include "saiga/core/image/imageView.h"
#include "saiga/core/util/DataStructures/ArrayView.h"

namespace Saiga
{
namespace CUDA
{
// dst = src1 - src2
SAIGA_CUDA_API void subtract(ImageView<float> src1, ImageView<float> src2, ImageView<float> dst);

// subtract multiple images at the same time
// src.n - 1 == dst.n
// dst[0] = src[0] - src[1]
// dst[1] = src[1] - src[2]
//...
SAIGA_CUDA_API void subtractMulti(ImageArrayView<float> src, ImageArrayView<float> dst);


SAIGA_CUDA_API void fill(ImageView<float> img, float value);
SAIGA_CUDA_API void mult(ImageView<float> img, float value);
SAIGA_CUDA_API void add(ImageView<float> img, float value);
SAIGA_CUDA_API void abs(ImageView<float> img);


//==================== Image Scaling ======================

template <typename T>
SAIGA_CUDA_API void scaleDown2EveryOther(ImageView<T> src, ImageView<T> dst);

SAIGA_CUDA_API void scaleUp2Linear(ImageView<float> src, ImageView<float> dst);

//==================== Image Format Conversions ======================

SAIGA_CUDA_API void convertRGBtoRGBA(ImageView<uchar3> src, ImageView<uchar4> dst, unsigned char alpha = 255);
SAIGA_CUDA_API void convertRGBAtoRGB(ImageView<uchar4> src, ImageView<uchar3> dst);
SAIGA_CUDA_API void convertRGBAtoGrayscale(ImageView<uchar4> src, ImageView<float> dst);
SAIGA_CUDA_API void convertBGRtoRGBA(ImageView<uchar3> src, ImageView<uchar4> dst, unsigned char alpha = 255);
SAIGA_CUDA_API void convertRGBAtoBGR(ImageView<uchar4> src, ImageView<uchar3> dst);

//==================== Convolution ======================

#define SAIGA_MAX_CONVOLUTION_RADIUS 24
#define SAIGA_MAX_KERNEL_SIZE (SAIGA_MAX_CONVOLUTION_RADIUS * 2 + 1)

SAIGA_CUDA_API void convolveSinglePassSeparateOuterLinear(ImageView<float> src, ImageView<float> dst,
                                                        Saiga::ArrayView<float> kernel, int radius);
SAIGA_CUDA_API void convolveSinglePassSeparateOuterHalo(ImageView<float> src, ImageView<float> dst,
                                                      Saiga::ArrayView<float> kernel, int radius);
SAIGA_CUDA_API void convolveSinglePassSeparateInner(ImageView<float> src, ImageView<float> dst,
                                                  Saiga::ArrayView<float> kernel, int radius);
SAIGA_CUDA_API void convolveSinglePassSeparateInner75(ImageView<float> src, ImageView<float> dst,
                                                    Saiga::ArrayView<float> kernel, int radius);
SAIGA_CUDA_API void convolveSinglePassSeparateInnerShuffle(ImageView<float> src, ImageView<float> dst,
                                                         Saiga::ArrayView<float> kernel, int radius);

SAIGA_CUDA_API void convolveRow(ImageView<float> src, ImageView<float> dst, Saiga::ArrayView<float> kernel, int radius);
SAIGA_CUDA_API void convolveCol(ImageView<float> src, ImageView<float> dst, Saiga::ArrayView<float> kernel, int radius);

SAIGA_CUDA_API thrust::device_vector<float> createGaussianBlurKernel(int radius, float sigma);

// uploads kernel and convoles images
SAIGA_CUDA_API void applyFilterSeparate(ImageView<float> src, ImageView<float> dst, ImageView<float> tmp,
                                      ArrayView<float> kernelRow, ArrayView<float> kernelCol);
SAIGA_CUDA_API void applyFilterSeparateSinglePass(ImageView<float> src, ImageView<float> dst, ArrayView<float> kernel);


}  // namespace CUDA
}  // namespace Saiga
