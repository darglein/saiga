/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/imageView.h"
#include "saiga/core/math/math.h"
#include "saiga/vision/opencv/OpenCV_GLM.h"

#include "opencv2/opencv.hpp"

#ifdef SAIGA_CUDA_INCLUDED
#    include <vector_types.h>
#endif

#if __has_include(<opencv2/core/cuda.hpp>)
#    define SAIGA_WITH_OPENCV_CUDA
#    include <opencv2/core/cuda.hpp>
#endif

// here are only some cv::Mat <-> ImageView conversions
// so saiga's cuda image processing can be used
// Note: these functions DON'T copy the actual image data

namespace Saiga
{
template <typename T>
inline ImageView<T> MatToImageView(cv::Mat& img)
{
    auto res = ImageView<T>(img.rows, img.cols, (size_t)img.step, img.data);
    SAIGA_ASSERT(res.size() == (size_t)img.step * img.rows);

    SAIGA_ASSERT(!((img.type() == CV_8UC1) ^ (typeid(T) == typeid(uchar))));

    return res;
}

template <typename T>
inline cv::Mat ImageViewToMat(ImageView<T> img)
{
    int type = -1;
#if defined(SAIGA_CUDA_INCLUDED)
    if (typeid(T) == typeid(uchar3)) type = CV_8UC3;
    if (typeid(T) == typeid(uchar4)) type = CV_8UC4;
#endif
    if (typeid(T) == typeid(float)) type = CV_32FC1;
    if (typeid(T) == typeid(ucvec3)) type = CV_8UC3;
    if (typeid(T) == typeid(ucvec4)) type = CV_8UC4;
    if (typeid(T) == typeid(uchar)) type = CV_8UC1;
    SAIGA_ASSERT(type != -1);
    return cv::Mat(img.height, img.width, type, img.data, img.pitchBytes);
}


#ifdef SAIGA_WITH_OPENCV_CUDA
template <typename T>
inline ImageView<T> MatToImageView(cv::cuda::GpuMat& img)
{
    auto res = ImageView<T>(img.rows, img.cols, (size_t)img.step, img.data);
    SAIGA_ASSERT(res.size() == (size_t)img.step * img.rows);
    return res;
}

template <typename T>
inline cv::cuda::GpuMat ImageViewToGpuMat(ImageView<T> img)
{
    int type = -1;
#    if defined(SAIGA_CUDA_INCLUDED)
    if (typeid(T) == typeid(uchar3)) type = CV_8UC3;
    if (typeid(T) == typeid(uchar4)) type = CV_8UC4;
#    endif
    if (typeid(T) == typeid(float)) type = CV_32FC1;
    if (typeid(T) == typeid(ucvec3)) type = CV_8UC3;
    if (typeid(T) == typeid(ucvec4)) type = CV_8UC4;
    if (typeid(T) == typeid(uchar)) type = CV_8UC1;
    SAIGA_ASSERT(type != -1);
    return cv::cuda::GpuMat(img.height, img.width, type, img.data, img.pitchBytes);
}
#endif

/**
 * Computes the scaled intrinsics matrix of K.
 * Useful for example when a downsampled version of the image is used.
 */
inline mat3 scaleK(mat3 K, float scale)
{
    K *= scale;
    K(2, 2) = 1;
    return K;
}

}  // namespace Saiga
