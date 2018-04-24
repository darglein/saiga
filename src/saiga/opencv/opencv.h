/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "opencv2/opencv.hpp"
#include "saiga/cuda/imageProcessing/imageView.h"

#if defined(SAIGA_USE_CUDA)
#include <vector_types.h>
#endif

//here are only some cv::Mat <-> ImageView conversions
//so saiga's cuda image processing can be used
//Note: these functions DON'T copy the actual image data

namespace Saiga {


template<typename T>
inline
ImageView<T> MatToImageView(cv::Mat& img){
    auto res = ImageView<T>(img.rows,img.cols,img.step,img.data);
    SAIGA_ASSERT(res.size() == img.step[0] * img.rows);
    return res;
}

template<typename T>
inline
cv::Mat ImageViewToMat(ImageView<T> img){
    int type;
#if defined(SAIGA_USE_CUDA)
    if(typeid(T) == typeid(uchar3))type = CV_8UC3;
    if(typeid(T) == typeid(uchar4))type = CV_8UC4;
#endif
    if(typeid(T) == typeid(float))type = CV_32FC1;
    return cv::Mat(img.height,img.width,type);
}




}
