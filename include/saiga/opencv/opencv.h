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

namespace Saiga {


template<typename T>
ImageView<T> MatToImageView(cv::Mat& img){
    auto res = ImageView<T>(img.cols,img.rows,img.step,img.data);
    SAIGA_ASSERT(res.size() == img.size);
    return res;
}

template<typename T>
cv::Mat ImageViewToMat(ImageView<T> img){
    int type;
    switch(typeid(T)){
#if defined(SAIGA_USE_CUDA)
    case typeid(uchar3):
        type = CV_8UC3;
        break;
    case typeid(uchar4):
        type = CV_8UC4;
        break;
#endif
    case typeid(float):
        type = CV_32FC1;
        break;
    default:
        SAIGA_ASSERT(0,"Unknown Type");
    }
    return cv::Mat(img.height,img.width,type,img.pitchBytes,img.data);
}




}
