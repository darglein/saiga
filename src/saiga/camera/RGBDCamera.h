/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include <chrono>
#include "saiga/image/image.h"

namespace Saiga {

using RGBImageType = TemplatedImage<ucvec4> ;
using DepthImageType = TemplatedImage<float>;


class SAIGA_GLOBAL RGBDCamera
{
public:

    struct FrameData
    {
          RGBImageType colorImg;
          DepthImageType depthImg;
          int frameId;
          std::chrono::steady_clock::time_point  captureTime;
    };

    struct CameraOptions
    {
        int w = 640;
        int h = 480;
        int fps = 30;
    };



    RGBDCamera(){}
    RGBDCamera(CameraOptions rgbo, CameraOptions deptho);


    virtual std::shared_ptr<FrameData> waitForImage() = 0;
    virtual std::shared_ptr<FrameData> tryGetImage() { return waitForImage(); }
    virtual mat4 getGroundTruth(int frame) { return mat4(1); }

    virtual bool isOpened() { return true; }
protected:
    CameraOptions rgbo, deptho;
    int currentId = 0;
    std::shared_ptr<FrameData> makeFrameData();
    void setNextFrame(FrameData& data);
};

}
