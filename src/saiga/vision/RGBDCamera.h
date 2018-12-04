/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/image/image.h"
#include "saiga/vision/DepthmapPreprocessor.h"

#include <chrono>

namespace Saiga
{
using RGBImageType   = TemplatedImage<ucvec4>;
using DepthImageType = TemplatedImage<float>;


class SAIGA_GLOBAL RGBDCamera
{
   public:
    struct FrameData
    {
        RGBImageType colorImg;
        DepthImageType depthImg;
        int frameId;
        std::chrono::steady_clock::time_point captureTime;
    };

    struct CameraOptions
    {
        int w   = 640;
        int h   = 480;
        int fps = 30;
    };



    RGBDCamera() {}
    RGBDCamera(CameraOptions rgbo, CameraOptions deptho);


    virtual std::shared_ptr<FrameData> waitForImage() = 0;
    virtual std::shared_ptr<FrameData> tryGetImage() { return waitForImage(); }

    // Close the camera.
    // Blocking calls to waitForImage should return a 'nullptr'
    virtual void close() {}
    virtual bool isOpened() { return true; }


    void setDmpp(const std::shared_ptr<DMPP>& value);

   protected:
    std::shared_ptr<DMPP> dmpp;
    CameraOptions rgbo, deptho;
    int currentId = 0;
    std::shared_ptr<FrameData> makeFrameData();
    void setNextFrame(FrameData& data);
};

}  // namespace Saiga
