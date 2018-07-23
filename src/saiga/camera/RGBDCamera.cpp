/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "RGBDCamera.h"

namespace Saiga {

std::shared_ptr<RGBDCamera::FrameData> RGBDCamera::makeFrameData()
{
    auto fd = std::make_shared<RGBDCamera::FrameData>();
    fd->colorImg.create(colorH,colorW);
    fd->depthImg.create(depthH,depthW);
    return fd;
}

void RGBDCamera::setNextFrame(RGBDCamera::FrameData &data)
{
    data.frameId = currentId++;
    data.captureTime = std::chrono::steady_clock::now();
}

}
