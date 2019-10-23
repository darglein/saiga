/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "RGBDCamera.h"

#include "saiga/core/util/Thread/threadName.h"

namespace Saiga
{
void RGBDCamera::makeFrameData(RGBDFrameData& data)
{
    data.colorImg.create(intrinsics().imageSize.h, intrinsics().imageSize.w);
    data.depthImg.create(intrinsics().depthImageSize.h, intrinsics().depthImageSize.w);
}

void RGBDCamera::setNextFrame(RGBDFrameData& data)
{
    data.id = currentId++;
}



}  // namespace Saiga
