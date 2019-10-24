/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/image/image.h"
#include "saiga/vision/cameraModel/Distortion.h"
#include "saiga/vision/util/DepthmapPreprocessor.h"

#include "CameraBase.h"
#include "CameraData.h"

namespace Saiga
{
class SAIGA_VISION_API RGBDCamera : public CameraBase<RGBDFrameData>
{
   public:
    RGBDCamera() {}
    RGBDCamera(const RGBDIntrinsics& intr) : _intrinsics(intr) {}
    virtual ~RGBDCamera() {}



    const RGBDIntrinsics& intrinsics() { return _intrinsics; }

   protected:
    RGBDIntrinsics _intrinsics;
    int currentId = 0;

    // Create a frame data object with the images already allocated in the correct size
    void makeFrameData(RGBDFrameData& data);

    // Set frame id and capture time
    void setNextFrame(RGBDFrameData& data);
};

}  // namespace Saiga
