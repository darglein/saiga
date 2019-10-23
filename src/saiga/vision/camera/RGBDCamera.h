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

#include "CameraData.h"


namespace Saiga
{
class SAIGA_VISION_API RGBDCamera
{
   public:
    RGBDCamera() {}
    RGBDCamera(const RGBDIntrinsics& intr) : _intrinsics(intr) {}
    virtual ~RGBDCamera() {}

    // Blocks until the next image is available
    virtual bool getImageSync(RGBDFrameData& data) = 0;

    // Returns false if no image is currently available
    virtual bool getImage(RGBDFrameData& data) { return getImageSync(data); }



    // Close the camera.
    // Blocking calls to waitForImage should return a 'nullptr'
    virtual void close() {}
    virtual bool isOpened() { return true; }


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
