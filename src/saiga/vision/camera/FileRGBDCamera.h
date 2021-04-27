/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/time/timer.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/camera/CameraBase.h"


namespace Saiga
{
class SAIGA_VISION_API FileRGBDCamera : public DatasetCameraBase
{
   public:
    FileRGBDCamera(const DatasetParameters& params, const RGBDIntrinsics& intr);
    ~FileRGBDCamera();

    RGBDIntrinsics intrinsics() { return _intrinsics; }

   private:
    void preload(const std::string& datasetDir, bool multithreaded);

    RGBDIntrinsics _intrinsics;
};

}  // namespace Saiga
