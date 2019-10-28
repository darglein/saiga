/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/time/timer.h"
#include "saiga/vision/VisionTypes.h"

#include "RGBDCamera.h"



namespace Saiga
{
class SAIGA_VISION_API KittiDataset : public DatasetCameraBase<StereoFrameData>
{
   public:
    KittiDataset(const DatasetParameters& params);

    StereoIntrinsics intrinsics;
};

}  // namespace Saiga
