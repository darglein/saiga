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
class SAIGA_VISION_API KittiDataset : public DatasetCameraBase
{
   public:
    KittiDataset(const DatasetParameters& params);

    virtual int LoadMetaData() override;
    virtual void LoadImageData(FrameData& data) override;

    StereoIntrinsics intrinsics;
};

}  // namespace Saiga
