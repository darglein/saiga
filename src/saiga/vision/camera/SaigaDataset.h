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
class SAIGA_VISION_API SaigaDataset : public DatasetCameraBase
{
   public:
    // If freiburg == -1 then the name is parsed from the dataset directory.
    // Otherwise it should be 1,2, or 3.
    SaigaDataset(const DatasetParameters& params, bool scale_down_depth = false);
    ~SaigaDataset();



    RGBDIntrinsics intrinsics() { return _intrinsics; }


    virtual void LoadImageData(FrameData& data) override;
    virtual int LoadMetaData() override;


   private:
    RGBDIntrinsics _intrinsics;
    std::vector<std::string> frame_dirs;
    bool scale_down_depth;
};

}  // namespace Saiga
