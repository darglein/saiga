/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/time/timer.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/camera/CameraBase.h"


// The meta data is given in .yaml files so we need that dependency.
#ifdef SAIGA_USE_YAML_CPP

namespace Saiga
{
// EuRoC Stereo MAV dataset (flying drones)
// The dataset directory in the params should be the path to mav0/.
// For example:
//   DatasetParameters params;
//   params.dir = "/ssd2/slam/euroc/V1_03/mav0";
//   auto c = std::make_unique<Saiga::EuRoCDataset>(params);
class SAIGA_VISION_API EuRoCDataset : public DatasetCameraBase<StereoFrameData>
{
   public:
    EuRoCDataset(const DatasetParameters& params);

    StereoIntrinsics intrinsics;


    virtual void LoadImageData(StereoFrameData& data) override;
    virtual int LoadMetaData() override;


    static std::vector<std::string> DatasetNames()
    {
        return {"V1_01", "V1_02", "V1_03", "V2_01", "V2_02", "V2_03", "MH_01", "MH_02", "MH_03", "MH_04", "MH_05"};
    }

   private:
    SE3 extrinsics_cam0, extrinsics_cam1, extrinsics_gt;
    SE3 groundTruthToCamera;

    // Tmp loading data
    std::vector<std::pair<double, std::string>> cam0_images, cam1_images;
};

}  // namespace Saiga

#endif
