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

// The meta data is given in .yaml files so we need that dependency.
#ifdef SAIGA_USE_YAML_CPP

namespace Saiga
{
class SAIGA_VISION_API EuRoCDataset : public DatasetCameraBase<StereoFrameData>
{
   public:
    EuRoCDataset(const DatasetParameters& params);

    StereoIntrinsics intrinsics;

    virtual SE3 CameraToGroundTruth() override { return groundTruthToCamera.inverse(); }

   private:
    SE3 extrinsics_cam0, extrinsics_cam1, extrinsics_gt;
    SE3 groundTruthToCamera;


    // Tmp loading data
    std::vector<std::pair<double, std::string>> cam0_images, cam1_images;
    std::vector<std::pair<double, SE3>> ground_truth;
};

}  // namespace Saiga

#endif
