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
//
// The sequence parameter is used to add time offsets to the ground truth data.
// If set to unknown, we try to extract the sequence from the file name.
//
class SAIGA_VISION_API EuRoCDataset : public DatasetCameraBase
{
   public:
    enum Sequence
    {
        MH_01 = 0,
        MH_02,
        MH_03,
        MH_04,
        MH_05,
        V1_01,
        V1_02,
        V1_03,
        V2_01,
        V2_02,
        V2_03,
        UNKNOWN
    };

    EuRoCDataset(const DatasetParameters& params, Sequence sequence = UNKNOWN);

    StereoIntrinsics intrinsics;


    virtual void LoadImageData(FrameData& data) override;
    virtual int LoadMetaData() override;


    static std::vector<std::string> DatasetNames()
    {
        return {
            "MH_01", "MH_02", "MH_03", "MH_04", "MH_05", "V1_01", "V1_02", "V1_03", "V2_01", "V2_02", "V2_03",
        };
    }

   private:
    void FindSequence();
    SE3 extrinsics_cam0, extrinsics_cam1, extrinsics_gt;

    bool use_raw_gt_data = false;
    Sequence sequence;
    // Tmp loading data
    std::vector<std::pair<double, std::string>> cam0_images, cam1_images;
};

}  // namespace Saiga

#endif
