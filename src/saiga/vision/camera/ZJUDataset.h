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
#include "saiga/vision/camera/RGBDCamera.h"


// Loads datasets from the "ZJU - SenseTime VISLAM Benchmark"
// http://www.zjucvg.net/eval-vislam/
//
// The dataset includes a monocular grayscale camera, an IMU, and 6DOF ground truth.
// The images and IMU are unfiltered input from smart phones.
//
// Xiaomi Mi8
//     Camera: 640x480 at 30fps, rolling shutter
//     IMU: 400Hz
// iPhone X
//     Camera: 640x480 at 30fps, rolling shutter
//     IMU: 100Hz
namespace Saiga
{
class SAIGA_VISION_API ZJUDataset : public DatasetCameraBase<MonocularFrameData>
{
   public:
    struct IsmarFrame
    {
        std::string image;
        double timestamp;
        std::optional<SE3> gt;
    };

    ZJUDataset(const DatasetParameters& params, const RGBDIntrinsics& intr);

    virtual SE3 CameraToGroundTruth() override { return groundTruthToCamera.inverse(); }

    RGBDIntrinsics intrinsics;

   private:
    void associate(const std::string& datasetDir);
    void load(const std::string& datasetDir, bool multithreaded);

    SE3 groundTruthToCamera;

    AlignedVector<IsmarFrame> framesRaw;
};

}  // namespace Saiga
