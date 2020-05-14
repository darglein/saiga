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
class SAIGA_VISION_API TumRGBDCamera : public DatasetCameraBase<RGBDFrameData>
{
   public:
    struct GroundTruth
    {
        double timestamp = -1;
        SE3 se3;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        bool operator<(const GroundTruth& other) const { return timestamp < other.timestamp; }
    };

    struct CameraData
    {
        double timestamp = -1;
        std::string img;

        bool operator<(const CameraData& other) const { return timestamp < other.timestamp; }
    };

    struct TumFrame
    {
        GroundTruth gt;
        CameraData depth;
        CameraData rgb;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    // If freiburg == -1 then the name is parsed from the dataset directory.
    // Otherwise it should be 1,2, or 3.
    TumRGBDCamera(const DatasetParameters& params, int freiburg = -1);
    ~TumRGBDCamera();


    SE3 getGroundTruth(int frame);

    RGBDIntrinsics intrinsics() { return _intrinsics; }


    void saveRaw(const std::string& dir);

    virtual void LoadImageData(RGBDFrameData& data) override;
    virtual int LoadMetaData() override;


   private:
    int freiburg;
    void associate(const std::string& datasetDir);
    void load(const std::string& datasetDir, bool multithreaded);


    AlignedVector<TumFrame> tumframes;


    RGBDIntrinsics _intrinsics;
};

}  // namespace Saiga
