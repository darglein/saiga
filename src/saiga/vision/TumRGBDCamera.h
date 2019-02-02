/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/time/timer.h"
#include "saiga/vision/RGBDCamera.h"
#include "saiga/vision/VisionTypes.h"


namespace Saiga
{
class SAIGA_VISION_API TumRGBDCamera : public RGBDCamera
{
   public:
    struct GroundTruth
    {
        double timeStamp = -1;
        SE3 se3;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    struct CameraData
    {
        double timestamp = -1;
        std::string img;
    };

    struct TumFrame
    {
        GroundTruth gt;
        CameraData depth;
        CameraData rgb;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    TumRGBDCamera(const std::string& datasetDir, double depthFactor = 1.0 / 5000, int maxFrames = -1, int fps = 30,
                  const std::shared_ptr<DMPP>& dmpp = nullptr);
    ~TumRGBDCamera();
    /**
     * Blocks until a new image arrives.
     */
    virtual std::shared_ptr<FrameData> waitForImage() override;


    SE3 getGroundTruth(int frame);

    virtual bool isOpened() override { return currentId < (int)frames.size(); }

    size_t getFrameCount() { return frames.size(); }

   private:
    void associate(const std::string& datasetDir);
    void associateFromFile(const std::string& datasetDir);
    void load(const std::string& datasetDir);

    double depthFactor;
    int maxFrames;

    std::vector<std::shared_ptr<FrameData>> frames;
    AlignedVector<TumFrame> tumframes;

    Timer timer;
    tick_t timeStep;
    tick_t lastFrameTime;
    tick_t nextFrameTime;
};

}  // namespace Saiga
