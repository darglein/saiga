/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/camera/RGBDCamera.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/time/timer.h"


namespace Saiga {


class SAIGA_GLOBAL TumRGBDCamera : public RGBDCamera
{
public:
    struct GroundTruth
    {
        double timeStamp;
        SE3 se3;
    };

    struct CameraData
    {
        double timestamp;
        std::string img;
    };

    struct TumFrame
    {
        GroundTruth gt;
        CameraData depth;
        CameraData rgb;
    };

    TumRGBDCamera(const std::string& datasetDir, double depthFactor = 1.0 / 5000, int maxFrames = -1, int fps = 30);
    ~TumRGBDCamera();
    /**
     * Blocks until a new image arrives.
     */
    virtual std::shared_ptr<FrameData> waitForImage() override;


    SE3 getGroundTruth(int frame);

    virtual bool isOpened() { return currentId < (int)frames.size(); }
private:
    void associate(const std::string& datasetDir);
    void load(const std::string& datasetDir);

    double depthFactor;
    int maxFrames;

    std::vector<std::shared_ptr<FrameData>> frames;
    std::vector<TumFrame> tumframes;

    Timer timer;
    tick_t timeStep;
    tick_t lastFrameTime;
    tick_t nextFrameTime;


};

}
