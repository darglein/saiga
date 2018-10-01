/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/camera/RGBDCamera.h"
namespace Saiga {


class SAIGA_GLOBAL TumRGBDCamera : public RGBDCamera
{
public:
    struct GroundTruth
    {
        double timeStamp;
        vec3 position;
        quat rotation;
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

    TumRGBDCamera(const std::string& datasetDir, double depthFactor = 1.0 / 5000, int maxFrames = -1);

    /**
     * Blocks until a new image arrives.
     */
    virtual std::shared_ptr<FrameData> waitForImage() override;


    mat4 getGroundTruth(int frame);
private:
    void associate(const std::string& datasetDir);
    void load(const std::string& datasetDir);

    double depthFactor;
    int maxFrames;

    std::vector<std::shared_ptr<FrameData>> frames;
    std::vector<TumFrame> tumframes;
};

}
