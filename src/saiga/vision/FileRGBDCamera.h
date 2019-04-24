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
class SAIGA_VISION_API FileRGBDCamera : public RGBDCamera
{
   public:
    FileRGBDCamera(const std::string& datasetDir, const RGBDIntrinsics& intr);
    ~FileRGBDCamera();
    /**
     * Blocks until a new image arrives.
     */
    virtual std::shared_ptr<RGBDFrameData> getImageSync() override;


    virtual bool isOpened() override { return currentId < (int)frames.size(); }

    size_t getFrameCount() { return frames.size(); }

    void load(const std::string& datasetDir);

   private:
    std::vector<std::shared_ptr<RGBDFrameData>> frames;

    Timer timer;
    tick_t timeStep;
    tick_t lastFrameTime;
    tick_t nextFrameTime;
};

}  // namespace Saiga
