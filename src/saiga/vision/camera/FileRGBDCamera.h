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
class SAIGA_VISION_API FileRGBDCamera : public RGBDCamera
{
   public:
    FileRGBDCamera(const std::string& datasetDir, const RGBDIntrinsics& intr, bool preload = true,
                   bool multithreaded = true);
    ~FileRGBDCamera();
    /**
     * Blocks until a new image arrives.
     */
    virtual std::unique_ptr<RGBDFrameData> getImageSync() override;


    virtual bool isOpened() override { return currentId < (int)frames.size(); }

    size_t getFrameCount() { return frames.size(); }


   private:
    void preload(const std::string& datasetDir, bool multithreaded);

    std::vector<std::unique_ptr<RGBDFrameData>> frames;

    Timer timer;
    tick_t timeStep;
    tick_t lastFrameTime;
    tick_t nextFrameTime;
};

}  // namespace Saiga
