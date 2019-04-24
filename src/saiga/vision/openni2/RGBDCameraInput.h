/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/image/image.h"
#include "saiga/core/util/synchronizedBuffer.h"

#include <thread>

#ifdef SAIGA_VISION
#    include "saiga/vision/RGBDCamera.h"

// Use shared pointer of openni objects so that we don't have to include the header here
namespace openni
{
class Device;
class VideoStream;
class VideoFrameRef;
}  // namespace openni

namespace Saiga
{
class SAIGA_VISION_API RGBDCameraOpenni : public RGBDCamera
{
   public:
    RGBDCameraOpenni(const RGBDIntrinsics& intr);
    virtual ~RGBDCameraOpenni();



    /**
     * Blocks until a new image arrives.
     */
    virtual std::shared_ptr<RGBDFrameData> getImageSync() override;

    /**
     * Tries to return the last dslr image.
     * If none are ready a nullptr is returned.
     */
    virtual std::shared_ptr<RGBDFrameData> getImage() override;


    virtual void close() override;
    virtual bool isOpened() override;

    // The user can change these variables, but must call 'updateCameraSettings' to make the take effect
    bool autoexposure = true;
    //    int exposure      = 33;

    bool autoWhiteBalance = true;
    //    int gain              = 300;
    void updateCameraSettings();


    void imgui();

   private:
    SynchronizedBuffer<std::shared_ptr<RGBDFrameData>> frameBuffer;

    std::shared_ptr<openni::Device> device;
    std::shared_ptr<openni::VideoStream> depth, color;
    std::shared_ptr<openni::VideoFrameRef> m_depthFrame, m_colorFrame;

    bool open();
    void resetCamera();
    bool waitFrame(RGBDFrameData& data);
    bool readDepth(DepthImageType::ViewType depthImg);
    bool readColor(RGBImageType::ViewType colorImg);

    std::thread eventThread;

    bool foundCamera = false;
    bool running     = false;
    float depthFactor;
    bool updateS = false;
    void updateSettingsIntern();

    void eventLoop();
};

}  // namespace Saiga
#endif
