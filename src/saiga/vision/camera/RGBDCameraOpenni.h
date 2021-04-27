/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#ifdef SAIGA_USE_OPENNI2
#    include "saiga/core/image/image.h"
#    include "saiga/core/util/Thread/SynchronizedBuffer.h"
#    include "saiga/vision/camera/CameraBase.h"

#    include <thread>

// Use shared pointer of openni objects so that we don't have to include the header here
namespace openni
{
class Device;
class VideoStream;
class VideoFrameRef;
}  // namespace openni

namespace Saiga
{
class SAIGA_VISION_API RGBDCameraOpenni : public CameraBase
{
   public:
    RGBDCameraOpenni(const RGBDIntrinsics& intr);
    virtual ~RGBDCameraOpenni();

    /**
     * Blocks until a new image arrives.
     */
    virtual bool getImageSync(FrameData& data) override;



    virtual void close() override;
    virtual bool isOpened() override;

    /**
     * Tries to open a camera and set the given parameters.
     * Returns true if it was sucessfull.
     */
    bool tryOpen();

    // The user can change these variables, but must call 'updateCameraSettings' to make the take effect
    bool autoexposure = true;
    //    int exposure      = 33;

    bool autoWhiteBalance = true;
    //    int gain              = 300;
    void updateCameraSettings();


    void imgui();
    const RGBDIntrinsics& intrinsics() { return _intrinsics; }

   private:
    std::shared_ptr<openni::Device> device;
    std::shared_ptr<openni::VideoStream> depth, color;
    std::shared_ptr<openni::VideoFrameRef> m_depthFrame, m_colorFrame;

    void resetCamera();
    bool waitFrame(FrameData& data);
    bool readDepth(DepthImageType::ViewType depthImg);
    bool readColor(RGBImageType::ViewType colorImg);

    bool foundCamera = false;
    bool updateS     = false;
    RGBDIntrinsics _intrinsics;
    void updateSettingsIntern();
};

}  // namespace Saiga
#endif
