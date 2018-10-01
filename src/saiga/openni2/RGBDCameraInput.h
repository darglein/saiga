/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/image/image.h"
#include "saiga/camera/RGBDCamera.h"
#include "saiga/util/synchronizedBuffer.h"

#include <thread>

// Use shared pointer of openni objects so that we don't have to include the header here
namespace openni {
class Device;
class VideoStream;
class VideoFrameRef;
}

namespace Saiga {


class SAIGA_GLOBAL RGBDCameraInput : public RGBDCamera
{
public:


    RGBDCameraInput(CameraOptions rgbo, CameraOptions deptho, float depthFactor = 1.0/1000.0);
    ~RGBDCameraInput();



    /**
     * Blocks until a new image arrives.
     */
    virtual std::shared_ptr<FrameData> waitForImage() override;

    /**
     * Tries to return the last dslr image.
     * If none are ready a nullptr is returned.
     */
    virtual std::shared_ptr<FrameData> tryGetImage() override;


    virtual bool isOpened() override;


    // The user can change these variables, but must call 'updateCameraSettings' to make the take effect
    bool autoexposure = true;
    int exposure = 33;

    bool autoWhiteBalance = true;
    int gain = 300;
    void updateCameraSettings();
private:

    SynchronizedBuffer<std::shared_ptr<FrameData>> frameBuffer;

    std::shared_ptr<openni::Device> device;
    std::shared_ptr<openni::VideoStream> depth, color;
    std::shared_ptr<openni::VideoFrameRef> m_depthFrame,m_colorFrame;

    bool open();
    void resetCamera();
    bool waitFrame(FrameData& data);
    bool readDepth(DepthImageType::ViewType depthImg);
    bool readColor(RGBImageType::ViewType colorImg);

    std::thread eventThread;

    bool foundCamera = false;
    bool running = false;
    float depthFactor;
    bool updateS = false;
    void updateSettingsIntern();

    void eventLoop();
};

}
