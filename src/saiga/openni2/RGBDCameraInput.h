/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/image/image.h"
#include "saiga/camera/RGBDCamera.h"


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
    struct CameraOptions
    {
        int w = 640;
        int h = 480;
        int fps = 30;
    };


    bool open(CameraOptions rgbo, CameraOptions deptho);

    bool readFrame(FrameData& data) override;
private:
    std::shared_ptr<openni::Device> device;
    std::shared_ptr<openni::VideoStream> depth, color;
    std::shared_ptr<openni::VideoFrameRef> m_depthFrame,m_colorFrame;

    bool readDepth(ImageView<unsigned short> depthImg);
    bool readColor(ImageView<ucvec3> colorImg);
};

}
