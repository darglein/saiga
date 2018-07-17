/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/image/image.h"


// Use shared pointer of openni objects so that we don't have to include the header here
namespace openni {
class Device;
class VideoStream;
class VideoFrameRef;
}

namespace Saiga {


class SAIGA_GLOBAL RGBDCamera
{
public:
    int colorW, colorH;
    int depthW, depthH;

    TemplatedImage<ucvec4> colorImg;
    TemplatedImage<unsigned short> depthImg;

    bool open();

    bool readFrame();
private:
    std::shared_ptr<openni::Device> device;
    std::shared_ptr<openni::VideoStream> depth, color;
    std::shared_ptr<openni::VideoFrameRef> m_depthFrame,m_colorFrame;
};

}
