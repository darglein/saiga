/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/image/image.h"

namespace Saiga {



class SAIGA_GLOBAL RGBDCamera
{
public:


    TemplatedImage<ucvec4> colorImg;
    TemplatedImage<unsigned short> depthImg;

    virtual bool readFrame() = 0;
};

}
