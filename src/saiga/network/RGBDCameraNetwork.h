/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/image/image.h"
#include "saiga/camera/RGBDCamera.h"




namespace Saiga {

class ImageTransmition;

class SAIGA_GLOBAL RGBDCameraNetwork : public RGBDCamera
{
public:

    void connect(std::string host, uint32_t port);

    bool readFrame() override;
private:
    std::shared_ptr<ImageTransmition> trans;
};

}

