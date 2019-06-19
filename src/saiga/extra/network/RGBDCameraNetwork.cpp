/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "RGBDCameraNetwork.h"

#if 0

#    ifdef SAIGA_VISION
#        include "saiga/core/image/imageTransformations.h"

#        include "internal/noGraphicsAPI.h"

#        include "ImageTransmition.h"


namespace Saiga
{
void RGBDCameraNetwork::connect(std::string host, uint32_t port)
{
    trans = std::make_shared<ImageTransmition>(host, port);
    trans->makeReciever();

    Image img;
    int gotC = false;
    int gotD = false;
    std::cout << "rec " << img << std::endl;
    while (!gotC || !gotD)
    {
        trans->recieveImage(img);
        std::cout << "rec " << img << std::endl;
        if (img.type == Saiga::UC3)
        {
            if (!gotC)
            {
                //                colorImg.create(img.height,img.width,img.pitchBytes);
                rgbo.w = img.width;

                rgbo.h = img.height;
                gotC   = true;
            }
        }
        else
        {
            if (!gotD)
            {
                deptho.w = img.width;

                deptho.h = img.height;
                //                depthImg.create(img.height,img.width,img.pitchBytes);
                gotD = true;
            }
        }
    }
}

std::unique_ptr<RGBDFrameData> RGBDCameraNetwork::waitForImage()
{
    auto data = makeFrameData();
    while (true)
    {
        while (!trans->recieveImageType(data->colorImg))
        {
        }

        if (trans->recieveImageType(data->depthImg)) return data;
    }

    return nullptr;
}

std::unique_ptr<RGBDFrameData> RGBDCameraNetwork::tryGetImage()
{
    return nullptr;
}



}  // namespace Saiga
#    endif


#endif
