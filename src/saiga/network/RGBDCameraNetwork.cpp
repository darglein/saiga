/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "RGBDCameraNetwork.h"
#include "saiga/image/imageTransformations.h"

#include "internal/noGraphicsAPI.h"
#include "saiga/network/ImageTransmition.h"


namespace Saiga {

void RGBDCameraNetwork::connect(std::string host, uint32_t port)
{
    trans = std::make_shared<ImageTransmition>(host,port);
    trans->makeReciever();

    Image img;
    int gotC = false;
    int gotD = false;
    cout << "rec " << img << endl;
    while(!gotC || !gotD){
        trans->recieveImage(img);
        cout << "rec " << img << endl;
        if(img.type == Saiga::UC4)
        {
            if(!gotC)
            {
//                colorImg.create(img.height,img.width,img.pitchBytes);
                colorW = img.width;

                colorH = img.height;
                gotC = true;
            }

        }else{

            if(!gotD)
            {
                depthW = img.width;

                depthH = img.height;
//                depthImg.create(img.height,img.width,img.pitchBytes);
                gotD = true;
            }
        }
    }
}


bool RGBDCameraNetwork::readFrame(FrameData &data)
{
    while(true)
    {
        while(!trans->recieveImageType(data.colorImg))
        {

        }

        if(trans->recieveImageType(data.depthImg))
            return true;
    }

    return false;
}

}
