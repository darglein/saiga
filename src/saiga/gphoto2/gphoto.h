/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/image/image.h"




#include <thread>
#include <mutex>

namespace Saiga {


class SAIGA_GLOBAL GPhoto
{
public:



    GPhoto();
    ~GPhoto();

    bool isOpenend() { return foundCamera; }


    bool hasNewImage(Image& img);
private:
    ArrayView<const char> adata;
    bool gotImage = false;
    std::string imageName;
    std::string imageDir;
    std::vector<uint8_t> data;

    std::mutex mut;
    std::thread eventThread;

    bool foundCamera = false;
    void *context;
    void	*camera;

    bool running = false;

    void eventLoop();

//    void trigger();

//    std::vector<queue_entry>waitCaptureComplete();

//    void clearEventQueue();

};

}
