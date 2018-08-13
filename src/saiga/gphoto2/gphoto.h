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


    bool hasNewImage(Image& img);
private:
    array_view<const char> adata;
    bool gotImage = false;
    std::string imageName;
    std::string imageDir;
    std::vector<uint8_t> data;

    std::mutex mut;
    std::thread eventThread;

    void *context;
    void	*camera;

    bool running = true;

    void eventLoop();

//    void trigger();

//    std::vector<queue_entry>waitCaptureComplete();

//    void clearEventQueue();

};

}
