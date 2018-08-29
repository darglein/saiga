/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/image/image.h"

#include "saiga/util/synchronizedBuffer.h"
#include "saiga/util/file.h"


#include <thread>
#include <mutex>

namespace Saiga {


class SAIGA_GLOBAL GPhoto
{
public:
    struct DSLRImage
    {
        Image img;
        std::vector<char> jpgImage;
        std::vector<char> rawImage;


        void jpgToImage(Image& img)
        {
            img.loadFromMemory(jpgImage);
        }
        void saveRaw(std::string file)
        {
            File::saveFileBinary(file+".cr2",rawImage.data(),rawImage.size());
        }
        void saveJpg(std::string file)
        {
            File::saveFileBinary(file+".jpg",jpgImage.data(),jpgImage.size());
        }
    };

    bool autoConvert = true;


    GPhoto();
    ~GPhoto();

    bool isOpenend() { return foundCamera; }


    std::shared_ptr<DSLRImage> waitForImage();
    std::shared_ptr<DSLRImage> tryWaitForImage();


//    bool hasNewImage();
//    void getImage(Image& img);
private:

    SynchronizedBuffer<std::shared_ptr<DSLRImage>> imageBuffer;


    std::mutex mut;
    std::thread eventThread;

    bool foundCamera = false;
    void *context;
    void	*camera;

    bool running = false;

    void eventLoop();
    void clearEvents();

//    void trigger();

//    std::vector<queue_entry>waitCaptureComplete();

//    void clearEventQueue();

};

}
