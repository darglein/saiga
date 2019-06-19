/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/image/image.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/Thread/SynchronizedBuffer.h"

#include <thread>

namespace Saiga
{
/**
 * Currently only tested with Canon 5D Mark 3!
 *
 * Simple helper class to get the images from a DSLR camera when the trigger is pressed.
 *
 * This class starts a background thread that checks for events and downloads the appropriate
 * files from the camera. The main thread can grab and process the images by calling waitForImage
 * or tryWaitForImage. By default the jpg image is also extracted and stored in the img variable
 * of DSLRImage.
 */

class SAIGA_EXTRA_API GPhoto
{
   public:
    struct DSLRImage
    {
        TemplatedImage<ucvec3> img;
        std::vector<char> jpgImage;
        std::vector<char> rawImage;


        void jpgToImage(Image& img)
        {
            img.loadFromMemory(jpgImage);
            SAIGA_ASSERT(img.type == TemplatedImage<ucvec3>::TType::type);
        }
        void saveRaw(std::string file) { File::saveFileBinary(file + ".cr2", rawImage.data(), rawImage.size()); }
        void saveJpg(std::string file) { File::saveFileBinary(file + ".jpg", jpgImage.data(), jpgImage.size()); }
    };

    bool autoConvert = true;


    /**
     * Initializes the context and connects to a camera.
     * You can use the function isOpenend to check for success.
     */
    GPhoto();
    ~GPhoto();

    bool isOpenend() { return foundCamera; }

    /**
     * Blocks until a new image arrives.
     */
    std::shared_ptr<DSLRImage> waitForImage();

    /**
     * Tries to return the last dslr image.
     * If none are ready a nullptr is returned.
     */
    std::shared_ptr<DSLRImage> tryGetImage();

   private:
    SynchronizedBuffer<std::shared_ptr<DSLRImage>> imageBuffer;

    std::thread eventThread;

    bool foundCamera = false;
    void* context;
    void* camera = nullptr;

    bool running = false;

    bool connectToCamera();

    void eventLoop();
    void clearEvents();

    //    void trigger();

    //    std::vector<queue_entry>waitCaptureComplete();

    //    void clearEventQueue();
};

}  // namespace Saiga
