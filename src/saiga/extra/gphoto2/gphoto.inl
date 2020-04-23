/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/Thread/threadName.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/tostring.h"

#include "internal/noGraphicsAPI.h"

#include "gphoto.h"

#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>



#define CHECK_GP(_X)                                                                             \
    {                                                                                            \
        auto ret = _X;                                                                           \
        if (ret != GP_OK)                                                                        \
        {                                                                                        \
            std::cout << "Gphoto error in " << #_X << std::endl << "code: " << ret << std::endl; \
            SAIGA_ASSERT(0);                                                                     \
        }                                                                                        \
    }

namespace Saiga
{
GPhoto::GPhoto() : imageBuffer(10)
{
    context = gp_context_new();
    SAIGA_ASSERT(context);

    /* All the parts below are optional! */
    //    gp_context_set_error_func ((GPContext*)context, ctx_error_func, NULL);
    //    gp_context_set_status_func ((GPContext*)context, ctx_status_func, NULL);

    eventThread = std::thread(&GPhoto::eventLoop, this);
}


GPhoto::~GPhoto()
{
    std::cout << "~GPhoto" << std::endl;
    if (running)
    {
        running = false;
        eventThread.join();
    }

    if (camera)
    {
        gp_camera_free((Camera*)camera);
        std::cout << "DSLR Camera closed." << std::endl;
    }
}

std::shared_ptr<GPhoto::DSLRImage> GPhoto::waitForImage()
{
    return imageBuffer.get();
}

std::shared_ptr<GPhoto::DSLRImage> GPhoto::tryGetImage()
{
    std::shared_ptr<GPhoto::DSLRImage> img;
    imageBuffer.tryGet(img);
    return img;
}

bool GPhoto::connectToCamera()
{
    if (camera)
    {
        gp_camera_free((Camera*)camera);
        camera = nullptr;
    }

    gp_camera_new((Camera**)&camera);

    if (!camera) return false;


    auto retval = gp_camera_init((Camera*)camera, (GPContext*)context);
    if (retval != GP_OK)
    {
        return false;
    }

    return true;
}



struct Event
{
    CameraEventType evtype;
    void* data;
};

bool getEvent(GPContext* context, Camera* camera, Event& event)
{
    auto retval = gp_camera_wait_for_event(camera, 10, &event.evtype, &event.data, context);

    if (retval != GP_OK)
    {
        std::cerr << "gp_camera_wait_for_event failed with error code " << retval << std::endl;
        return false;
    }
    return true;
}

void GPhoto::clearEvents()
{
    if (!camera) return;

    while (true)
    {
        Event event;
        CHECK_GP(gp_camera_wait_for_event((Camera*)camera, 10, &event.evtype, &event.data, (GPContext*)context));
        if (event.evtype == GP_EVENT_TIMEOUT) return;
    }
}

void GPhoto::eventLoop()
{
    setThreadName("Saiga::GPhoto");

    std::cout << "Starting GPhoto2... " << std::endl;

    running = true;

    CameraFile* file;
    gp_file_new(&file);

    // 0: no image recieved
    // 1: one image (either jpg or raw)
    // 2 == 0: both images recieved
    int state                      = 0;
    std::shared_ptr<DSLRImage> tmp = std::make_shared<DSLRImage>();


    while (running)
    {
        if (!foundCamera)
        {
            foundCamera = connectToCamera();

            if (!foundCamera)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            else
            {
                clearEvents();
                std::cout << "DSLR Camera opened." << std::endl;
            }
        }
        Event e;
        auto gotEvent = getEvent((GPContext*)context, (Camera*)camera, e);
        if (!gotEvent)
        {
            std::cout << "Camera connection lost!" << std::endl;
            foundCamera = false;
            continue;
        }

        if (e.evtype == GP_EVENT_FILE_ADDED)
        {
            CameraFilePath* path = (CameraFilePath*)e.data;

            std::string imageName = path->name;
            std::string imageDir  = path->folder;

            std::cout << "GP_EVENT_FILE_ADDED: " << imageName << std::endl;

            if (hasEnding(imageName, "jpg"))
            {
                CHECK_GP(gp_camera_file_get((Camera*)camera, imageDir.c_str(), imageName.c_str(), GP_FILE_TYPE_NORMAL,
                                            file, (GPContext*)context));
                size_t size;
                const char* data;
                CHECK_GP(gp_file_get_data_and_size(file, &data, &size));
                tmp->jpgImage.resize(size);
                std::copy(data, data + size, tmp->jpgImage.begin());
                state |= 1;
            }

            if (hasEnding(imageName, "cr2"))
            {
                CHECK_GP(gp_camera_file_get((Camera*)camera, imageDir.c_str(), imageName.c_str(), GP_FILE_TYPE_NORMAL,
                                            file, (GPContext*)context));
                size_t size;
                const char* data;
                CHECK_GP(gp_file_get_data_and_size(file, &data, &size));
                tmp->rawImage.resize(size);
                std::copy(data, data + size, tmp->rawImage.begin());
                state |= 2;
            }

            if (state == 3)
            {
                // Add to queue and create new image
                if (autoConvert)
                {
                    tmp->jpgToImage(tmp->img);
                }

                imageBuffer.add(tmp);
                tmp   = std::make_shared<DSLRImage>();
                state = 0;
            }
        }
    }

    gp_file_free(file);
}


}  // namespace Saiga
