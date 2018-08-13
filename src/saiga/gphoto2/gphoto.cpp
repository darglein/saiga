/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "gphoto.h"

#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "saiga/util/file.h"
#include "saiga/util/tostring.h"

#include "gphoto2/gphoto2.h"
#include "gphoto2/gphoto2-camera.h"

namespace Saiga {


static void
ctx_error_func (GPContext *context, const char *str, void *data)
{
    fprintf  (stderr, "\n*** Contexterror ***              \n%s\n",str);
    fflush   (stderr);
}

static void
ctx_status_func (GPContext *context, const char *str, void *data)
{
    fprintf  (stderr, "%s\n", str);
    fflush   (stderr);
}

GPhoto::GPhoto()
{


    cout << "GPhoto::GPhoto" << endl;
    context = gp_context_new();

    /* All the parts below are optional! */
    gp_context_set_error_func ((GPContext*)context, ctx_error_func, NULL);
    gp_context_set_status_func ((GPContext*)context, ctx_status_func, NULL);


    gp_camera_new((Camera**)&camera);

    printf("Camera init.  Takes about 10 seconds.\n");
    auto retval = gp_camera_init((Camera*)camera, (GPContext*)context);
    if (retval != GP_OK) {
        printf("  Retval of capture_to_file: %d\n", retval);
        exit (1);
    }



    eventThread = std::thread(&GPhoto::eventLoop,this);

}

GPhoto::~GPhoto()
{
    running = false;
    eventThread.join();

    gp_camera_free((Camera*)camera);

}

bool GPhoto::hasNewImage(Image &img)
{
    std::unique_lock<std::mutex> l(mut);

    if(gotImage)
    {

        img.loadFromMemory(adata);

        gotImage = false;
        return true;
    }

    return false;
}

struct Event{
    CameraEventType	evtype;
    void		*data;
};

Event getEvent(  GPContext *context,
                 Camera	*camera)
{
    Event event;
    auto retval = gp_camera_wait_for_event(camera, 10, &event.evtype, &event.data, context);

    if (retval != GP_OK) {
        fprintf (stderr, "return from waitevent in trigger sample with %d\n", retval);
        return event;
    }


    //      cout << "got event " << event.evtype << endl;
    return event;
}

void GPhoto::eventLoop()
{
    CameraFile *file;
    gp_file_new(&file);

    while(running)
    {
        auto e = getEvent((GPContext*)context,(Camera*)camera);

        if(e.evtype == GP_EVENT_FILE_ADDED)
        {
            CameraFilePath	*path = (CameraFilePath	*)e.data;
            std::string name = path->name;

            if(hasEnding(name,"jpg"))
            {
                std::unique_lock<std::mutex> l(mut);
                imageName = path->name;
                imageDir = path->folder;
                cout << imageName << " " << imageDir << endl;
                gotImage = true;


                gp_camera_file_get((Camera*)camera, imageDir.c_str(), imageName.c_str(),
                                   GP_FILE_TYPE_NORMAL, file, (GPContext*)context);



                size_t size;
                const char	*data;
                gp_file_get_data_and_size (file, &data, &size);

                cout << "size " << size << endl;

                adata = array_view<const char>(data,size);
            }

        }
    }

    gp_file_free(file);
}


#if 0
std::vector<GPhoto::queue_entry>
GPhoto::waitCaptureComplete ()
{
    CameraFilePath	*path;
    int		retval;

    std::vector<GPhoto::queue_entry> queue;

    while(true)
    {

        Event e = getEvent();

        cout << "got event " << e.evtype << endl;
        path = (CameraFilePath	*)e.data;
        switch (e.evtype) {

        case GP_EVENT_CAPTURE_COMPLETE:
            return queue;
        case GP_EVENT_UNKNOWN:
        case GP_EVENT_TIMEOUT:
        case GP_EVENT_FOLDER_ADDED:
            break;
        case GP_EVENT_FILE_ADDED:
            fprintf (stderr, "File %s / %s added to queue.\n", path->folder, path->name);
            queue_entry qe;
            qe.offset = 0;
            qe.path = *path;
            queue.push_back(qe);
            break;
        }
    }
    return queue;

    if (complete)
    {
        complete = false;

        queue_entry qe = queue[1];

        uint64_t	size = buffersize;


        //        retval = gp_camera_file_read (camera,
        //                                      qe.path.folder,
        //                                      qe.path.name,
        //                                      GP_FILE_TYPE_NORMAL,
        //                                      qe.offset,
        //                                      buffer,
        //                                      &size,
        //                                      context
        //                                      );

        CameraFile *file;
        retval = gp_file_new(&file);
        printf("  Retval: %d\n", retval);
        retval = gp_camera_file_get(camera, qe.path.folder, qe.path.name,
                                    GP_FILE_TYPE_NORMAL, file, context);
        printf("  Retval: %d\n", retval);



        const char	*data;
        gp_file_get_data_and_size (file, &data, &size);

        cout << "size " << size << endl;


        //        auto fd = open(qe.path.name, O_RDWR, 0644);

        //        cout << "fd " << fd << " " << qe.path.name << endl;
        //            if (-1 == lseek(fd, qe.offset, SEEK_SET))
        //                perror("lseek");
        //            if (-1 == write (fd, data, size))
        //                perror("write");
        //         close (fd);


        array_view<const char> adata(data,size);
        //        File::saveFileBinary("test.jpg",adata);


        Image img;
        img.loadFromMemory(adata);

        img.save("test.png");
        //        fipImage fimg;

        //        fipMemoryIO fipmem( (BYTE*)adata.data(),adata.size());
        //        fimg.loadFromMemory(fipmem);
        //        fimg.save("test.png");

        //        fimg.saveToMemory()


        gp_file_free(file);

        fprintf(stderr,"done camera readfile size was %d\n", size);
        if (retval != GP_OK) {
            fprintf (stderr,"gp_camera_file_read failed: %d\n", retval);
            return;
        }
    }
    return;
}



void GPhoto::clearEventQueue()
{
    Event e;

    while(true){
        e = getEvent();
        if(e.evtype == GP_EVENT_TIMEOUT)
            return;
    }
}



void GPhoto::trigger()
{
    auto retval = gp_camera_trigger_capture (camera, context);
    cout << "trigger " << retval << endl;

    //    CameraFilePath camera_file_path;
    //    auto retval = gp_camera_capture(camera, GP_CAPTURE_IMAGE, &camera_file_path, context);

    //    CameraFile* camera_file;
    //    gp_file_new(&camera_file);
    //    auto retval = gp_camera_capture_preview(camera, camera_file, context);
}

#endif

}
