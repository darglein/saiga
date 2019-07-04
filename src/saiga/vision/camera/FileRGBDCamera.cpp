/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "FileRGBDCamera.h"

#include "saiga/core/util/ProgressBar.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/tostring.h"

#include <fstream>
#include <thread>
namespace Saiga
{
FileRGBDCamera::FileRGBDCamera(const std::string& datasetDir, const RGBDIntrinsics& intr, bool _preload,
                               bool multithreaded)
    : RGBDCamera(intr)
{
    std::cout << "Loading File RGBD Dataset: " << datasetDir << std::endl;

    if (_preload)
    {
        preload(datasetDir, multithreaded);
    }
    else
    {
        SAIGA_EXIT_ERROR("Not implemented!");
    }
    timeStep = std::chrono::duration_cast<tick_t>(
        std::chrono::duration<double, std::milli>(1000.0 / double(intrinsics().fps)));

    timer.start();
    lastFrameTime = timer.stop();
    nextFrameTime = lastFrameTime + timeStep;
}

FileRGBDCamera::~FileRGBDCamera()
{
    std::cout << "~FileRGBDCamera" << std::endl;
}

bool FileRGBDCamera::getImageSync(RGBDFrameData& data)
{
    if (!isOpened())
    {
        return false;
    }


    auto t = timer.stop();

    if (t < nextFrameTime)
    {
        std::this_thread::sleep_for(nextFrameTime - t);
        nextFrameTime += timeStep;
    }
    else if (t < nextFrameTime + timeStep)
    {
        nextFrameTime += timeStep;
    }
    else
    {
        nextFrameTime = t + timeStep;
    }


    auto&& img = frames[currentId];
    setNextFrame(img);
    data = std::move(img);
    return true;
}


void FileRGBDCamera::preload(const std::string& datasetDir, bool multithreaded)
{
    Directory dir(datasetDir);


    std::vector<std::string> rgbImages;
    std::vector<std::string> depthImages;
    dir.getFiles(rgbImages, ".png");
    dir.getFiles(depthImages, ".saigai");


    SAIGA_ASSERT(rgbImages.size() == depthImages.size());
    SAIGA_ASSERT(rgbImages.size() > 0);

    std::sort(rgbImages.begin(), rgbImages.end());
    std::sort(depthImages.begin(), depthImages.end());

    if (intrinsics().maxFrames <= 0) _intrinsics.maxFrames = rgbImages.size();

    //    std::cout << "Found Color/Depth Images: " << rgbImages.size() << "/" << depthImages.size() << " Loading "
    //         << _intrinsics.maxFrames << " images..." << std::endl;

    int N = intrinsics().maxFrames;
    frames.resize(N);

    SyncedConsoleProgressBar loadingBar(std::cout, "Loading " + to_string(N) + " images ", N);

#pragma omp parallel for if (multithreaded)
    for (int i = 0; i < N; ++i)
    {
        auto& f = frames[i];
        makeFrameData(f);

        //        std::cout << "dir: " << dir() + rgbImages[i] << std::endl;
        RGBImageType cimg(dir() + "/" + rgbImages[i]);
        //        cimg.load(dir() + rgbImages[i]);

        // make sure it matches the defined intrinsics
        SAIGA_ASSERT(cimg.h == intrinsics().rgbo.h);
        SAIGA_ASSERT(cimg.w == intrinsics().rgbo.w);
        //        rgbo.h = cimg.h;
        //        rgbo.w = cimg.w;


        DepthImageType dimg(dir() + "/" + depthImages[i]);
        //        dimg.load(dir() + depthImages[i]);

        // make sure it matches the defined intrinsics
        SAIGA_ASSERT(dimg.h == intrinsics().deptho.h);
        SAIGA_ASSERT(dimg.w == intrinsics().deptho.w);

        //        bool downScale = (dmpp && dmpp->params.apply_downscale) ? true : false;
        //        int targetW    = downScale ? dimg.w / 2 : dimg.w;
        //        int targetH    = downScale ? dimg.h / 2 : dimg.h;
        //        deptho.w       = targetW;
        //        deptho.h       = targetH;



        //        if (dmpp)
        //        {
        //            (*dmpp)(dimg, f->depthImg.getImageView());
        //        }
        //        else
        {
            //            f->depthImg.load(dir() + depthImages[i]);
        }

        f.depthImg = std::move(dimg);
        f.colorImg = std::move(cimg);
        loadingBar.addProgress(1);
    }

    //    std::cout << "Loading done." << std::endl;


#if 0
    if (maxFrames >= 0)
    {
        tumframes.resize(std::min((size_t)maxFrames, tumframes.size()));
    }


    frames.resize(tumframes.size());

#    pragma omp parallel for
    for (int i = 0; i < (int)tumframes.size(); ++i)
    {
        TumFrame d = tumframes[i];
        //        std::cout << "loading " << d.rgb.img << std::endl;


        Image cimg(datasetDir + "/" + d.rgb.img);
        rgbo.h = cimg.h;
        rgbo.w = cimg.w;


        Image dimg(datasetDir + "/" + d.depth.img);

        bool downScale = (dmpp && dmpp->params.apply_downscale) ? true : false;
        int targetW    = downScale ? dimg.w / 2 : dimg.w;
        int targetH    = downScale ? dimg.h / 2 : dimg.h;

        deptho.w = targetW;
        deptho.h = targetH;

        auto f = makeFrameData();


        if (cimg.type == UC3)
        {
            // convert to rgba
            ImageTransformation::addAlphaChannel(cimg.getImageView<ucvec3>(), f->colorImg);
        }
        else if (cimg.type == UC4)
        {
            cimg.getImageView<ucvec4>().copyTo(f->colorImg.getImageView());
        }
        else
        {
            SAIGA_ASSERT(0);
        }

        DepthImageType tmp;
        tmp.create(dimg.h, dimg.w);

        if (dimg.type == US1)
        {
            dimg.getImageView<unsigned short>().copyTo(tmp.getImageView(), depthFactor);
        }
        else
        {
            SAIGA_ASSERT(0);
        }

        if (dmpp)
        {
            (*dmpp)(tmp, f->depthImg.getImageView());
        }
        else
        {
            tmp.getImageView().copyTo(f->depthImg.getImageView());
        }


        frames[i] = f;
    }

    std::cout << "Loaded " << tumframes.size() << " images." << std::endl;
#endif
}



}  // namespace Saiga
