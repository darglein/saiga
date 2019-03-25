/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "FileRGBDCamera.h"

#include "saiga/core/util/directory.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/tostring.h"

#include <fstream>
#include <thread>
namespace Saiga
{
FileRGBDCamera::FileRGBDCamera(const std::string& datasetDir, double depthFactor, int maxFrames, int fps,
                               const std::shared_ptr<DMPP>& dmpp)
    : maxFrames(maxFrames)
{
    this->dmpp = dmpp;
    cout << "Loading File RGBD Dataset: " << datasetDir << endl;

    load(datasetDir);

    timeStep = std::chrono::duration_cast<tick_t>(std::chrono::duration<double, std::milli>(1000.0 / double(fps)));

    timer.start();
    lastFrameTime = timer.stop();
    nextFrameTime = lastFrameTime + timeStep;
}

FileRGBDCamera::~FileRGBDCamera()
{
    cout << "~FileRGBDCamera" << endl;
}

std::shared_ptr<RGBDCamera::FrameData> FileRGBDCamera::waitForImage()
{
    if (!isOpened())
    {
        return nullptr;
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


    auto img = frames[currentId];
    setNextFrame(*img);
    return img;
}


void FileRGBDCamera::load(const std::string& datasetDir)
{
    Directory dir(datasetDir);


    std::vector<std::string> rgbImages;
    std::vector<std::string> depthImages;
    dir.getFiles(rgbImages, ".png");
    dir.getFiles(depthImages, ".saigai");

    cout << "Found Color/Depth Images: " << rgbImages.size() << "/" << depthImages.size() << endl;

    SAIGA_ASSERT(rgbImages.size() == depthImages.size());
    SAIGA_ASSERT(rgbImages.size() > 0);

    std::sort(rgbImages.begin(), rgbImages.end());
    std::sort(depthImages.begin(), depthImages.end());

    if (maxFrames <= 0) maxFrames = rgbImages.size();


    frames.resize(maxFrames);

#pragma omp parallel for
    for (int i = 0; i < maxFrames; ++i)
    {
        auto& f = frames[i];

        RGBImageType cimg;
        cimg.load(dir() + rgbImages[i]);
        rgbo.h = cimg.h;
        rgbo.w = cimg.w;


        DepthImageType dimg;
        dimg.load(dir() + depthImages[i]);
        bool downScale = (dmpp && dmpp->params.apply_downscale) ? true : false;
        int targetW    = downScale ? dimg.w / 2 : dimg.w;
        int targetH    = downScale ? dimg.h / 2 : dimg.h;
        deptho.w       = targetW;
        deptho.h       = targetH;


        f = makeFrameData();

        if (dmpp)
        {
            (*dmpp)(dimg, f->depthImg.getImageView());
        }
        else
        {
            f->depthImg.load(dir() + depthImages[i]);
        }

        f->colorImg = cimg;
    }

    cout << "Loading done." << endl;


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
        //        cout << "loading " << d.rgb.img << endl;


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

    cout << "Loaded " << tumframes.size() << " images." << endl;
#endif
}



}  // namespace Saiga
