/**
 * Copyright (c) 2021 Darius Rückert
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
FileRGBDCamera::FileRGBDCamera(const DatasetParameters& params, const RGBDIntrinsics& intr)
    : DatasetCameraBase(params), _intrinsics(intr)
{
    std::cout << "Loading File RGBD Dataset: " << params.dir << std::endl;
    preload(params.dir, params.multiThreadedLoad);
}

FileRGBDCamera::~FileRGBDCamera() {}


void FileRGBDCamera::preload(const std::string& datasetDir, bool multithreaded)
{
    Directory dir(datasetDir);


    std::vector<std::string> rgbImages;
    std::vector<std::string> depthImages;
    rgbImages   = dir.getFilesEnding(".png");
    depthImages = dir.getFilesEnding(".saigai");


    SAIGA_ASSERT(rgbImages.size() == depthImages.size());
    SAIGA_ASSERT(rgbImages.size() > 0);

    std::sort(rgbImages.begin(), rgbImages.end());
    std::sort(depthImages.begin(), depthImages.end());

    if (params.maxFrames <= 0) params.maxFrames = rgbImages.size();

    //    std::cout << "Found Color/Depth Images: " << rgbImages.size() << "/" << depthImages.size() << " Loading "
    //         << _intrinsics.maxFrames << " images..." << std::endl;

    int N = params.maxFrames;
    frames.resize(N);

    ProgressBar loadingBar(std::cout, "Loading " + to_string(N) + " images ", N);

#pragma omp parallel for if (multithreaded)
    for (int i = 0; i < N; ++i)
    {
        auto& f = frames[i];

        RGBImageType cimg(dir() + "/" + rgbImages[i]);
        //        cimg.load(dir() + rgbImages[i]);

        // make sure it matches the defined intrinsics
        SAIGA_ASSERT(cimg.dimensions() == intrinsics().imageSize);

        //        rgbo.h = cimg.h;
        //        rgbo.w = cimg.w;


        DepthImageType dimg(dir() + "/" + depthImages[i]);
        //        dimg.load(dir() + depthImages[i]);

        // make sure it matches the defined intrinsics
        SAIGA_ASSERT(dimg.dimensions() == intrinsics().depthImageSize);

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

        f.depth_image = std::move(dimg);
        f.image_rgb   = std::move(cimg);
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
