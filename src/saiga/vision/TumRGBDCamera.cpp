/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "TumRGBDCamera.h"

#include "saiga/util/tostring.h"

#include <fstream>
#include <thread>
namespace Saiga
{
static std::vector<TumRGBDCamera::CameraData> readCameraData(std::string file)
{
    std::vector<TumRGBDCamera::CameraData> data;
    {
        std::ifstream strm(file);
        SAIGA_ASSERT(strm.is_open());
        std::string line;
        while (std::getline(strm, line))
        {
            if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> elements = Saiga::split(line, ' ');
            SAIGA_ASSERT(elements.size() == 2);
            TumRGBDCamera::CameraData d;
            d.timestamp = Saiga::to_double(elements[0]);
            d.img       = elements[1];
            data.push_back(d);
        }
    }
    return data;
}

static std::vector<TumRGBDCamera::GroundTruth> readGT(std::string file)
{
    std::vector<TumRGBDCamera::GroundTruth> data;
    {
        std::ifstream strm(file);
        SAIGA_ASSERT(strm.is_open());
        std::string line;
        while (std::getline(strm, line))
        {
            if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> elements = Saiga::split(line, ' ');
            SAIGA_ASSERT(elements.size() == 8);
            TumRGBDCamera::GroundTruth d;
            d.timeStamp = Saiga::to_double(elements[0]);

            Vec3 t;
            t(0) = Saiga::to_double(elements[1]);
            t(1) = Saiga::to_double(elements[2]);
            t(2) = Saiga::to_double(elements[3]);

            Quat r;
            r.x() = Saiga::to_float(elements[4]);
            r.y() = Saiga::to_float(elements[5]);
            r.z() = Saiga::to_float(elements[6]);
            r.w() = Saiga::to_float(elements[7]);
            r.normalize();

            d.se3 = SE3(r, t);

            data.push_back(d);
        }
    }
    return data;
}



TumRGBDCamera::TumRGBDCamera(const std::string& datasetDir, double depthFactor, int maxFrames, int fps,
                             const std::shared_ptr<DMPP>& dmpp)
    : depthFactor(depthFactor), maxFrames(maxFrames)
{
    this->dmpp = dmpp;
    cout << "Loading TUM RGBD Dataset: " << datasetDir << endl;
    associate(datasetDir);

    load(datasetDir);

    timeStep = std::chrono::duration_cast<tick_t>(std::chrono::duration<double, std::milli>(1000.0 / double(fps)));

    timer.start();
    lastFrameTime = timer.stop();
    nextFrameTime = lastFrameTime + timeStep;
}

TumRGBDCamera::~TumRGBDCamera()
{
    cout << "~TumRGBDCamera" << endl;
}

std::shared_ptr<RGBDCamera::FrameData> TumRGBDCamera::waitForImage()
{
    if (!isOpened())
    //    if(currentId == (int)frames.size())
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

SE3 TumRGBDCamera::getGroundTruth(int frame)
{
    SAIGA_ASSERT(frame >= 0 && frame < (int)tumframes.size());
    GroundTruth gt = tumframes[frame].gt;
    return gt.se3;
}

void TumRGBDCamera::associate(const std::string& datasetDir)
{
    std::vector<CameraData> rgbData   = readCameraData(datasetDir + "/rgb.txt");
    std::vector<CameraData> depthData = readCameraData(datasetDir + "/depth.txt");
    std::vector<GroundTruth> gt       = readGT(datasetDir + "/groundtruth.txt");

    // Find for each rgb image the best depth and gt
    int cdepth = 0;
    int cgt    = 0;

    int lastBest = -1;
    for (auto r : rgbData)
    {
        TumFrame tf;

        auto t = r.timestamp;
        int ibest;

        {
            int ismaller = cdepth;
            int ibigger  = cdepth;

            auto smaller = depthData[ismaller].timestamp;
            auto bigger  = smaller;

            while (bigger < t && cdepth + 1 < (int)depthData.size())
            {
                smaller  = bigger;
                ismaller = ibigger;

                cdepth++;
                ibigger = cdepth;
                bigger  = depthData[ibigger].timestamp;
            }

            ibest = t - smaller < bigger - t ? ismaller : ibigger;

            tf.rgb   = r;
            tf.depth = depthData[ibest];
        }

#if 0
        int igtbest;
#endif
        GroundTruth bestGT;
        {
            int ismaller = cgt;
            int ibigger  = cgt;

            auto smaller = gt[ismaller].timeStamp;
            auto bigger  = smaller;

            while (bigger < t && cgt + 1 < (int)gt.size())
            {
                smaller  = bigger;
                ismaller = ibigger;

                cgt++;
                ibigger = cgt;
                bigger  = gt[ibigger].timeStamp;
            }


#if 1
            GroundTruth a    = gt[ismaller];
            GroundTruth b    = gt[ibigger];
            bestGT.timeStamp = t;
            double alpha     = (t - a.timeStamp) / (b.timeStamp - a.timeStamp);
            bestGT.se3       = slerp(a.se3, b.se3, alpha);
#else
            igtbest = t - smaller < bigger - t ? ismaller : ibigger;
            bestGT  = gt[igtbest];
#endif
        }


        tf.rgb   = r;
        tf.depth = depthData[ibest];
        //        tf.gt = gt[igtbest];
        tf.gt = bestGT;


#if 0
        {
            // Debug check
            int ibest2 = 0;
            double diffbest = 2135388888787;
            for(int i = 0; i <depthData.size(); ++i)
            {
                double diff = glm::abs(t - depthData[i].timestamp);
                if(diff < diffbest)
                {
                    ibest2 = i;
                    diffbest = diff;
                }
            }
            SAIGA_ASSERT(ibest == ibest2);
        }
#endif



        if (ibest != lastBest)
        {
            tumframes.push_back(tf);
        }
        lastBest = ibest;
    }
}

void TumRGBDCamera::load(const std::string& datasetDir)
{
    if (maxFrames >= 0)
    {
        tumframes.resize(std::min((size_t)maxFrames, tumframes.size()));
    }


    frames.resize(tumframes.size());

#pragma omp parallel for
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
}



}  // namespace Saiga
