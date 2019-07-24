/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "TumRGBDCamera.h"

#include "saiga/core/util/ProgressBar.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/tostring.h"

#include <algorithm>
#include <fstream>
#include <thread>
namespace Saiga
{
static AlignedVector<TumRGBDCamera::CameraData> readCameraData(std::string file)
{
    AlignedVector<TumRGBDCamera::CameraData> data;
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
    std::sort(data.begin(), data.end());
    return data;
}

static AlignedVector<TumRGBDCamera::GroundTruth> readGT(std::string file)
{
    AlignedVector<TumRGBDCamera::GroundTruth> data;
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
            d.timestamp = Saiga::to_double(elements[0]);

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
    std::sort(data.begin(), data.end());
    return data;
}



TumRGBDCamera::TumRGBDCamera(const std::string& datasetDir, const RGBDIntrinsics& intr, bool multithreaded)
    : RGBDCamera(intr)
{
    VLOG(1) << "Loading TUM RGBD Dataset: " << datasetDir;
    associate(datasetDir);
    //    associateFromFile(datasetDir + "/associations.txt");

    load(datasetDir, multithreaded);

    timeStep = std::chrono::duration_cast<tick_t>(
        std::chrono::duration<double, std::micro>(1000000.0 / double(intrinsics().fps)));

    timer.start();
    lastFrameTime = timer.stop();
    nextFrameTime = lastFrameTime + timeStep;
}

TumRGBDCamera::~TumRGBDCamera() {}

bool TumRGBDCamera::getImageSync(RGBDFrameData& data)
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

SE3 TumRGBDCamera::getGroundTruth(int frame)
{
    SAIGA_ASSERT(frame >= 0 && frame < (int)tumframes.size());
    GroundTruth gt = tumframes[frame].gt;
    return gt.se3;
}

void TumRGBDCamera::saveRaw(const std::string& dir)
{
    std::cout << "Saving TUM dataset as Saiga-Raw dataset in " << dir << std::endl;
#pragma omp parallel for
    for (int i = 0; i < (int)frames.size(); ++i)
    {
        auto str  = Saiga::leadingZeroString(i, 5);
        auto& tmp = frames[i];
        tmp.colorImg.save(std::string(dir) + str + ".png");
        tmp.depthImg.save(std::string(dir) + str + ".saigai");
    }
    std::cout << "... Done saving the raw dataset." << std::endl;
}


void TumRGBDCamera::associate(const std::string& datasetDir)
{
    AlignedVector<CameraData> rgbData   = readCameraData(datasetDir + "/rgb.txt");
    AlignedVector<CameraData> depthData = readCameraData(datasetDir + "/depth.txt");
    AlignedVector<GroundTruth> gt       = readGT(datasetDir + "/groundtruth.txt");

    for (auto&& r : rgbData)
    {
        TumFrame tf;
        tf.rgb = r;
        auto t = r.timestamp;

        {
            // find best depth image
            auto depthIt = std::lower_bound(depthData.begin(), depthData.end(), CameraData{r.timestamp, ""});
            if (depthIt == depthData.end() || depthIt == depthData.begin()) continue;
            auto prevDepthIt = depthIt--;
            auto bestDepth = std::abs(r.timestamp - depthIt->timestamp) < std::abs(r.timestamp - prevDepthIt->timestamp)
                                 ? depthIt
                                 : prevDepthIt;
            tf.depth = *bestDepth;
        }
        {
            // find best gt
            auto gtIt = std::lower_bound(gt.begin(), gt.end(), GroundTruth{r.timestamp, {}});
            if (gtIt == gt.end() || gtIt == gt.begin()) continue;
            auto prevGTIt = gtIt--;


#if 1
            // interpolate
            double alpha = (t - prevGTIt->timestamp) / (gtIt->timestamp - prevGTIt->timestamp);
            if (prevGTIt->timestamp == gtIt->timestamp) alpha = 0;
            tf.gt.se3       = slerp(prevGTIt->se3, gtIt->se3, alpha);
            tf.gt.timestamp = t;
#else
            // nearest neighbor
            auto bestGt =
                std::abs(r.timestamp - gtIt->timestamp) < std::abs(r.timestamp - prevGTIt->timestamp) ? gtIt : prevGTIt;
            tf.gt = *bestGt;
#endif
        }
        tumframes.push_back(tf);
    }
}

void TumRGBDCamera::associateFromFile(const std::string& datasetDir)
{
    auto lines = File::loadFileStringArray(datasetDir);
    SAIGA_ASSERT(lines.size() > 1);

    for (auto& l : lines)
    {
        TumFrame tf;
        auto v = split(l, ' ');
        if (v.size() != 4) continue;
        //        SAIGA_ASSERT(v.size() == 4);
        tf.rgb.timestamp   = to_double(v[0]);
        tf.rgb.img         = v[1];
        tf.depth.timestamp = to_double(v[2]);
        tf.depth.img       = v[3];
        tumframes.push_back(tf);
    }
    SAIGA_ASSERT(tumframes.size() > 1);
}


void TumRGBDCamera::load(const std::string& datasetDir, bool multithreaded)
{
    if (intrinsics().maxFrames >= 0)
    {
        tumframes.resize(std::min((size_t)intrinsics().maxFrames, tumframes.size()));
    }
    _intrinsics.maxFrames = tumframes.size();


    int N = tumframes.size();
    frames.resize(N);

    {
        SyncedConsoleProgressBar loadingBar(std::cout, "Loading " + to_string(N) + " images ", N);
#pragma omp parallel for if (multithreaded)
        for (int i = 0; i < N; ++i)
        {
            TumFrame d = tumframes[i];
            Image cimg(datasetDir + "/" + d.rgb.img);
            Image dimg(datasetDir + "/" + d.depth.img);

            RGBDFrameData f;
            makeFrameData(f);

            if (cimg.type == UC3)
            {
                // convert to rgba
                ImageTransformation::addAlphaChannel(cimg.getImageView<ucvec3>(), f.colorImg);
            }
            else if (cimg.type == UC4)
            {
                cimg.getImageView<ucvec4>().copyTo(f.colorImg.getImageView());
            }
            else
            {
                SAIGA_EXIT_ERROR("invalid image type");
            }

            if (dimg.type == US1)
            {
                dimg.getImageView<unsigned short>().copyTo(f.depthImg.getImageView(), 1.0 / intrinsics().depthFactor);
            }
            else
            {
                SAIGA_EXIT_ERROR("invalid image type");
            }

            if (d.gt.timestamp != -1)
            {
                f.groundTruth = d.gt.se3;
            }

            frames[i] = std::move(f);
            loadingBar.addProgress(1);
        }
    }
    VLOG(1) << "Loaded " << tumframes.size() << " images.";
}



}  // namespace Saiga
