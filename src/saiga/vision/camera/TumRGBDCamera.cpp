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

#include "TimestampMatcher.h"

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



TumRGBDCamera::TumRGBDCamera(const DatasetParameters& _params, const RGBDIntrinsics& intr)
    : DatasetCameraBase<RGBDFrameData>(_params), _intrinsics(intr)
{
    VLOG(1) << "Loading TUM RGBD Dataset: " << params.dir;

    if (_intrinsics.depthFactor != 5000)
    {
        std::cerr << "Depth Factor should be 5000." << std::endl;
        _intrinsics.depthFactor = 5000;
    }
    associate(params.dir);
    load(params.dir, params.multiThreadedLoad);
}

TumRGBDCamera::~TumRGBDCamera() {}


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



    std::vector<double> rgbTimestamps, depthTimestamps, gtTimestamps;
    for (auto&& r : rgbData) rgbTimestamps.push_back(r.timestamp);
    for (auto&& r : depthData) depthTimestamps.push_back(r.timestamp);
    for (auto&& r : gt) gtTimestamps.push_back(r.timestamp);


    for (auto&& r : rgbData)
    {
        TumFrame tf;
        tf.rgb = r;
        auto t = r.timestamp;

        auto id = TimestampMatcher::findNearestNeighbour(t, depthTimestamps);
        if (id == -1) continue;

        tf.depth = depthData[id];

        auto [id1, id2, alpha] = TimestampMatcher::findLowHighAlphaNeighbour(t, gtTimestamps);
        if (id1 == -1) continue;

        if (id1 != -1)
        {
            tf.gt.se3       = slerp(gt[id1].se3, gt[id2].se3, alpha);
            tf.gt.timestamp = t;
        }

        tumframes.push_back(tf);
    }

    std::cout << "Loaded " << tumframes.size() << std::endl;
}


void TumRGBDCamera::load(const std::string& datasetDir, bool multithreaded)
{
    SAIGA_ASSERT(params.startFrame < tumframes.size());
    tumframes.erase(tumframes.begin(), tumframes.begin() + params.startFrame);

    if (params.maxFrames >= 0)
    {
        tumframes.resize(std::min((size_t)params.maxFrames, tumframes.size()));
    }
    params.maxFrames = tumframes.size();


    int N = tumframes.size();
    frames.resize(N);

    {
        // load the first image to get image sizes
        TumFrame first = tumframes.front();
        Image cimg(datasetDir + "/" + first.rgb.img);
        Image dimg(datasetDir + "/" + first.depth.img);

        if (!(cimg.dimensions() == intrinsics().imageSize))
        {
            std::cerr << "Warning: Intrinsics Image Size does not match actual image size." << std::endl;
            _intrinsics.imageSize = cimg.dimensions();
        }

        if (!(dimg.dimensions() == intrinsics().depthImageSize))
        {
            std::cerr << "Warning: Depth Intrinsics Image Size does not match actual depth image size." << std::endl;
            _intrinsics.depthImageSize = dimg.dimensions();
        }
    }

    {
        SyncedConsoleProgressBar loadingBar(std::cout, "Loading " + to_string(N) + " images ", N);
#pragma omp parallel for if (params.multiThreadedLoad)
        for (int i = 0; i < N; ++i)
        {
            TumFrame d = tumframes[i];
            Image cimg(datasetDir + "/" + d.rgb.img);
            Image dimg(datasetDir + "/" + d.depth.img);

            RGBDFrameData& f = frames[i];
            //            makeFrameData(f);

            f.colorImg.create(intrinsics().imageSize.h, intrinsics().imageSize.w);
            f.depthImg.create(intrinsics().depthImageSize.h, intrinsics().depthImageSize.w);

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

            loadingBar.addProgress(1);
        }
    }
    VLOG(1) << "Loaded " << tumframes.size() << " images.";
}



}  // namespace Saiga
