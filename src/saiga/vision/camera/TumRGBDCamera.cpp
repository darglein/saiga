/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "TumRGBDCamera.h"

#include "saiga/core/util/FileSystem.h"
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



TumRGBDCamera::TumRGBDCamera(const DatasetParameters& _params, int freiburg)
    : DatasetCameraBase<RGBDFrameData>(_params), freiburg(freiburg)
{
    Load();
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

void TumRGBDCamera::LoadImageData(RGBDFrameData& data)
{
    Image cimg(data.file);
    Image dimg(data.depth_file);
    if (cimg.type == UC3)
    {
        // convert to rgba
        ImageTransformation::addAlphaChannel(cimg.getImageView<ucvec3>(), data.colorImg);
    }
    else if (cimg.type == UC4)
    {
        cimg.getImageView<ucvec4>().copyTo(data.colorImg.getImageView());
    }
    else
    {
        SAIGA_EXIT_ERROR("invalid image type");
    }

    if (dimg.type == US1)
    {
        dimg.getImageView<unsigned short>().copyTo(data.depthImg.getImageView(), 1.0 / intrinsics().depthFactor);
    }
    else
    {
        SAIGA_EXIT_ERROR("invalid image type");
    }
}

int TumRGBDCamera::LoadMetaData()
{
    std::cout << "Loading TUM RGBD Dataset: " << params.dir << std::endl;

    if (freiburg == -1)
    {
        std::filesystem::path p(params.dir + "/");
        std::string dir = p.parent_path().filename().string();
        auto pos        = dir.find("freiburg");

        if (pos < dir.size())
        {
            std::string number = dir.substr(pos + 8, 1);
            freiburg           = std::atoi(number.c_str());
        }
    }

    _intrinsics.fps         = 30;
    _intrinsics.depthFactor = 5000;

    _intrinsics.imageSize.width       = 640;
    _intrinsics.imageSize.height      = 480;
    _intrinsics.depthImageSize.width  = 640;
    _intrinsics.depthImageSize.height = 480;
    _intrinsics.bf                    = 40;

    if (freiburg == 1)
    {
        _intrinsics.model.K   = Intrinsics4(517.306408, 516.469215, 318.643040, 255.313989);
		_intrinsics.model.dis  << 0.262383, -0.953104, -0.005358, 0.002628, 1.163314;
    }
    else if (freiburg == 2)
    {
        _intrinsics.model.K   = Intrinsics4(520.908620, 521.007327, 325.141442, 249.701764);
		_intrinsics.model.dis << 0.231222, -0.784899, -0.003257, -0.000105, 0.917205;
    }
    else if (freiburg == 3)
    {
        _intrinsics.model.K   = Intrinsics4(535.4, 539.2, 320.1, 247.6);
		_intrinsics.model.dis << 0, 0, 0, 0, 0;
    }
    else
    {
        SAIGA_EXIT_ERROR("Invalid Freiburg");
    }

    _intrinsics.depthModel = _intrinsics.model;

    std::cout << "Loaded Freiburg " << freiburg << " params:" << std::endl;
    std::cout << _intrinsics << std::endl;

    associate(params.dir);
    load(params.dir, params.multiThreadedLoad);
    return frames.size();
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


    {
        // == Imu Data ==
        // Format:
        //  timestamp ax ay az
        auto sensorFile = params.dir + "/" + "accelerometer.txt";
        auto lines      = File::loadFileStringArray(sensorFile);
        StringViewParser csvParser(", ");

        for (auto&& l : lines)
        {
            if (l.empty()) continue;
            if (l[0] == '#') continue;
            csvParser.set(l);

            auto svTime = csvParser.next();
            if (svTime.empty()) continue;
            auto time = to_double(svTime);


            Vec3 acceleration;
            for (int i = 0; i < 3; ++i)
            {
                auto sv = csvParser.next();
                SAIGA_ASSERT(!sv.empty());
                acceleration(i) = to_double(sv);
            }
            imuData.emplace_back(Vec3(0, 0, 0), acceleration, time);
        }
    }
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
        //        SyncedConsoleProgressBar loadingBar(std::cout, "Loading " + to_string(N) + " images ", N);
        //#pragma omp parallel for if (params.multiThreadedLoad)
        for (int i = 0; i < N; ++i)
        {
            TumFrame d = tumframes[i];


            RGBDFrameData& f = frames[i];
            //            makeFrameData(f);

            f.id = i;
            f.colorImg.create(intrinsics().imageSize.h, intrinsics().imageSize.w);
            f.depthImg.create(intrinsics().depthImageSize.h, intrinsics().depthImageSize.w);
            f.timeStamp  = d.rgb.timestamp;
            f.file       = datasetDir + "/" + d.rgb.img;
            f.depth_file = datasetDir + "/" + d.depth.img;


            if (d.gt.timestamp != -1)
            {
                f.groundTruth = d.gt.se3;
            }

            //            loadingBar.addProgress(1);
        }
    }
    //    computeImuDataPerFrame();
    VLOG(1) << "Loaded " << tumframes.size() << " images.";
}



}  // namespace Saiga
