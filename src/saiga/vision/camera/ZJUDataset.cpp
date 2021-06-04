/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ZJUDataset.h"

#ifdef SAIGA_USE_YAML_CPP

#    include "saiga/core/util/ProgressBar.h"
#    include "saiga/core/util/directory.h"
#    include "saiga/core/util/easylogging++.h"
#    include "saiga/core/util/file.h"
#    include "saiga/core/util/fileChecker.h"
#    include "saiga/core/util/tostring.h"
#    include "saiga/core/util/yaml.h"
#    include "saiga/vision/camera/TimestampMatcher.h"

//#include "IsmarFrameData.h"

#    include <algorithm>
#    include <fstream>
#    include <thread>

namespace Saiga
{
ZJUDataset::ZJUDataset(const DatasetParameters& params) : DatasetCameraBase(params)
{
    camera_type = CameraInputType::Mono;
    Load();
}

void ZJUDataset::LoadImageData(FrameData& data)
{
    SAIGA_ASSERT(data.image.rows == 0);

    Image cimg(data.image_file);
    if (cimg.type == UC1)
    {
        //                frames[i] = std::move(cimg);
        GrayImageType imgGray(cimg.h, cimg.w);
        cimg.getImageView<unsigned char>().copyTo(imgGray.getImageView());
        data.image = imgGray;
    }
    else if (cimg.type == UC3 || cimg.type == UC3)
    {
        // this is currently only the case for "black frames"
        GrayImageType imgGray(cimg.h, cimg.w);
        data.image = imgGray;
    }
    else
    {
        SAIGA_EXIT_ERROR("Unknown image type");
    }

    SAIGA_ASSERT(cimg.w == intrinsics.imageSize.w);
}

int ZJUDataset::LoadMetaData()
{
    std::cout << "Loading ZJUDataset: " << params.dir << std::endl;



    YAML::Node config = YAML::LoadFile(params.dir + "/camera/sensor.yaml");


    auto cameraModel = config["camera_model"].as<std::string>();
    SAIGA_ASSERT(cameraModel == "pinhole");



    YAML::Node cameraNode = config["intrinsic"]["camera"];
    std::vector<double> cameraParams;
    for (auto n : cameraNode)
    {
        cameraParams.push_back(n.as<double>());
    }
    SAIGA_ASSERT(cameraParams.size() == 4);


    intrinsics.imageSize.w = 640;
    intrinsics.imageSize.h = 480;

    intrinsics.model.K.fx = cameraParams[0];
    intrinsics.model.K.fy = cameraParams[1];
    intrinsics.model.K.cx = cameraParams[2];
    intrinsics.model.K.cy = cameraParams[3];
    intrinsics.fps        = config["frequency"].as<double>();

    std::cout << intrinsics << std::endl;


    YAML::Node extrinsicsNode = config["extrinsic"];
    std::vector<double> extrQ;
    for (auto n : extrinsicsNode["q"])
    {
        extrQ.push_back(n.as<double>());
    }
    SAIGA_ASSERT(extrQ.size() == 4);

    std::vector<double> extrT;
    for (auto n : extrinsicsNode["p"])
    {
        extrT.push_back(n.as<double>());
    }
    SAIGA_ASSERT(extrT.size() == 3);

    Quat q;
    q.x() = extrQ[0];
    q.y() = extrQ[1];
    q.z() = extrQ[2];
    q.w() = extrQ[3];

    Vec3 t;
    t.x() = extrT[0];
    t.y() = extrT[1];
    t.z() = extrT[2];

    groundTruthToCamera       = SE3(q, t);
    intrinsics.camera_to_body = groundTruthToCamera.inverse();
    intrinsics.camera_to_gt   = intrinsics.camera_to_body;
    //    std::cout << "Extrinsics: " << groundTruthToCamera << std::endl;


    {
        // == IMU ==
        // Load camera meta data
        auto imuSensor    = params.dir + "/imu/sensor.yaml";
        YAML::Node config = YAML::LoadFile(imuSensor);

        Vec4 q = readYamlMatrix<Vec4>(config["extrinsic"]["q"]);
        Vec3 p = readYamlMatrix<Vec3>(config["extrinsic"]["p"]);

        Quat qua;
        qua.x() = q(0);
        qua.y() = q(1);
        qua.z() = q(2);
        qua.w() = q(3);

        imu                 = Imu::Sensor();
        imu->sensor_to_body = SE3(qua, p);

        imu->frequency                = config["frequency"].as<double>();
        imu->frequency_sqrt           = sqrt(imu->frequency);
        imu->omega_sigma              = config["intrinsic"]["sigma_w"].as<double>();
        imu->omega_random_walk        = config["intrinsic"]["sigma_bw"].as<double>();
        imu->acceleration_sigma       = config["intrinsic"]["sigma_a"].as<double>();
        imu->acceleration_random_walk = config["intrinsic"]["sigma_ba"].as<double>();

        std::cout << *imu << std::endl;
    }


    associate(params.dir);
    load(params.dir, params.multiThreadedLoad);
    return frames.size();
}


void ZJUDataset::associate(const std::string& datasetDir)
{
    // timestamp - filename
    std::vector<std::pair<double, std::string>> images;
    {
        auto lines = File::loadFileStringArray(datasetDir + "/" + "camera/data.csv");
        StringViewParser csvParser(", ");


        for (auto&& l : lines)
        {
            if (l.empty()) continue;
            if (l[0] == '#') continue;

            csvParser.set(l);

            auto svTime = csvParser.next();
            if (svTime.empty()) continue;
            auto svImg = csvParser.next();
            if (svImg.empty()) continue;

            images.emplace_back(to_double(svTime), svImg);
        }
        std::sort(images.begin(), images.end());
    }

    std::vector<std::pair<double, SE3>> gt;
    std::vector<double> gtTimes;
    {
        auto lines = File::loadFileStringArray(datasetDir + "/" + "groundtruth/data.csv");
        StringViewParser csvParser(", ");
        for (auto&& l : lines)
        {
            if (l.empty()) continue;
            if (l[0] == '#') continue;
            csvParser.set(l);

            auto svTime = csvParser.next();
            if (svTime.empty()) continue;

            Eigen::Matrix<double, 7, 1> data;
            for (int i = 0; i < 7; ++i)
            {
                auto sv = csvParser.next();
                SAIGA_ASSERT(!sv.empty());
                data(i) = to_double(sv);
            }

            Quat q;
            q.coeffs() = data.segment<4>(0);
            Vec3 t     = data.segment<3>(4);

            SE3 value(q, t);
            //        value     = value * groundTruthToCamera;
            auto time = to_double(svTime);
            gtTimes.push_back(time);
            gt.emplace_back(time, value);
        }
        std::sort(gt.begin(), gt.end(), [](auto a, auto b) { return a.first < b.first; });
        std::cout << "Found " << images.size() << " images and " << gt.size() << " ground truth meassurements."
                  << std::endl;
    }


    {
        // == Imu Data ==
        // Format:
        //   timestamp [s],
        //   w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],
        //   a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
        //
        // Note: Same format as Euroc. Only difference is that the timestamp is given in seconds
        auto sensorFile = params.dir + "/" + "imu/data.csv";
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

            Vec3 omega;
            for (int i = 0; i < 3; ++i)
            {
                auto sv = csvParser.next();
                SAIGA_ASSERT(!sv.empty());
                omega(i) = to_double(sv);
            }

            Vec3 acceleration;
            for (int i = 0; i < 3; ++i)
            {
                auto sv = csvParser.next();
                SAIGA_ASSERT(!sv.empty());
                acceleration(i) = to_double(sv);
            }
            imuData.emplace_back(omega, acceleration, time);
        }
    }


    if (params.normalize_timestamps)
    {
        double first_time = images.front().first;

        for (auto& i : images) i.first -= first_time;
        for (auto& i : gt) i.first -= first_time;
        for (auto& i : gtTimes) i -= first_time;
        for (auto& i : imuData) i.timestamp -= first_time;
    }

    for (auto&& r : images)
    {
        IsmarFrame tf;
        tf.image     = r.second;
        double t     = r.first;
        tf.timestamp = t;

#    if 1
        auto [id1, id2, alpha] = TimestampMatcher::findLowHighAlphaNeighbour(t, gtTimes);
        if (id1 != -1)
        {
            tf.gt = slerp(gt[id1].second, gt[id2].second, alpha);
        }
#    else
        auto id1 = TimestampMatcher::findNearestNeighbour(t, gtTimes);
        if (id1 != -1)
        {
            tf.gt = gt[id1].second;
        }
#    endif
        framesRaw.push_back(tf);
    }
}



void ZJUDataset::load(const std::string& datasetDir, bool multithreaded)
{
    SAIGA_ASSERT(params.startFrame < (int)framesRaw.size());
    framesRaw.erase(framesRaw.begin(), framesRaw.begin() + params.startFrame);

    if (params.maxFrames >= 0)
    {
        framesRaw.resize(std::min((size_t)params.maxFrames, framesRaw.size()));
    }
    params.maxFrames = framesRaw.size();


    int N = framesRaw.size();
    frames.resize(N);


    std::string imageDir = datasetDir + "/camera/images";



    for (int i = 0; i < N; ++i)
    {
        auto imgStr  = framesRaw[i].image;
        FrameData& f = frames[i];


        f.groundTruth = framesRaw[i].gt;
        f.timeStamp   = framesRaw[i].timestamp;
        f.id          = i;
        f.image_file  = imageDir + "/" + imgStr;
    }
}



}  // namespace Saiga

#endif
