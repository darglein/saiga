/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "EuRoCDataset.h"

#include "saiga/core/util/FileSystem.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vision/camera/TimestampMatcher.h"

#ifdef SAIGA_USE_YAML_CPP

#    include "saiga/core/util/easylogging++.h"
#    include "saiga/core/util/yaml.h"

namespace Saiga
{
// Reads csv files of the following format:
//
// #timestamp [ns],filename
// 1403636579763555584,1403636579763555584.png
// 1403636579813555456,1403636579813555456.png
// 1403636579863555584,1403636579863555584.png
// ...
static std::vector<std::pair<double, std::string>> loadTimestapDataCSV(const std::string& file)
{
    auto lines = File::loadFileStringArray(file);
    File::removeWindowsLineEnding(lines);

    StringViewParser csvParser(", ");

    // timestamp - filename
    std::vector<std::pair<double, std::string>> data;
    for (auto&& l : lines)
    {
        if (l.empty()) continue;
        if (l[0] == '#') continue;

        csvParser.set(l);

        auto svTime = csvParser.next();
        if (svTime.empty()) continue;
        auto svImg = csvParser.next();
        if (svImg.empty()) continue;

        double t = to_double(svTime);
        // The data is given in nano seconds
        t = t / 1e9;
        data.emplace_back(t, svImg);
    }
    std::sort(data.begin(), data.end());
    return data;
}

struct Associations
{
    // left and right image id
    int left, right;
    // id into gt array
    // gtlow is the closest gt index smaller and gthigh is the closest gt index larger.
    int gtlow, gthigh;
    // the interpolation value between low and high
    double gtAlpha;

    double timestamp;
};


EuRoCDataset::EuRoCDataset(const DatasetParameters& _params, Sequence sequence)
    : DatasetCameraBase(_params), sequence(sequence)
{
    camera_type = CameraInputType::Stereo;
    Load();
}

void EuRoCDataset::LoadImageData(FrameData& data)
{
    //    std::cout << "EuRoCDataset::LoadImageData " << data.id << std::endl;
    SAIGA_ASSERT(data.image.rows == 0);
    // Load if it's not loaded already

    data.image.load(data.image_file);
    if (!params.force_monocular)
    {
        data.right_image.load(data.right_image_file);
    }
}

int EuRoCDataset::LoadMetaData()
{
    std::cout << "Loading EuRoCDataset Stereo Dataset: " << params.dir << std::endl;

    auto leftImageSensor  = params.dir + "/cam0/sensor.yaml";
    auto rightImageSensor = params.dir + "/cam1/sensor.yaml";
    auto imuSensor        = params.dir + "/imu0/sensor.yaml";

    SAIGA_ASSERT(std::filesystem::exists(leftImageSensor));
    SAIGA_ASSERT(std::filesystem::exists(rightImageSensor));
    SAIGA_ASSERT(std::filesystem::exists(imuSensor));

    FindSequence();

    {
        // == Cam 0 ==
        // Load camera meta data
        YAML::Node config = YAML::LoadFile(leftImageSensor);
        SAIGA_ASSERT(config);
        SAIGA_ASSERT(!config.IsNull());

        VLOG(1) << config["comment"].as<std::string>();
        SAIGA_ASSERT(config["camera_model"].as<std::string>() == "pinhole");
        intrinsics.fps = config["rate_hz"].as<double>();
        intrinsics.model.K.coeffs(readYamlMatrix<Vec5>(config["intrinsics"]));
        auto res               = readYamlMatrix<ivec2>(config["resolution"]);
        intrinsics.imageSize.w = res(0);
        intrinsics.imageSize.h = res(1);
        // 4 parameter rad-tan model
        Vec4 d                  = readYamlMatrix<Vec4>(config["distortion_coefficients"]);
        intrinsics.model.dis.k1 = d(0);
        intrinsics.model.dis.k2 = d(1);
        intrinsics.model.dis.p1 = d(2);
        intrinsics.model.dis.p2 = d(3);
        Mat4 m                  = readYamlMatrix<Mat4>(config["T_BS"]["data"]);
        extrinsics_cam0         = SE3::fitToSE3(m);
        cam0_images             = loadTimestapDataCSV(params.dir + "/cam0/data.csv");
    }

    {
        // == Cam 1 ==
        // Load camera meta data
        YAML::Node config = YAML::LoadFile(rightImageSensor);
        VLOG(1) << config["comment"].as<std::string>();
        SAIGA_ASSERT(config["camera_model"].as<std::string>() == "pinhole");
        intrinsics.rightModel.K.coeffs(readYamlMatrix<Vec5>(config["intrinsics"]));
        auto res                    = readYamlMatrix<ivec2>(config["resolution"]);
        intrinsics.rightImageSize.w = res(0);
        intrinsics.rightImageSize.h = res(1);
        // 4 parameter rad-tan model
        Vec4 d                       = readYamlMatrix<Vec4>(config["distortion_coefficients"]);
        intrinsics.rightModel.dis.k1 = d(0);
        intrinsics.rightModel.dis.k2 = d(1);
        intrinsics.rightModel.dis.p1 = d(2);
        intrinsics.rightModel.dis.p2 = d(3);
        Mat4 m                       = readYamlMatrix<Mat4>(config["T_BS"]["data"]);
        extrinsics_cam1              = SE3::fitToSE3(m);
        cam1_images                  = loadTimestapDataCSV(params.dir + "/cam1/data.csv");
    }

    {
        // == IMU ==
        // Load camera meta data
        YAML::Node config = YAML::LoadFile(imuSensor);
        VLOG(1) << config["comment"].as<std::string>();
        Mat4 m = readYamlMatrix<Mat4>(config["T_BS"]["data"]);


        imu                 = Imu::Sensor();
        imu->sensor_to_body = SE3::fitToSE3(m);

        imu->frequency                = config["rate_hz"].as<double>();
        imu->frequency_sqrt           = sqrt(imu->frequency);
        imu->omega_sigma              = config["gyroscope_noise_density"].as<double>();
        imu->omega_random_walk        = config["gyroscope_random_walk"].as<double>();
        imu->acceleration_sigma       = config["accelerometer_noise_density"].as<double>();
        imu->acceleration_random_walk = config["accelerometer_random_walk"].as<double>();

        VLOG(1) << *imu;
    }



    auto vicon0_file = params.dir + "/" + "vicon0/sensor.yaml";
    auto leica0_file = params.dir + "/" + "leica0/sensor.yaml";

    if (use_raw_gt_data && std::filesystem::exists(vicon0_file))
    {
        YAML::Node config = YAML::LoadFile(vicon0_file);
        Mat4 m            = readYamlMatrix<Mat4>(config["T_BS"]["data"]);
        extrinsics_gt     = SE3::fitToSE3(m);


        auto data_file = params.dir + "/" + "vicon0/data.csv";

        auto lines = File::loadFileStringArray(data_file);
        StringViewParser csvParser(", ");
        for (auto&& l : lines)
        {
            if (l.empty()) continue;
            if (l[0] == '#') continue;
            csvParser.set(l);

            auto svTime = csvParser.next();
            if (svTime.empty()) continue;

            Vec3 data;
            for (int i = 0; i < 3; ++i)
            {
                auto sv = csvParser.next();
                SAIGA_ASSERT(!sv.empty());
                data(i) = to_double(sv);
            }

            Vec4 dataq;
            for (int i = 0; i < 4; ++i)
            {
                auto sv = csvParser.next();
                SAIGA_ASSERT(!sv.empty());
                dataq(i) = to_double(sv);
            }

            Quat q;
            q.x() = dataq(0);
            q.y() = dataq(1);
            q.z() = dataq(2);
            q.w() = dataq(3);


            auto time = to_double(svTime) / 1e9;

            time += params.ground_truth_time_offset;
            ground_truth.emplace_back(time, SE3(q, data));
        }
    }
    else if (use_raw_gt_data && std::filesystem::exists(leica0_file))
    {
        YAML::Node config = YAML::LoadFile(leica0_file);
        Mat4 m            = readYamlMatrix<Mat4>(config["T_BS"]["data"]);
        extrinsics_gt     = SE3::fitToSE3(m);

        auto data_file = params.dir + "/" + "leica0/data.csv";

        auto lines = File::loadFileStringArray(data_file);
        StringViewParser csvParser(", ");
        for (auto&& l : lines)
        {
            if (l.empty()) continue;
            if (l[0] == '#') continue;
            csvParser.set(l);

            auto svTime = csvParser.next();
            if (svTime.empty()) continue;

            Vec3 data;
            for (int i = 0; i < 3; ++i)
            {
                auto sv = csvParser.next();
                SAIGA_ASSERT(!sv.empty());
                data(i) = to_double(sv);
            }
            auto time = to_double(svTime) / 1e9;

            time += params.ground_truth_time_offset;
            ground_truth.emplace_back(time, SE3(Quat::Identity(), data));

            //            std::cout << "gt " << ground_truth.back().second << std::endl;
        }
        // extrinsics_gt = SE3();
        std::cout << "extr " << extrinsics_gt << std::endl;
    }
    else
    {
        //        SAIGA_EXIT_ERROR("no gt file");
        // == Ground truth position ==
        auto sensorFile = params.dir + "/" + "state_groundtruth_estimate0/data.csv";


        auto lines = File::loadFileStringArray(sensorFile);
        StringViewParser csvParser(", ");
        for (auto&& l : lines)
        {
            if (l.empty()) continue;
            if (l[0] == '#') continue;
            csvParser.set(l);

            auto svTime = csvParser.next();
            if (svTime.empty()) continue;

            Vec3 data;
            for (int i = 0; i < 3; ++i)
            {
                auto sv = csvParser.next();
                SAIGA_ASSERT(!sv.empty());
                data(i) = to_double(sv);
            }

            Vec4 dataq;
            for (int i = 0; i < 4; ++i)
            {
                auto sv = csvParser.next();
                SAIGA_ASSERT(!sv.empty());
                dataq(i) = to_double(sv);
            }

            Quat q;
            q.x() = dataq(0);
            q.y() = dataq(1);
            q.z() = dataq(2);
            q.w() = dataq(3);


            auto time = to_double(svTime) / 1e9;

            time += params.ground_truth_time_offset;
            ground_truth.emplace_back(time, SE3(q, data));
        }

        YAML::Node config = YAML::LoadFile(params.dir + "/state_groundtruth_estimate0/sensor.yaml");
        Mat4 m            = readYamlMatrix<Mat4>(config["T_BS"]["data"]);
        extrinsics_gt     = SE3::fitToSE3(m);
    }


    std::sort(ground_truth.begin(), ground_truth.end(), [](auto a, auto b) { return a.first < b.first; });

    // SE3 groundTruthToCamera = extrinsics_gt.inverse() * extrinsics_cam0;
    intrinsics.camera_to_gt   = extrinsics_cam0.inverse() * extrinsics_gt;
    intrinsics.camera_to_body = extrinsics_cam0.inverse();

    std::cout << "Camera -> GT  : " << intrinsics.camera_to_gt << std::endl;
    std::cout << "Camera -> Body: " << intrinsics.camera_to_body << std::endl;



    intrinsics.left_to_right = extrinsics_cam1.inverse() * extrinsics_cam0;
    intrinsics.maxDepth      = 35;
    intrinsics.bf            = intrinsics.left_to_right.translation().norm() * intrinsics.model.K.fx;
    //    std::cout << "Left->Right: " << intrinsics.left_to_right << std::endl;

    {
        // == Imu Data ==
        // Format:
        //   timestamp [ns],
        //   w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],
        //   a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
        auto sensorFile = params.dir + "/" + "imu0/data.csv";
        auto lines      = File::loadFileStringArray(sensorFile);
        StringViewParser csvParser(", ");

        for (auto&& l : lines)
        {
            if (l.empty()) continue;
            if (l[0] == '#') continue;
            csvParser.set(l);

            auto svTime = csvParser.next();
            if (svTime.empty()) continue;
            // time is given in nano seconds
            auto time = to_double(svTime) / 1e9;

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

    //    std::cout << std::setprecision(20);
    //    std::cout << "First image at " << cam0_images.front().first << " " << cam1_images.front().first << " First IMU
    //    at "
    //              << imuData.front().timestamp << " First GT at " << ground_truth.front().first << std::endl;


    std::cout << "Found " << cam1_images.size() << " images and " << ground_truth.size()
              << " ground truth meassurements and " << imuData.size() << " IMU meassurements." << std::endl;

    SAIGA_ASSERT(intrinsics.imageSize == intrinsics.rightImageSize);
    VLOG(1) << intrinsics;



    if (params.normalize_timestamps)
    {
        double first_time = cam0_images.front().first;

        for (auto& i : cam0_images) i.first -= first_time;
        for (auto& i : cam1_images) i.first -= first_time;
        for (auto& i : ground_truth) i.first -= first_time;
        for (auto& i : imuData) i.timestamp -= first_time;
    }


    std::vector<Associations> assos;
    // =========== Associate ============
    {
        // extract timestamps so the association matcher works
        std::vector<double> left_timestamps, right_timestamps;
        std::vector<double> gt_timestamps;

        for (auto i : cam0_images) left_timestamps.push_back(i.first);
        for (auto i : cam1_images) right_timestamps.push_back(i.first);
        for (auto i : ground_truth) gt_timestamps.push_back(i.first);


        for (int i = 0; i < cam0_images.size(); ++i)
        {
            Associations a;
            a.left                 = i;
            a.timestamp            = left_timestamps[i];
            a.right                = TimestampMatcher::findNearestNeighbour(left_timestamps[i], right_timestamps);
            auto [id1, id2, alpha] = TimestampMatcher::findLowHighAlphaNeighbour(left_timestamps[i], gt_timestamps);
            a.gtlow                = id1;
            a.gthigh               = id2;
            a.gtAlpha              = alpha;



            if (a.right == -1 || a.gtlow == -1 || a.gthigh == -1)
            {
                //                std::cout << i << " " << a.right << " " << a.gtlow << " " << a.gthigh << std::endl;
                continue;
            }

#    if 0
            double max_time_diff =
                std::max(std::abs(gt_timestamps[id1] - a.timestamp), std::abs(gt_timestamps[id2] - a.timestamp));

            if (max_time_diff > 0.5)
            {
                a.gtlow  = -1;
                a.gthigh = -1;
            }
#    endif



            assos.push_back(a);
        }
    }
    //    std::cout << "assos: " << assos.size() << std::endl;



    // ==== Actual Image Loading ====
    {
        SAIGA_ASSERT(params.startFrame < (int)assos.size());
        assos.erase(assos.begin(), assos.begin() + params.startFrame);

        if (params.maxFrames >= 0)
        {
            assos.resize(std::min((size_t)params.maxFrames, assos.size()));
        }
        params.maxFrames = assos.size();


        int N = assos.size();
        frames.resize(N);

        for (int i = 0; i < N; ++i)
        {
            auto a      = assos[i];
            auto& frame = frames[i];
            frame.id    = i;
            if (a.gtlow >= 0 && a.gthigh >= 0 && a.gtlow != a.gthigh)
            {
                frame.groundTruth = slerp(ground_truth[a.gtlow].second, ground_truth[a.gthigh].second, a.gtAlpha);
            }
            else
            {
                // std::cout << "no gt for frame " << i << std::endl;
            }
            frame.timeStamp = cam0_images[a.left].first;



            frame.image_file       = params.dir + "/cam0/data/" + cam0_images[a.left].second;
            frame.right_image_file = params.dir + "/cam1/data/" + cam1_images[a.right].second;
        }
        return N;
    }
}

void EuRoCDataset::FindSequence()
{
    if (sequence == UNKNOWN)
    {
        // Try to extract sequence from file
        std::filesystem::path p(params.dir + "/");
        std::string dir = p.parent_path().filename().string();

        int found = 0;
        if (dir.find("mav0") < dir.size())
        {
            dir = p.parent_path().parent_path().filename().string();
        }

        if (dir.find("MH") < dir.size())
        {
            if (dir.find("01") < dir.size())
            {
                sequence = MH_01;
                found++;
            }
            if (dir.find("02") < dir.size())
            {
                sequence = MH_02;
                found++;
            }
            if (dir.find("03") < dir.size())
            {
                sequence = MH_03;
                found++;
            }
            if (dir.find("04") < dir.size())
            {
                sequence = MH_04;
                found++;
            }
            if (dir.find("05") < dir.size())
            {
                sequence = MH_05;
                found++;
            }
        }

        if (dir.find("V1") < dir.size())
        {
            if (dir.find("01") < dir.size())
            {
                sequence = V1_01;
                found++;
            }
            if (dir.find("02") < dir.size())
            {
                sequence = V1_02;
                found++;
            }
            if (dir.find("03") < dir.size())
            {
                sequence = V1_03;
                found++;
            }
        }

        if (dir.find("V2") < dir.size())
        {
            if (dir.find("01") < dir.size())
            {
                sequence = V2_01;
                found++;
            }
            if (dir.find("02") < dir.size())
            {
                sequence = V2_02;
                found++;
            }
            if (dir.find("03") < dir.size())
            {
                sequence = V2_03;
                found++;
            }
        }

        SAIGA_ASSERT(found == 1);
        SAIGA_ASSERT(sequence != UNKNOWN);

        std::cout << "Extracted Sequence from file name: " << DatasetNames()[int(sequence)] << std::endl;
    }
    else
    {
        std::cout << "Using sequence parameter: " << DatasetNames()[int(sequence)] << std::endl;
    }



    // These values were computed with snake-slam by temporal alignment of the trajectories.
    const std::pair<double, bool> offsets[] = {// MH
                                               {0.006, false},
                                               {0.011, false},
                                               {0.012, false},
                                               {0.029, false},
                                               {0.032, false},
                                               // V1
                                               {0.010, true},
                                               {0.013, true},
                                               {0.042, true},
                                               // V2
                                               {-0.217, true},
                                               {-0.209, true},
                                               {-0.200, true}};



    params.ground_truth_time_offset = -offsets[int(sequence)].first;
    use_raw_gt_data                 = offsets[int(sequence)].second;

    std::cout << "Use Raw GT data: " << use_raw_gt_data << std::endl;
    std::cout << "Ground truth offset: " << params.ground_truth_time_offset << std::endl;
}



}  // namespace Saiga

#endif
