/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "EuRoCDataset.h"

#include "saiga/core/util/ProgressBar.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vision/camera/TimestampMatcher.h"

#include <filesystem>

#ifdef SAIGA_USE_YAML_CPP

#    include "saiga/core/util/easylogging++.h"

#    include "yaml-cpp/yaml.h"
namespace Saiga
{
template <typename MatrixType>
MatrixType readYamlMatrix(const YAML::Node& node)
{
    MatrixType matrix;
    std::vector<double> data;
    for (auto n : node)
    {
        data.push_back(n.as<double>());
    }
    SAIGA_ASSERT(data.size() == (MatrixType::RowsAtCompileTime * MatrixType::ColsAtCompileTime));
    for (int i = 0; i < MatrixType::RowsAtCompileTime; ++i)
    {
        for (int j = 0; j < MatrixType::ColsAtCompileTime; ++j)
        {
            matrix(i, j) = data[i * MatrixType::ColsAtCompileTime + j];
        }
    }
    return matrix;
}

// Reads csv files of the following format:
//
// #timestamp [ns],filename
// 1403636579763555584,1403636579763555584.png
// 1403636579813555456,1403636579813555456.png
// 1403636579863555584,1403636579863555584.png
// ...
std::vector<std::pair<double, std::string>> loadTimestapDataCSV(const std::string& file)
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

        data.emplace_back(to_double(svTime), svImg);
    }
    std::sort(data.begin(), data.end());
    return data;
}

struct Associations
{
    // left and right image id
    int left, right;
    // id into gt array
    // gtlow is the closest timestamp smaller and gthigh is the closest timestamp larger.
    int gtlow, gthigh;
    // the interpolation value between low and high
    double gtAlpha;
};


EuRoCDataset::EuRoCDataset(const DatasetParameters& _params) : DatasetCameraBase<StereoFrameData>(_params)
{
    intrinsics.fps = params.fps;

    VLOG(1) << "Loading EuRoCDataset Stereo Dataset: " << params.dir;

    auto leftImageSensor  = params.dir + "/cam0/sensor.yaml";
    auto rightImageSensor = params.dir + "/cam1/sensor.yaml";

    SAIGA_ASSERT(std::filesystem::exists(leftImageSensor));
    SAIGA_ASSERT(std::filesystem::exists(rightImageSensor));


    {
        // == Cam 0 ==
        // Load camera meta data
        YAML::Node config = YAML::LoadFile(leftImageSensor);
        SAIGA_ASSERT(config);
        SAIGA_ASSERT(!config.IsNull());

        VLOG(1) << config["comment"].as<std::string>();
        SAIGA_ASSERT(config["camera_model"].as<std::string>() == "pinhole");
        intrinsics.model.K.coeffs(readYamlMatrix<Vec4>(config["intrinsics"]));
        auto res               = readYamlMatrix<ivec2>(config["resolution"]);
        intrinsics.imageSize.w = res(0);
        intrinsics.imageSize.h = res(1);
        // 4 parameter rad-tan model
        intrinsics.model.dis.segment<4>(0) = readYamlMatrix<Vec4>(config["distortion_coefficients"]);
        intrinsics.model.dis(4)            = 0;
        Mat4 m                             = readYamlMatrix<Mat4>(config["T_BS"]["data"]);
        extrinsics_cam0                    = SE3::fitToSE3(m);
        cam0_images                        = loadTimestapDataCSV(params.dir + "/cam0/data.csv");
    }

    {
        // == Cam 1 ==
        // Load camera meta data
        YAML::Node config = YAML::LoadFile(rightImageSensor);
        VLOG(1) << config["comment"].as<std::string>();
        SAIGA_ASSERT(config["camera_model"].as<std::string>() == "pinhole");
        intrinsics.rightModel.K.coeffs(readYamlMatrix<Vec4>(config["intrinsics"]));
        auto res                    = readYamlMatrix<ivec2>(config["resolution"]);
        intrinsics.rightImageSize.w = res(0);
        intrinsics.rightImageSize.h = res(1);
        // 4 parameter rad-tan model
        intrinsics.rightModel.dis.segment<4>(0) = readYamlMatrix<Vec4>(config["distortion_coefficients"]);
        intrinsics.rightModel.dis(4)            = 0;
        Mat4 m                                  = readYamlMatrix<Mat4>(config["T_BS"]["data"]);
        extrinsics_cam1                         = SE3::fitToSE3(m);
        cam1_images                             = loadTimestapDataCSV(params.dir + "/cam1/data.csv");
    }

    {
        // == Ground truth position ==

        auto sensorFile = params.dir + "/" + "state_groundtruth_estimate0/data.csv";


        auto lines = File::loadFileStringArray(sensorFile);
        StringViewParser csvParser(", ");
        std::vector<double> gtTimes;
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


            auto time = to_double(svTime);
            gtTimes.push_back(time);
            ground_truth.emplace_back(time, SE3(q, data));
        }

        YAML::Node config = YAML::LoadFile(params.dir + "/state_groundtruth_estimate0/sensor.yaml");
        Mat4 m            = readYamlMatrix<Mat4>(config["T_BS"]["data"]);
        extrinsics_gt     = SE3::fitToSE3(m);

        std::sort(ground_truth.begin(), ground_truth.end(), [](auto a, auto b) { return a.first < b.first; });
    }

    groundTruthToCamera = extrinsics_gt.inverse() * extrinsics_cam0;
    //    groundTruthToCamera = extrinsics_gt * extrinsics_cam0.inverse();

    std::cout << extrinsics_gt << std::endl;
    std::cout << extrinsics_cam0 << std::endl;
    std::cout << extrinsics_cam1 << std::endl;
    std::cout << groundTruthToCamera << std::endl;



    std::cout << "Found " << cam1_images.size() << " images and " << ground_truth.size()
              << " ground truth meassurements." << std::endl;

    SAIGA_ASSERT(intrinsics.imageSize == intrinsics.rightImageSize);
    std::cout << intrinsics << std::endl;



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
            a.left = i;

            a.right                = TimestampMatcher::findNearestNeighbour(left_timestamps[i], right_timestamps);
            auto [id1, id2, alpha] = TimestampMatcher::findLowHighAlphaNeighbour(left_timestamps[i], gt_timestamps);
            a.gtlow                = id1;
            a.gthigh               = id2;
            a.gtAlpha              = alpha;

            if (a.right == -1 || a.gtlow == -1 || a.gthigh == -1) continue;
            assos.push_back(a);
        }
    }

    //    assos.resize(100);
    //    assos.erase(assos.begin(), assos.begin() + 1000);
    //    assos.resize(200);
    // ==== Actual Image Loading ====
    {
        int N = assos.size();


        if (params.maxFrames == -1)
        {
            params.maxFrames = N;
        }

        params.maxFrames = std::min(N - params.startFrame, params.maxFrames);

        frames.resize(params.maxFrames);
        N = params.maxFrames;



        SyncedConsoleProgressBar loadingBar(std::cout, "Loading " + to_string(N) + " images ", N);
#    pragma omp parallel for if (params.multiThreadedLoad)
        for (int i = 0; i < N; ++i)
        {
            auto a      = assos[i];
            auto& frame = frames[i];


            std::string leftFile  = params.dir + "/cam0/data/" + cam0_images[a.left].second;
            std::string rightFile = params.dir + "/cam1/data/" + cam1_images[a.right].second;

            if (a.gtlow >= 0 && a.gthigh >= 0 && a.gtlow != a.gthigh)
            {
                //                Vec3 gtpos =
                //                    (1.0 - a.gtAlpha) * ground_truth[a.gtlow].second + a.gtAlpha *
                //                    ground_truth[a.gthigh].second;
                frame.groundTruth = slerp(ground_truth[a.gtlow].second, ground_truth[a.gthigh].second, a.gtAlpha);

                frame.grayImg.load(leftFile);
                frame.grayImg2.load(rightFile);
            }
            frame.timeStamp = cam0_images[a.left].first / 1e9;
            loadingBar.addProgress(1);
        }
    }
}

}  // namespace Saiga

#endif
