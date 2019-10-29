/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "KittiDataset.h"

#include "saiga/core/util/ProgressBar.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vision/camera/TimestampMatcher.h"

#include <filesystem>

namespace Saiga
{
KittiDataset::KittiDataset(const DatasetParameters& params_) : DatasetCameraBase<StereoFrameData>(params_)
{
    // Kitti was recorded with 10 fps
    intrinsics.fps = 10;

    VLOG(1) << "Loading KittiDataset Stereo Dataset: " << params.dir;

    auto leftImageDir    = params.dir + "/image_0";
    auto rightImageDir   = params.dir + "/image_1";
    auto calibFile       = params.dir + "/calib.txt";
    auto timesFile       = params.dir + "/times.txt";
    auto groundtruthFile = params.groundTruth;

    SAIGA_ASSERT(std::filesystem::exists(leftImageDir));
    SAIGA_ASSERT(std::filesystem::exists(rightImageDir));
    SAIGA_ASSERT(std::filesystem::exists(calibFile));
    SAIGA_ASSERT(std::filesystem::exists(timesFile));

    {
        // load calibration matrices
        // They are stored like this:
        // P0: a00, a01, a02, ...
        auto lines = File::loadFileStringArray(calibFile);

        std::vector<Eigen::Matrix<double, 3, 4>> matrices;

        StringViewParser parser(" ");
        for (auto l : lines)
        {
            if (l.empty()) continue;
            parser.set(l);

            // parse the "P0"
            parser.next();

            Eigen::Matrix<double, 3, 4> m;
            for (int i = 0; i < 12; ++i)
            {
                auto sv = parser.next();
                SAIGA_ASSERT(!sv.empty());
                m(i / 4, i % 4) = to_double(sv);
            }
            //            std::cout << m << std::endl << std::endl;
            matrices.push_back(m);
        }
        SAIGA_ASSERT(matrices.size() == 4);

        // Extract K and bf
        // Distortion is 0
        Mat3 K1   = matrices[0].block<3, 3>(0, 0);
        Mat3 K2   = matrices[1].block<3, 3>(0, 0);
        double bf = -matrices[1](0, 3);

        intrinsics.model.K      = K1;
        intrinsics.rightModel.K = K2;
        intrinsics.bf           = bf;
    }


    std::cout << intrinsics << std::endl;

    std::vector<double> timestamps;
    {
        // load timestamps
        auto lines = File::loadFileStringArray(timesFile);
        for (auto l : lines)
        {
            if (l.empty()) continue;
            timestamps.push_back(Saiga::to_double(l));
        }
        std::cout << "got " << timestamps.size() << " timestamps" << std::endl;
    }

    std::vector<SE3> groundTruth;

    if (std::filesystem::exists(params.groundTruth))
    {
        // load ground truth
        std::cout << "loading ground truth " << std::endl;
        auto lines = File::loadFileStringArray(params.groundTruth);

        StringViewParser parser(" ");

        for (auto l : lines)
        {
            if (l.empty()) continue;
            parser.set(l);

            Eigen::Matrix<double, 3, 4> m;
            for (int i = 0; i < 12; ++i)
            {
                auto sv = parser.next();
                SAIGA_ASSERT(!sv.empty());
                m(i / 4, i % 4) = to_double(sv);
            }
            //            std::cout << m << std::endl << std::endl;
            //            matrices.push_back(m);
            Mat4 m4              = Mat4::Identity();
            m4.block<3, 4>(0, 0) = m;
            groundTruth.push_back(SE3::fitToSE3(m4));
        }



        SAIGA_ASSERT(groundTruth.size() == timestamps.size());
    }



    {
        // load left and right images
        //        frames.resize(N);


        int N = timestamps.size();

        if (params.maxFrames == -1)
        {
            params.maxFrames = N;
        }

        params.maxFrames = std::min(N - params.startFrame, params.maxFrames);

        frames.resize(params.maxFrames);
        N = params.maxFrames;



        SyncedConsoleProgressBar loadingBar(std::cout, "Loading " + to_string(N) + " images ", N);
#pragma omp parallel for if (params.multiThreadedLoad)
        for (int id = 0; id < params.maxFrames; ++id)
        {
            auto& frame = frames[id];

            int i = id + params.startFrame;

            std::string leftFile  = leftImageDir + "/" + leadingZeroString(i, 6) + ".png";
            std::string rightFile = rightImageDir + "/" + leadingZeroString(i, 6) + ".png";

            frame.grayImg.load(leftFile);
            frame.grayImg2.load(rightFile);

            SAIGA_ASSERT(frame.grayImg);
            SAIGA_ASSERT(frame.grayImg2);


            if (!groundTruth.empty()) frame.groundTruth = groundTruth[i];

            frame.timeStamp = timestamps[i];
            loadingBar.addProgress(1);
        }

        auto firstFrame           = frames.front();
        intrinsics.imageSize      = firstFrame.grayImg.dimensions();
        intrinsics.rightImageSize = firstFrame.grayImg2.dimensions();
    }
}

}  // namespace Saiga
