/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "KittiDataset.h"

#include "saiga/core/util/FileSystem.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vision/camera/TimestampMatcher.h"

namespace Saiga
{
KittiDataset::KittiDataset(const DatasetParameters& params_) : DatasetCameraBase(params_)
{
    camera_type = CameraInputType::Stereo;
    // Kitti was recorded with 10 fps
    intrinsics.fps = 10;
    Load();
}

int KittiDataset::LoadMetaData()
{
    std::cout << "Loading KittiDataset Stereo Dataset: " << params.dir << std::endl;

    auto leftImageDir  = params.dir + "/image_0";
    auto rightImageDir = params.dir + "/image_1";
    auto calibFile     = params.dir + "/calib.txt";
    auto timesFile     = params.dir + "/times.txt";
    //    auto groundtruthFile = params.groundTruth;

    SAIGA_ASSERT(std::filesystem::exists(leftImageDir));
    SAIGA_ASSERT(std::filesystem::exists(rightImageDir));
    SAIGA_ASSERT(std::filesystem::exists(calibFile));
    SAIGA_ASSERT(std::filesystem::exists(timesFile));


    std::filesystem::path p(params.dir + "/");
    std::string sequence_number_str = p.parent_path().filename().string();
    int sequence_number             = std::atoi(sequence_number_str.c_str());
    SAIGA_ASSERT(sequence_number >= 0 && sequence_number <= 21);



    //    std::array<double, 22> bias_table = {
    //        0.0531029,   // 00
    //        -0.0526353,  // 01
    //        0,           // 02
    //        -0.0112762,  // 03
    //        -0.0271705,  // 04
    //        -0.0827276,  // 05
    //        -0.160229,   // 06
    //        0.0324279,   // 07
    //        -0.0223322,  // 08
    //        0.0144021,   // 09
    //        0,           // 10
    //        -0.0124689,  // 11
    //        0,           // 12
    //        -0.0545794,  // 13
    //        -0.0262804,  // 14
    //        0.00235434,  // 15
    //        0.0669488,   // 16
    //        0.10344,     // 17
    //        -0.0482539,  // 18
    //        0.0173151,   // 19
    //        -0.0721045,  // 20
    //        0.156678,    // 21
    //    };



    //    if (sequence_number <= 2)
    //    {
    //        intrinsics.depth_bias = 1.00255;
    //    }
    //    else if (sequence_number == 3)
    //    {
    //        intrinsics.depth_bias = 0.994022;
    //    }
    //    else if (sequence_number >= 4)
    //    {
    //        intrinsics.depth_bias = 0.992174;
    //    }

    //    intrinsics.depth_bias = bias_table[sequence_number];

    // search for ground truth
    std::string groundtruthFile = "";
    if (hasEnding(groundtruthFile, ".txt"))
    {
        SAIGA_ASSERT(std::filesystem::exists(groundtruthFile));
    }
    else
    {
        groundtruthFile = "";
    }

    if (groundtruthFile.empty())
    {
        auto target_file = sequence_number_str + ".txt";

        FileChecker checker;
        checker.addSearchPath(params.dir);
        checker.addSearchPath(params.dir + "/..");
        checker.addSearchPath(params.dir + "/../poses/");
        checker.addSearchPath(params.dir + "/../../");
        checker.addSearchPath(params.dir + "/../../poses/");

        //        std::cout << sequence_number_str << std::endl;
        //        std::cout << checker << std::endl;

        groundtruthFile = checker.getFile(target_file);
    }


    if (!groundtruthFile.empty())
    {
        std::cout << "Found Ground Truth: " << groundtruthFile << std::endl;
    }

    //    SAIGA_ASSERT(!groundtruthFile.empty());



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
        intrinsics.maxDepth     = 35;
    }



    std::vector<double> timestamps;
    {
        // load timestamps
        auto lines = File::loadFileStringArray(timesFile);
        for (auto l : lines)
        {
            if (l.empty()) continue;
            timestamps.push_back(Saiga::to_double(l));
        }
    }

    std::vector<SE3> groundTruth;

    if (!groundtruthFile.empty())
    {
        // load ground truth
        auto lines = File::loadFileStringArray(groundtruthFile);

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


        for (int id = 0; id < params.maxFrames; ++id)
        {
            auto& frame = frames[id];

            int i = id + params.startFrame;

            frame.image_file       = leftImageDir + "/" + leadingZeroString(i, 6) + ".png";
            frame.right_image_file = rightImageDir + "/" + leadingZeroString(i, 6) + ".png";

            frame.id = id;



            if (!groundTruth.empty()) frame.groundTruth = groundTruth[i];

            frame.timeStamp = timestamps[i];
        }

        {
            auto firstFrame = frames.front();
            LoadImageData(firstFrame);
            intrinsics.imageSize      = firstFrame.image.dimensions();
            intrinsics.rightImageSize = firstFrame.right_image.dimensions();
        }
    }


    VLOG(1) << intrinsics;
    return frames.size();
}

void KittiDataset::LoadImageData(FrameData& data)
{
    SAIGA_ASSERT(data.image.rows == 0);

    data.image.load(data.image_file);
    if (!params.force_monocular)
    {
        data.right_image.load(data.right_image_file);
    }
}

}  // namespace Saiga
