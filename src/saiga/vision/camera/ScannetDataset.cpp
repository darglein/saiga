/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ScannetDataset.h"

#include "saiga/core/util/FileSystem.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/tostring.h"

#include "TimestampMatcher.h"

#include <algorithm>
#include <fstream>
#include <thread>
namespace Saiga
{
ScannetDataset::ScannetDataset(const DatasetParameters& _params, bool scale_down_color, bool scale_down_depth)
    : DatasetCameraBase(_params), scale_down_color(scale_down_color), scale_down_depth(scale_down_depth)
{
    camera_type = CameraInputType::RGBD;
    Load();
}


void ScannetDataset::LoadImageData(FrameData& data)
{
    data.image_rgb.create(intrinsics().imageSize.h, intrinsics().imageSize.w);
    data.depth_image.create(intrinsics().depthImageSize.h, intrinsics().depthImageSize.w);


    Image cimg(data.image_file);
    Image dimg(data.depth_file);

    SAIGA_ASSERT(cimg.valid());
    SAIGA_ASSERT(dimg.valid());

    if (cimg.type == UC3)
    {
        if (scale_down_color)
        {
            RGBImageType tmp(cimg.h, cimg.w);
            ImageTransformation::addAlphaChannel(cimg.getImageView<ucvec3>(), tmp);
            ImageTransformation::ScaleDown2(tmp.getImageView(), data.image_rgb.getImageView());
        }
        else
        {
            // convert to rgba
            ImageTransformation::addAlphaChannel(cimg.getImageView<ucvec3>(), data.image_rgb);
        }
    }
    else if (cimg.type == UC4)
    {
        if (scale_down_color)
        {
            RGBImageType tmp(cimg.h, cimg.w);
            cimg.getImageView<ucvec4>().copyTo(tmp.getImageView());
            ImageTransformation::ScaleDown2(tmp.getImageView(), data.image_rgb.getImageView());
        }
        else
        {
            cimg.getImageView<ucvec4>().copyTo(data.image_rgb.getImageView());
        }
    }
    else
    {
        SAIGA_EXIT_ERROR("invalid image type");
    }



    if (dimg.type == US1)
    {
        if (scale_down_depth)
        {
            DepthImageType tmp(dimg.h, dimg.w);
            dimg.getImageView<unsigned short>().copyTo(tmp.getImageView(), 1.0 / intrinsics().depthFactor);
            dmpp.scaleDown2median(tmp.getImageView(), data.depth_image.getImageView());
        }
        else
        {
            dimg.getImageView<unsigned short>().copyTo(data.depth_image.getImageView(), 1.0 / intrinsics().depthFactor);
        }
    }
    else
    {
        SAIGA_EXIT_ERROR("invalid image type");
    };
}

int ScannetDataset::LoadMetaData()
{
    std::cout << "Loading Scannet RGBD Dataset: " << params.dir << std::endl;

    SAIGA_ASSERT(std::filesystem::exists(params.dir + "/color/"));
    SAIGA_ASSERT(std::filesystem::exists(params.dir + "/depth/"));
    SAIGA_ASSERT(std::filesystem::exists(params.dir + "/intrinsic/"));


    _intrinsics.fps         = 30;
    _intrinsics.depthFactor = 1000;

    _intrinsics.imageSize.width       = 1296;
    _intrinsics.imageSize.height      = 968;
    _intrinsics.depthImageSize.width  = 640;
    _intrinsics.depthImageSize.height = 480;
    _intrinsics.bf                    = 40;



    {
        // Color Intrinsics
        std::array<double, 16> posearray;
        std::ifstream pstream(params.dir + "/intrinsic/intrinsic_color.txt");
        for (auto& pi : posearray)
        {
            pstream >> pi;
        }
        Mat4 intrinsics     = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(posearray.data());
        _intrinsics.model.K = IntrinsicsPinholed(Mat3(intrinsics.block<3, 3>(0, 0)));
    }

    {
        // Color Intrinsics
        std::array<double, 16> posearray;
        std::ifstream pstream(params.dir + "/intrinsic/intrinsic_depth.txt");
        for (auto& pi : posearray)
        {
            pstream >> pi;
        }
        Mat4 intrinsics          = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(posearray.data());
        _intrinsics.depthModel.K = IntrinsicsPinholed(Mat3(intrinsics.block<3, 3>(0, 0)));
    }



    if (scale_down_color)
    {
        _intrinsics.model.K = _intrinsics.model.K.scale(0.5);
        _intrinsics.imageSize.width /= 2;
        _intrinsics.imageSize.height /= 2;
    }

    if (scale_down_depth)
    {
        _intrinsics.depthModel.K = _intrinsics.depthModel.K.scale(0.5);
        _intrinsics.depthImageSize.width /= 2;
        _intrinsics.depthImageSize.height /= 2;
    }

    Directory color_dir(params.dir + "/color/");

    std::vector<std::string> files;
    files = color_dir.getFilesEnding(".jpg");

    int N = files.size();

    if (params.maxFrames > 0)
    {
        N = std::min<int>(files.size(), params.maxFrames);
    }
    frames.resize(N);

    for (int i = 0; i < N; ++i)
    {
        auto& frame = frames[i];
        frame.id    = i;

        auto number_str = std::to_string(i + params.startFrame);

        frame.timeStamp  = i * 0.1;
        frame.image_file = params.dir + "/color/" + number_str + ".jpg";
        frame.depth_file = params.dir + "/depth/" + number_str + ".png";

        auto pose_str = params.dir + "/pose/" + number_str + ".txt";
        if (std::filesystem::exists(pose_str))
        {
            // Precomputed pose used as ground truth.
            // Note: this is not actual GT just for reference.
            std::array<double, 16> posearray;
            std::ifstream pstream(pose_str);
            for (auto& pi : posearray)
            {
                pstream >> pi;
            }
            Mat4 ex           = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(posearray.data());
            SE3 pose          = SE3::fitToSE3(ex);
            frame.groundTruth = pose;
        }
    }


    return frames.size();
}



}  // namespace Saiga
