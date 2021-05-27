/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "SaigaDataset.h"

#include "saiga/core/util/FileSystem.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vision/util/DepthmapPreprocessor.h"

#include "TimestampMatcher.h"

#include <algorithm>
#include <fstream>
#include <thread>
namespace Saiga
{
SaigaDataset::SaigaDataset(const DatasetParameters& _params, bool scale_down_depth)
    : DatasetCameraBase(_params), scale_down_depth(scale_down_depth)
{
    camera_type = CameraInputType::RGBD;
    Load();
}

SaigaDataset::~SaigaDataset() {}



void SaigaDataset::LoadImageData(FrameData& data)
{
    auto old_id = data.id;
    data.Load(data.image_file);
    data.id = old_id;

    if (scale_down_depth)
    {
        TemplatedImage<float> tmp(_intrinsics.depthImageSize);
        DMPP::scaleDown2median(data.depth_image.getImageView(), tmp.getImageView());
        data.depth_image = tmp;
    }
}

int SaigaDataset::LoadMetaData()
{
    std::cout << "Loading Saiga Dataset: " << params.dir << std::endl;


    auto camera_file = params.dir + "/camera.ini";
    auto frame_dir   = params.dir + "/frames/";

    SAIGA_ASSERT(std::filesystem::exists(camera_file));
    SAIGA_ASSERT(std::filesystem::exists(frame_dir));
    SAIGA_ASSERT(std::filesystem::is_directory(frame_dir));

    _intrinsics.fromConfigFile(camera_file);



    if (scale_down_depth && _intrinsics.imageSize.width == _intrinsics.depthImageSize.width)
    {
        _intrinsics.depthModel.K = _intrinsics.depthModel.K.scale(0.5);
        _intrinsics.depthImageSize.width /= 2;
        _intrinsics.depthImageSize.height /= 2;
        std::cout << "Scaling down depth image." << std::endl;
    }
    else
    {
        std::cout << "Don't scale down." << std::endl;
        scale_down_depth = false;
    }

    std::cout << _intrinsics << std::endl;

    Directory d(frame_dir);


    frame_dirs = d.getDirectories();

    frame_dirs.erase(
        std::remove_if(frame_dirs.begin(), frame_dirs.end(),
                       [=](auto str) { return !std::filesystem::exists(frame_dir + "/" + str + "/info.txt"); }),
        frame_dirs.end());

    std::sort(frame_dirs.begin(), frame_dirs.end());


    if (params.maxFrames >= 0)
    {
        frame_dirs.resize(std::min((size_t)params.maxFrames, frame_dirs.size()));
    }
    params.maxFrames = frame_dirs.size();


    frames.resize(frame_dirs.size());

    for (int i = 0; i < frames.size(); ++i)
    {
        frames[i].image_file = frame_dir + "/" + frame_dirs[i] + "/";
        frames[i].id         = i;
    }


    imu                 = Imu::Sensor();
    imu->frequency      = 1600;
    imu->frequency_sqrt = sqrt(imu->frequency);

    return frame_dirs.size();
}


}  // namespace Saiga
