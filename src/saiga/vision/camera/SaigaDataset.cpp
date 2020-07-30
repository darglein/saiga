/**
 * Copyright (c) 2017 Darius Rückert
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

#include "TimestampMatcher.h"

#include <algorithm>
#include <fstream>
#include <thread>
namespace Saiga
{
SaigaDataset::SaigaDataset(const DatasetParameters& _params) : DatasetCameraBase<RGBDFrameData>(_params)
{
    Load();
}

SaigaDataset::~SaigaDataset() {}



void SaigaDataset::LoadImageData(RGBDFrameData& data)
{
    auto old_id = data.id;
    data.Load(data.file);
    data.id = old_id;
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



    std::cout << _intrinsics << std::endl;

    Directory d(frame_dir);

    frame_dirs.clear();
    d.getDirectories(frame_dirs);

    frame_dirs.erase(
        std::remove_if(frame_dirs.begin(), frame_dirs.end(),
                       [=](auto str) { return !std::filesystem::exists(frame_dir + "/" + str + "/info.txt"); }),
        frame_dirs.end());

    std::sort(frame_dirs.begin(), frame_dirs.end());

    frames.resize(frame_dirs.size());

    for (int i = 0; i < frames.size(); ++i)
    {
        frames[i].file = frame_dir + "/" + frame_dirs[i] + "/";
        frames[i].id   = i;
    }


    imu                 = Imu::Sensor();
    imu->frequency      = 1600;
    imu->frequency_sqrt = sqrt(imu->frequency);

    return frame_dirs.size();
#if 0
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
        _intrinsics.model.K = Intrinsics4(517.306408, 516.469215, 318.643040, 255.313989);
        // 0.262383 , -0.953104, -0.005358, 0.002628, 1.163314;
        _intrinsics.model.dis.k1 = 0.262383;
        _intrinsics.model.dis.k2 = -0.953104;
        _intrinsics.model.dis.p1 = -0.005358;
        _intrinsics.model.dis.p2 = 0.002628;
        _intrinsics.model.dis.k3 = 1.163314;
    }
    else if (freiburg == 2)
    {
        _intrinsics.model.K = Intrinsics4(520.908620, 521.007327, 325.141442, 249.701764);
        //        _intrinsics.model.dis << 0.231222, -0.784899, -0.003257, -0.000105, 0.917205;
        _intrinsics.model.dis.k1 = 0.231222;
        _intrinsics.model.dis.k2 = -0.784899;
        _intrinsics.model.dis.p1 = -0.003257;
        _intrinsics.model.dis.p2 = -0.000105;
        _intrinsics.model.dis.k3 = 0.917205;
    }
    else if (freiburg == 3)
    {
        _intrinsics.model.K = Intrinsics4(535.4, 539.2, 320.1, 247.6);
        //        _intrinsics.model.dis << 0, 0, 0, 0, 0;
    }
    else
    {
        SAIGA_EXIT_ERROR("Invalid Freiburg");
    }

    _intrinsics.depthModel = _intrinsics.model;

    VLOG(1) << "Found Freiburg " << freiburg << " dataset.";
    VLOG(1) << _intrinsics;

    associate(params.dir);
    load(params.dir, params.multiThreadedLoad);
    return frames.size();
#endif
}


}  // namespace Saiga
