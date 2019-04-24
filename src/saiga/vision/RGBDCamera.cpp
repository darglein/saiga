/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "RGBDCamera.h"

#include "saiga/core/util/ini/ini.h"
#include "saiga/core/util/threadName.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vision/Ini.h"
namespace Saiga
{
void RGBDIntrinsics::fromConfigFile(const std::string& file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());


    INI_GETADD_LONG(ini, "Sensor", fps);
    INI_GETADD_DOUBLE(ini, "Sensor", depthFactor);
    INI_GETADD_LONG(ini, "Sensor", maxFrames);

    rgbo.w = ini.GetAddLong("Color", "width", rgbo.w);
    rgbo.h = ini.GetAddLong("Color", "height", rgbo.h);

    deptho.w = ini.GetAddLong("Depth", "width", deptho.w);
    deptho.h = ini.GetAddLong("Depth", "height", deptho.h);

    auto Kstr = toIniString(K);

    if (ini.changed()) ini.SaveFile(file.c_str());
}
std::shared_ptr<RGBDFrameData> RGBDCamera::makeFrameData()
{
    auto fd = std::make_aligned_shared<RGBDFrameData>();
    fd->colorImg.create(intrinsics().rgbo.h, intrinsics().rgbo.w);
    fd->depthImg.create(intrinsics().deptho.h, intrinsics().deptho.w);
    return fd;
}

void RGBDCamera::setNextFrame(RGBDFrameData& data)
{
    data.frameId     = currentId++;
    data.captureTime = std::chrono::steady_clock::now();
}



}  // namespace Saiga
