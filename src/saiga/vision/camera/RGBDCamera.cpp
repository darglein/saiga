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

    // K
    auto Kstr = toIniString(K);
    Kstr      = ini.GetAddString("ColorIntr", "K", Kstr.c_str(), "#fx,fy,cx,cy,bf");
    fromIniString(Kstr, K);

    // Dis
    auto Dstr = toIniString(dis);
    Dstr      = ini.GetAddString("ColorIntr", "dis", Dstr.c_str(), "#p1,p2,p3,p4,p5");
    fromIniString(Dstr, dis);


    double scale = double(deptho.w) / rgbo.w;
    depthK       = K;
    depthK.scale(scale);

    if (ini.changed()) ini.SaveFile(file.c_str());
}

std::ostream& operator<<(std::ostream& strm, const RGBDIntrinsics& value)
{
    strm << "[RGBDIntrinsics]" << endl;
    strm << "K: " << value.K.coeffs().transpose() << endl;
    strm << "depthK: " << value.depthK.coeffs().transpose() << endl;
    strm << "Distortion: " << value.dis.transpose() << endl;
    strm << "Color: " << value.rgbo.w << "x" << value.rgbo.h << endl;
    strm << "Depth: " << value.deptho.w << "x" << value.deptho.h << endl;
    strm << "Fps: " << value.fps << endl;
    strm << "MaxFrames: " << value.maxFrames << endl;
    return strm;
}

std::unique_ptr<RGBDFrameData> RGBDCamera::makeFrameData()
{
    auto fd = std::make_unique<RGBDFrameData>();
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
