/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "CameraData.h"

#include "saiga/core/util/ini/ini.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vision/util/Ini.h"
namespace Saiga
{
std::ostream& operator<<(std::ostream& strm, const MonocularIntrinsics& value)
{
    strm << "[MonocularIntrinsics]" << std::endl;
    strm << "K: " << value.model.K.coeffs().transpose() << std::endl;
    strm << "Distortion: " << value.model.dis.transpose() << std::endl;
    strm << "Color: " << value.imageSize.w << "x" << value.imageSize.h << std::endl;
    strm << "Fps: " << value.fps << std::endl;
    return strm;
}

void RGBDIntrinsics::fromConfigFile(const std::string& file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    auto group = "RGBD-Sensor";

    INI_GETADD_LONG(ini, group, fps);


    INI_GETADD_DOUBLE_COMMENT(ini, group, depthFactor, "# The depth values are divided by this value to get meters.");
    INI_GETADD_DOUBLE_COMMENT(ini, group, maxDepth, "# Depth values above this value are unstable.");

    INI_GETADD_LONG_COMMENT(ini, group, imageSize.w, "# RGB Image");
    INI_GETADD_LONG(ini, group, imageSize.h);

    INI_GETADD_LONG_COMMENT(ini, group, depthImageSize.w, "# Depth Image");
    INI_GETADD_LONG(ini, group, depthImageSize.h);


    auto Kstr = toIniString(model.K);
    Kstr      = ini.GetAddString(group, "K", Kstr.c_str(), "#fx,fy,cx,cy");
    fromIniString(Kstr, model.K);


    INI_GETADD_DOUBLE(ini, group, bf);

    // Dis
    auto Dstr = toIniString(model.dis);
    Dstr      = ini.GetAddString(group, "dis", Dstr.c_str(), "#p1,p2,p3,p4,p5");
    fromIniString(Dstr, model.dis);


    double scale = double(depthImageSize.w) / imageSize.w;
    depthModel.K = model.K;
    depthModel.K.scale(scale);

    if (ini.changed()) ini.SaveFile(file.c_str());
}

std::ostream& operator<<(std::ostream& strm, const RGBDIntrinsics& value)
{
    strm << "[RGBDIntrinsics]" << std::endl;
    strm << "K: " << value.model.K.coeffs().transpose() << std::endl;
    strm << "depthK: " << value.depthModel.K.coeffs().transpose() << std::endl;
    strm << "Distortion: " << value.model.dis.transpose() << std::endl;
    strm << "Color: " << value.imageSize.w << "x" << value.imageSize.h << std::endl;
    strm << "Depth: " << value.depthImageSize.w << "x" << value.depthImageSize.h << std::endl;
    strm << "Fps: " << value.fps << std::endl;
    return strm;
}

std::ostream& operator<<(std::ostream& strm, const StereoIntrinsics& value)
{
    strm << "[StereoIntrinsics]" << std::endl;
    strm << "K:                " << value.model.K.coeffs().transpose() << std::endl;
    strm << "K right:          " << value.rightModel.K.coeffs().transpose() << std::endl;
    strm << "Distortion:       " << value.model.dis.transpose() << std::endl;
    strm << "Distortion right: " << value.rightModel.dis.transpose() << std::endl;
    strm << "Color:            " << value.imageSize.w << "x" << value.imageSize.h << std::endl;
    strm << "Color:            " << value.rightImageSize.w << "x" << value.rightImageSize.h << std::endl;
    strm << "Fps:              " << value.fps << std::endl;
    strm << "B * fx:           " << value.bf << std::endl;
    strm << "B (meters):       " << value.bf / value.model.K.fx << std::endl;
    return strm;
}
}  // namespace Saiga
