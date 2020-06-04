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


    INI_GETADD_LONG(ini, "Sensor", fps);
    INI_GETADD_DOUBLE(ini, "Sensor", depthFactor);

    imageSize.w = ini.GetAddLong("Color", "width", imageSize.w);
    imageSize.h = ini.GetAddLong("Color", "height", imageSize.h);

    depthImageSize.w = ini.GetAddLong("Depth", "width", depthImageSize.w);
    depthImageSize.h = ini.GetAddLong("Depth", "height", depthImageSize.h);

    // K
    auto Kstr = toIniString(model.K);
    Kstr      = ini.GetAddString("ColorIntr", "K", Kstr.c_str(), "#fx,fy,cx,cy");
    fromIniString(Kstr, model.K);


    INI_GETADD_DOUBLE(ini, "ColorIntr", bf);

    // Dis
    auto Dstr = toIniString(model.dis);
    Dstr      = ini.GetAddString("ColorIntr", "dis", Dstr.c_str(), "#p1,p2,p3,p4,p5");
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
