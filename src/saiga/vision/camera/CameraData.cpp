/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "CameraData.h"

#include "saiga/core/util/BinaryFile.h"
#include "saiga/core/util/ini/ini.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vision/util/Ini.h"

#include <fstream>
namespace Saiga
{
std::ostream& operator<<(std::ostream& strm, const MonocularIntrinsics& value)
{
    strm << "[MonocularIntrinsics]" << std::endl;
    strm << "K: " << value.model.K.coeffs().transpose() << std::endl;
    strm << "Distortion: " << value.model.dis.Coeffs().transpose() << std::endl;
    strm << "Color: " << value.imageSize.w << "x" << value.imageSize.h << std::endl;
    strm << "Fps: " << value.fps << std::endl;
    return strm;
}

void RGBDIntrinsics::fromConfigFile(const std::string& file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    auto group = "RGBD-Sensor";

    INI_GETADD_BOOL(ini, group, bgr);

    INI_GETADD_LONG(ini, group, fps);


    INI_GETADD_DOUBLE_COMMENT(ini, group, depthFactor, "# The depth values are divided by this value to get meters.");
    INI_GETADD_DOUBLE_COMMENT(ini, group, maxDepth, "# Depth values above this value are unstable.");

    INI_GETADD_DOUBLE_COMMENT(ini, group, bf, "# Baseline times fx");

    INI_GETADD_LONG_COMMENT(ini, group, imageSize.w, "# RGB Image");
    INI_GETADD_LONG(ini, group, imageSize.h);

    INI_GETADD_LONG_COMMENT(ini, group, depthImageSize.w, "# Depth Image");
    INI_GETADD_LONG(ini, group, depthImageSize.h);



    {
        // Color Intrinsics
        // K
        auto Kstr = toIniString(model.K);
        Kstr      = ini.GetAddString(group, "color_K", Kstr.c_str(), "#fx,fy,cx,cy,s");
        fromIniString(Kstr, model.K);

        // Dis
        Eigen::Matrix<double, 8, 1> dis = model.dis.Coeffs();
        auto Dstr                       = toIniString(dis);
        Dstr                            = ini.GetAddString(group, "color_dis", Dstr.c_str(), "#k1-k6 p1-p2");
        fromIniString(Dstr, dis);
        model.dis = Distortion(dis);
    }


    {
        // Depth Intrinsics
        // K
        auto Kstr = toIniString(depthModel.K);
        Kstr      = ini.GetAddString(group, "depth_K", Kstr.c_str(), "#fx,fy,cx,cy");
        fromIniString(Kstr, depthModel.K);

        // Dis
        Eigen::Matrix<double, 8, 1> dis = depthModel.dis.Coeffs();
        auto Dstr                       = toIniString(dis);
        Dstr                            = ini.GetAddString(group, "depth_dis", Dstr.c_str(), "#k1-k6 p1-p2");
        fromIniString(Dstr, dis);
        depthModel.dis = Distortion(dis);
    }

    {
        // extrinsics
        Eigen::Matrix<double, 7, 1> v = camera_to_body.params();
        auto Dstr                     = toIniString(v);
        Dstr                          = ini.GetAddString(group, "camera_to_body", Dstr.c_str(), "#SE3 (quat,vec3)");
        fromIniString(Dstr, v);


        Eigen::Map<Sophus::Vector<double, 7>> pose_map(camera_to_body.data());
        pose_map = v;
    }


    if (ini.changed()) ini.SaveFile(file.c_str());
}

std::ostream& operator<<(std::ostream& strm, const RGBDIntrinsics& value)
{
    strm << "[RGBDIntrinsics]" << std::endl;
    strm << "K : " << value.model.K.coeffs().transpose() << std::endl;
    strm << "dK: " << value.depthModel.K.coeffs().transpose() << std::endl;
    strm << "Distortion : " << value.model.dis.Coeffs().transpose() << std::endl;
    strm << "dDistortion: " << value.depthModel.dis.Coeffs().transpose() << std::endl;
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
    strm << "Distortion:       " << value.model.dis.Coeffs().transpose() << std::endl;
    strm << "Distortion right: " << value.rightModel.dis.Coeffs().transpose() << std::endl;
    strm << "Color:            " << value.imageSize.w << "x" << value.imageSize.h << std::endl;
    strm << "Color:            " << value.rightImageSize.w << "x" << value.rightImageSize.h << std::endl;
    strm << "Fps:              " << value.fps << std::endl;
    strm << "B * fx:           " << value.bf << std::endl;
    strm << "B (meters):       " << value.bf / value.model.K.fx << std::endl;
    return strm;
}

void FrameMetaData::Save(const std::string& dir) const
{
    std::ofstream ostream(dir + "/info.txt");
    ostream << std::setprecision(15);
    ostream << "# Id, Timestamp, Has GT" << std::endl;
    ostream << id << " " << timeStamp << " " << groundTruth.has_value() << std::endl;
    if (groundTruth.has_value())
    {
        ostream << "# GT SE3" << std::endl;
        ostream << groundTruth.value().params().transpose() << std::endl;
    }

    imu_data.Save(dir + "/imu.txt");
}

void FrameMetaData::Load(const std::string& dir)
{
    std::ifstream istream(dir + "/info.txt");

    std::string l;
    std::getline(istream, l);

    bool has_gt;
    istream >> id >> timeStamp >> has_gt;

    SAIGA_ASSERT(!has_gt);

    imu_data.Load(dir + "/imu.txt");
}

CameraInputType FrameData::CameraType()
{
    if (right_image || right_image_rgb)
    {
        return CameraInputType::Stereo;
    }

    if (depth_image)
    {
        return CameraInputType::RGBD;
    }

    if (image || image_rgb)
    {
        return CameraInputType::Mono;
    }

    return CameraInputType::Unknown;
}

void FrameData::Save(const std::string& dir) const
{
    FrameMetaData::Save(dir);

    // mono
    if (image_rgb.valid()) image_rgb.save(dir + "/color.png");
    if (image.valid()) image.save(dir + "/gray.png");

    // rgbd
    if (depth_image.valid()) depth_image.save(dir + "/depth.saigai");

    // stereo
    if (right_image_rgb.valid()) right_image_rgb.save(dir + "/right_color.png");
    if (right_image.valid()) right_image.save(dir + "/right_gray.png");
}

void FrameData::Load(const std::string& dir)
{
    FrameMetaData::Load(dir);

    // mono
    image_rgb.load(dir + "/color.png");
    image.load(dir + "/gray.png");

    // rgbd
    depth_image.load(dir + "/depth.saigai");

    // stereo
    right_image_rgb.load(dir + "/right_color.png");
    right_image.load(dir + "/right_gray.png");
}

}  // namespace Saiga
