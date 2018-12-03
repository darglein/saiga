/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "MotionModel.h"
#include "saiga/util/ini/ini.h"

namespace Saiga
{
void MotionModel::Parameters::fromConfigFile(const std::string& file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    smoothness = ini.GetAddLong("MotionModel", "smoothness", smoothness);
    alpha      = ini.GetAddDouble("MotionModel", "alpha", alpha);
    damping    = ini.GetAddDouble("MotionModel", "damping", damping);
    fps        = ini.GetAddDouble("MotionModel", "fps", fps);

    if (ini.changed()) ini.SaveFile(file.c_str());
}

MotionModel::MotionModel(const MotionModel::Parameters& params) : params(params) {}

void MotionModel::addRelativeMotion(const SE3& T)
{
    data.emplace_back(T);
}

SE3 MotionModel::getFrameVelocity()
{
    if (data.empty() || params.smoothness == 0) return SE3();

    int s      = std::min((int)data.size(), params.smoothness);
    SE3 result = data[data.size() - s];

    for (auto i = data.size() - s + 1; i < data.size(); ++i)
    {
        result = slerp(result, data[i], params.alpha);
    }
    return result;
    return scale(result, params.damping);
}

SE3 MotionModel::getRealVelocity()
{
    double factor = params.fps;
    return scale(getFrameVelocity(), factor);
}

void MotionModel::renderVelocityGraph()
{
    SE3 v = getRealVelocity();

    Eigen::AngleAxisd aa(v.unit_quaternion());
    Vec3 t = v.translation();

    double vt = t.norm();
    double va = aa.angle();

    grapht.addValue(vt);
    grapht.renderImGui();

    grapha.addValue(va);
    grapha.renderImGui();
}


}  // namespace Saiga
