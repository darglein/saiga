/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "MotionModel.h"

#include "saiga/core/util/ini/ini.h"

namespace Saiga
{
void MotionModel::Settings::fromConfigFile(const std::string& file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    INI_GETADD(ini, "MotionModel", valid_range);
    INI_GETADD(ini, "MotionModel", damping);

    if (ini.changed()) ini.SaveFile(file.c_str());
}



void MotionModel::addRelativeMotion(const SE3& velocity, int frameId)
{
    // Sanity check. Maybe remove?
    SAIGA_ASSERT(frameId >= 0);
    SAIGA_ASSERT(frameId < 100 * 1000);

    std::unique_lock<std::mutex> lock(mut);
    // Make sure the new frame fits into the array
    data.resize(frameId + 1);
    data[frameId].velocity = velocity;
    data[frameId].valid    = true;
}

std::optional<SE3> MotionModel::predictVelocityForFrame(int frameId)
{
    if (frameId < data.size())
    {
        // let's check if we have an actual meassurement for this frameid.
        if (data[frameId].valid)
        {
            return data[frameId].velocity;
        }
    }

    // Find the last valid velocity.
    int start_idx = std::min(frameId, (int)data.size() - 1);
    while (start_idx >= 0 && !data[start_idx].valid)
    {
        start_idx--;
    }

    if (start_idx < 0)
    {
        // didn't found anything
        return {};
    }

    int frame_distance = frameId - start_idx;
    if (frame_distance > params.valid_range)
    {
        // last prediction was too far away
        return {};
    }

    auto v = data[start_idx].velocity;
    v      = scale(v, params.damping);
    return v;
}

void MotionModel::ScaleLinearVelocity(double scale)
{
    for(auto& d : data)
    {
        d.velocity.translation() *= scale;
    }
}


#if 0
SE3 MotionModel::computeVelocity()
{
    if (data.empty() || params.smoothness == 0) return SE3();

    // Number of values to consider
    int s = std::min((int)data.size(), params.smoothness);

    //    return data.back().v;
    weights.resize(s);
    double weightSum = 0;
    for (auto i = 0; i < s; ++i)
    {
        auto dataId = data.size() - s + i;

        double disToCurrent = s - (i + 1);
        auto w              = data[dataId].weight * std::pow(params.alpha, disToCurrent);
        weights[i]          = w;
        weightSum += w;
    }

    if (weightSum < 1e-20) return SE3();

    // normalize weights
    for (auto i = 0; i < s; ++i)
    {
        weights[i] /= weightSum;
    }

    SE3 result           = data[data.size() - s].v;
    double currentWeight = weights[0];

    for (auto i = 1; i < s; ++i)
    {
        double nextWeight = weights[i];
        auto dataId       = data.size() - s + i;

        if (currentWeight + nextWeight < 1e-20) continue;

        double slerpAlpha = nextWeight / (currentWeight + nextWeight);

        result = slerp(result, data[dataId].v, slerpAlpha);
        currentWeight += nextWeight;
    }
    result = scale(result, params.damping);
    //    std::cout << result << std::endl;
    //    std::cout << data.back().v << std::endl << std::endl;
    return result;
    SE3 result = data[data.size() - s].v;
    for (auto i = 1; i < s; ++i)
    {
        auto dataId = data.size() - s + i;

        result = slerp(result, data[dataId].v, params.alpha);
    }
    result.translation() *= params.damping;
    //    result = scale(result, params.damping);
    return result;




}
#endif


}  // namespace Saiga
