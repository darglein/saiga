/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/PrototypeWorld.h"

#include "saiga/vulkan/scene/SceneCulling.h"

#include <cmath>

namespace Saiga
{
namespace Vulkan
{
namespace
{
constexpr double twoPi = 6.283185307179586;
}  // namespace

StudioEnvironment prototypeStudio()
{
    // S19a's PrototypeSettings defaults: the sun from azimuth 35, elevation 38 degrees with intensity 1, the hemisphere
    // ambient 0.25, the albedo 0.55, the height fog 0.00015 / 0.0012 in gray 0.42
    StudioEnvironment e;
    const float az = radians(35.0f), el = radians(38.0f);
    e.sunDir       = vec3(std::cos(el) * std::sin(az), std::sin(el), -std::cos(el) * std::cos(az));
    e.sunIntensity = 1.0f;
    e.ambient      = 0.25f;
    e.hemisphere   = true;
    e.albedo       = 0.55f;
    e.fog          = true;
    e.fogDensity   = 0.00015f;
    e.fogFalloff   = 0.0012f;
    e.fogColor     = vec3(0.42f, 0.42f, 0.42f);
    return e;
}

uint32_t packPrototypeFoliageColor(int layer, float ao, float wind)
{
    SAIGA_ASSERT(layer < 255, "packPrototypeFoliageColor: the layer must be < 255 (255: opaque)");
    SAIGA_ASSERT(ao >= 0 && ao <= 1, "packPrototypeFoliageColor: the AO must be in [0, 1]");
    SAIGA_ASSERT(wind >= 0 && wind <= 1, "packPrototypeFoliageColor: the wind weight must be in [0, 1]");
    const uint32_t l = layer < 0 ? 0xFFu : uint32_t(layer);
    const uint32_t a = uint32_t(std::lround(std::min(std::max(ao, 0.0f), 1.0f) * 255.0f));
    const uint32_t w = uint32_t(std::lround(std::min(std::max(wind, 0.0f), 1.0f) * 255.0f));
    return l | (a << 8) | (w << 16);
}

float prototypeWindPhase(double time, double frequency)
{
    double p = std::fmod(time * frequency, twoPi);
    if (p < 0) p += twoPi;
    return float(p);
}


}  // namespace Vulkan
}  // namespace Saiga
