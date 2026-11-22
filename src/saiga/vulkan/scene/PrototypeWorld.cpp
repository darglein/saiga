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

// Towards the sun from the azimuth and elevation (degrees), as the prototype
vec3 sunDirection(const PrototypeSettings& s)
{
    float az = radians(s.sunAzimuth), el = radians(s.sunElevation);
    return vec3(std::cos(el) * std::sin(az), std::sin(el), -std::cos(el) * std::cos(az));
}
}  // namespace

StudioEnvironment studioFromPrototype(const PrototypeSettings& s)
{
    StudioEnvironment e;
    e.sunDir       = sunDirection(s);
    e.sunIntensity = s.sunIntensity;
    e.ambient      = s.ambient;
    e.hemisphere   = true;
    e.albedo       = s.albedo;
    e.fog          = s.fog;
    e.fogDensity   = s.fogDensity;
    e.fogFalloff   = s.fogFalloff;
    e.fogColor     = vec3(0.42f, 0.42f, 0.42f);
    return e;
}

StudioEnvironment prototypeStudio()
{
    return studioFromPrototype(PrototypeSettings());
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

void prototypeLighting(const PrototypeSettings& s, double time, float fogBase, SceneLighting& out)
{
    const vec3 sun = sunDirection(s);
    out.sunDir     = vec4(sun(0), sun(1), sun(2), s.sunIntensity);
    out.ambient    = vec4(s.ambient, 1, s.albedo, s.alphaCutoff);
    out.fog        = vec4(s.fogDensity, s.fogFalloff, s.fog ? 1 : 0, fogBase);
    out.fogColor   = vec4(0.42f, 0.42f, 0.42f, 1);
    // The tree.vert frequencies of the sway (x, z) and the flutter
    out.wind =
        vec4(prototypeWindPhase(time, 1.1), prototypeWindPhase(time, 0.9), prototypeWindPhase(time, 4.0), s.wind);
    out.debug = vec4(float(s.debugMode), 0, 0, 0);
}

}  // namespace Vulkan
}  // namespace Saiga
