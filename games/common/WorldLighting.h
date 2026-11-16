/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/world/WorldRenderer.h"

#include "games/common/MeshRenderer.h"

namespace Saiga
{
namespace Game
{
// The lighting of the world prototype (sun from the settings, the same height fog), so the game meshes match it
inline MeshRenderer::Lighting worldLighting(const Vulkan::WorldRenderer& world)
{
    const auto& s = world.settings;
    double az = s.sunAzimuth * kPi / 180.0, el = s.sunElevation * kPi / 180.0;
    MeshRenderer::Lighting l;
    l.sunDirection  = Vec3(std::cos(el) * std::sin(az), std::sin(el), -std::cos(el) * std::cos(az));
    l.sunIntensity  = s.sunIntensity;
    l.ambient       = s.ambient;
    l.fog           = s.fog;
    l.fogDensity    = s.fogDensity;
    l.fogFalloff    = s.fogFalloff;
    l.fogBaseHeight = world.heightField().waterLevel();
    l.fogColor      = Vec3(0.42, 0.42, 0.42);
    return l;
}

// The ground of the world prototype for the game physics: terrain or the water surface, whichever is higher
inline GroundHeight worldGround(const Vulkan::WorldRenderer& world, bool includeWater)
{
    const auto* hf = &world.heightField();
    return [hf, includeWater](double x, double z)
    {
        double h = hf->height(float(x), float(z));
        return includeWater ? std::max(h, double(hf->waterLevel())) : h;
    };
}

}  // namespace Game
}  // namespace Saiga
