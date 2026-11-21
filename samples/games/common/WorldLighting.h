/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/world/WorldRenderer.h"

#include "games/common/GameMath.h"

namespace Saiga
{
namespace Game
{
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
