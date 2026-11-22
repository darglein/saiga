/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/terrain/TerrainSystem.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Engine
{
TerrainSystem::TerrainSystem(const std::vector<SpaceTemplate>& templates, const std::vector<uint64_t>& seeds)
{
    SAIGA_ASSERT(templates.size() == seeds.size());
    for (size_t i = 0; i < templates.size(); ++i)
    {
        Space s;
        s.settings = templates[i];
        s.seed     = seeds[i];
        s.source   = TerrainHeightSource::fromHook(s.settings.prototypeHeight, s.seed);
        spaces.push_back(std::move(s));
    }
}

TerrainSystem::~TerrainSystem() = default;

const SpaceTemplate& TerrainSystem::spaceTemplate(SpaceId space) const
{
    SAIGA_ASSERT(space < spaces.size(), "TerrainSystem: unknown space");
    return spaces[space].settings;
}

const TerrainHeightSource& TerrainSystem::heightSource(SpaceId space) const
{
    SAIGA_ASSERT(space < spaces.size(), "TerrainSystem: unknown space");
    return spaces[space].source;
}

int32_t TerrainSystem::sample(SpaceId space, int64_t sx, int64_t sz) const
{
    const Space& s = spaces[space];
    if (!s.settings.hasTerrain()) return 0;
    const int64_t spacing = s.settings.baseSpacing;
    return s.source(sx * spacing, sz * spacing);
}

double TerrainSystem::height(SpaceId space, double x, double z) const
{
    SAIGA_ASSERT(space < spaces.size(), "TerrainSystem: unknown space");
    const Space& s = spaces[space];
    if (!s.settings.hasTerrain()) return 0;
    const double size = s.settings.sizeMeters();
    // Sample coordinates: exact for spacing 1 or 2
    const double spacing = s.settings.baseSpacing;
    const double gx      = std::min(std::max(x, 0.0), size) / spacing;
    const double gz      = std::min(std::max(z, 0.0), size) / spacing;
    const int64_t n      = int64_t(s.settings.sizeChunks) * 256 / s.settings.baseSpacing;
    // The last cell at the far edge (fx = 1: the sample at the edge exists, tiles share their borders)
    const int64_t ci  = std::min(int64_t(std::floor(gx)), n - 1);
    const int64_t cj  = std::min(int64_t(std::floor(gz)), n - 1);
    const double fx   = gx - double(ci);
    const double fz   = gz - double(cj);
    const int32_t h00 = sample(space, ci, cj), h11 = sample(space, ci + 1, cj + 1);
    // Only the samples of the containing triangle (A: fx >= fz uses h10, B: h01)
    int32_t h10 = 0, h01 = 0;
    if (fx >= fz)
        h10 = sample(space, ci + 1, cj);
    else
        h01 = sample(space, ci, cj + 1);
    return terrainSurfaceDouble(h00, h10, h01, h11, fx, fz) * kHeightUnit;
}

}  // namespace Engine
}  // namespace Saiga
