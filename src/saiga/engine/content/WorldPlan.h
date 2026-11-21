/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The world plan of a space instance (docs/engine_design.md §12.1; S23a): the integer template view a generator sees,
// the sites of authored places (prefabs), the space instances and the integer plan grids. The generation pipeline sets
// seed and space, the generator's plan() fills sites, spaces and the grids, the pipeline fills sitesByChunk. Integer
// only (content code: the ContentLint test scans this header).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/world/SpaceKind.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
// The only part of a space template a generator sees: integers and the name (a double field cannot leak into content;
// generators never include the template header). Hashed into the plan hash.
struct GenerationTemplate
{
    std::string name;
    SpaceKind kind         = SpaceKind::OpenWorld;
    uint16_t templateIndex = 0;
    int32_t baseSpacing = 2, sizeChunks = 128;
    std::optional<int32_t> seaLevel;
};

// An integer grid over the space (biomes, rivers, ...): cells of cellSize metres (a power of two), the origin and the
// size in cells; row-major values
struct PlanGrid
{
    int32_t cellSize = 0;
    int32_t originX = 0, originZ = 0, width = 0, height = 0;
    std::vector<int32_t> values;
    // The value of a cell; a cell outside the grid takes the nearest border cell; 0 for an empty grid
    int32_t at(int64_t cellX, int64_t cellZ) const
    {
        if (width <= 0 || height <= 0 || values.empty()) return 0;
        int64_t x = cellX - originX, z = cellZ - originZ;
        x = x < 0 ? 0 : (x >= width ? width - 1 : x);
        z = z < 0 ? 0 : (z >= height ? height - 1 : z);
        return values[uint32_t(z * width + x)];
    }
};

// An authored place: a prefab at an integer position with a rotation in quarter turns (yaw direction, about +y)
struct Site
{
    uint32_t key = 0;  // unique in the plan
    std::string prefab;
    uint16_t prefabVersion = 1;
    int64_t x = 0, z = 0;      // the prefab origin, integer metres in the space
    uint8_t quarterTurns = 0;  // 0..3
    // Height units; prefab y = 0 lies here (the generator's choice; convention: the natural height at the site origin)
    int32_t baseHeight = 0;
    // [x0, z0, x1, z1) in local metres (checked against the prefab's footprint)
    int32_t footprint[4] = {};
};

// One space of the world: the pipeline lists the overworld, the generator's listed instances, the sites' instances and
// one per unreferenced template (§12.1)
struct SpaceInstance
{
    enum Origin : uint8_t
    {
        Overworld    = 0,
        Listed       = 1,
        FromSite     = 2,
        Unreferenced = 3
    };
    uint16_t templateIndex = 0;
    uint64_t seed          = 0;
    SpaceId parent         = 0;  // the space that created a site instance
    uint32_t siteKey       = 0;  // the creating site
    uint8_t link           = 0;  // the site's n-th prefab space
    uint8_t origin         = Overworld;
};

struct WorldPlan
{
    // Set by the pipeline before plan() (unchanged by it, asserted); read by plan, height, candidate and grassMask
    uint64_t seed = 0;
    GenerationTemplate space;
    // Filled by plan()
    std::vector<Site> sites;
    std::vector<SpaceInstance> spaces;  // listed instances (space 0 only)
    PlanGrid biomes, rivers;
    PlanGrid cloudCoverage;  // [S42] not hashed
    // Filled by the pipeline after plan() returns (empty before, asserted): site indices per chunk, sorted
    std::map<ChunkId, std::vector<uint32_t>> sitesByChunk;
};

}  // namespace Engine
}  // namespace Saiga
