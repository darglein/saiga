/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// SpaceTemplate (docs/engine_design.md §7.9, §12.2; S20): the template of a space of the world (WorldConfig::
// spaceTemplates). S20 implements name, kind (open world only), baseSpacing, seaLevel, sizeChunks and the prototype
// height hook; the other §7.9 fields come with their stages. validateSpaceTemplates checks a list (the World
// constructor; fatal in every build).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/math/TerrainSurface.h"
#include "saiga/core/util/assert.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
// Enclosed: S21 (rejected before)
enum class SpaceKind : uint8_t
{
    OpenWorld,
    Enclosed
};

struct SpaceTemplate
{
    std::string name = "overworld";
    SpaceKind kind   = SpaceKind::OpenWorld;
    // 1 (257 x 257 tiles) or 2 (129 x 129)
    int baseSpacing = 2;
    // Height units; drawn only for a space with terrain (hasTerrain)
    std::optional<int32_t> seaLevel = 0;
    // Open world: a multiple of 16 (4096 m root nodes), 16..512
    int32_t sizeChunks = 128;
    // Null: flat, y = 0, no terrain drawn (D1)
    TerrainHeightHook prototypeHeight = nullptr;

    // E6: the one test every terrain rule keys on (terrain and sea drawn, tiles requested, readiness, the mover's idle
    // rule, S21's describe_world.has_terrain and enclosed validation). Today prototypeHeight != nullptr; S23a
    // reimplements it as Generator::hasTerrain()
    bool hasTerrain() const { return prototypeHeight != nullptr; }
    // The size of the space in metres (sizeChunks * 256)
    double sizeMeters() const { return double(sizeChunks) * chunkSize; }
};

// The checks of WorldConfig::spaceTemplates (D1): SAIGA_EXIT_ERROR (every build) with a message naming the field
inline void validateSpaceTemplates(const std::vector<SpaceTemplate>& templates)
{
    if (templates.size() > 1)
        SAIGA_EXIT_ERROR("WorldConfig::spaceTemplates: more than one space template (further spaces come with S21)");
    for (const SpaceTemplate& t : templates)
    {
        const std::string what = "space template \"" + t.name + "\": ";
        if (t.kind == SpaceKind::Enclosed) SAIGA_EXIT_ERROR(what + "kind Enclosed is not supported before S21");
        if (t.baseSpacing != 1 && t.baseSpacing != 2)
            SAIGA_EXIT_ERROR(what + "baseSpacing must be 1 or 2 (is " + std::to_string(t.baseSpacing) + ")");
        if (t.kind == SpaceKind::OpenWorld && (t.sizeChunks % 16 != 0 || t.sizeChunks < 16 || t.sizeChunks > 512))
            SAIGA_EXIT_ERROR(what + "sizeChunks of an open-world space must be a multiple of 16 in 16..512 (is " +
                             std::to_string(t.sizeChunks) + ")");
        if (t.seaLevel && (*t.seaLevel > kTerrainHeightLimit || *t.seaLevel < -kTerrainHeightLimit))
            SAIGA_EXIT_ERROR(what + "seaLevel beyond kTerrainHeightLimit (" + std::to_string(*t.seaLevel) + ")");
    }
}

}  // namespace Engine
}  // namespace Saiga
