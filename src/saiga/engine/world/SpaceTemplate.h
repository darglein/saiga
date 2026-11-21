/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// SpaceTemplate (docs/engine_design.md §7.9, §12.2, §12.4; S20, S21): the template of a space of the world
// (WorldConfig::spaceTemplates, SpaceId = index). S20 implements name, kind, baseSpacing, seaLevel, sizeChunks and the
// prototype height hook; S21 enclosed spaces (SpaceTemplate::enclosed, the site origin) and the background; the other
// §7.9 fields come with their stages. validateSpaceTemplates checks a list (the World constructor; fatal in every
// build).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/math/TerrainSurface.h"
#include "saiga/core/math/Types.h"
#include "saiga/core/util/assert.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/engine/world/SpaceKind.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
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
    // Null: flat, y = 0, no terrain drawn (D1); an enclosed space has none. Transitional (S20): replaced by the
    // template's generator (a HookGenerator) in S23a's step 7
    TerrainHeightHook prototypeHeight = nullptr;
    // The content generator of the space (§12.1; S23a): a GeneratorId of WorldSetup::generators; 0 is the engine's
    // FlatGenerator (flat, no terrain, no content)
    GeneratorId generator = 0;
    // Prefab sites within this distance of a Player (metres) are stamped (§7.9, §12.3; S23a); scenario worlds lower
    // it to at most 128 m
    double fullRadius = 1000;
    // The clear color of an enclosed space (linear RGB; open-world spaces draw the sky). S36's ambient / fog replace it
    Vec3 background = Vec3(0.02, 0.02, 0.025);

    // An enclosed space of sizeChunks x sizeChunks chunks (chunk range [0, sizeChunks)^2; 4: 1 km): no terrain, no sea
    static SpaceTemplate enclosed(std::string name, int32_t sizeChunks = 4)
    {
        SpaceTemplate t;
        t.name       = std::move(name);
        t.kind       = SpaceKind::Enclosed;
        t.seaLevel   = std::nullopt;
        t.sizeChunks = sizeChunks;
        return t;
    }
    // The site origin of an enclosed space (D12, E4): the centre of the middle chunk, on both axes, in metres
    static int64_t enclosedOrigin(int32_t sizeChunks) { return int64_t(sizeChunks / 2) * 256 + 128; }

    // E6: the one test every terrain rule keys on (terrain and sea drawn, tiles requested, readiness, the mover's idle
    // rule, S21's describe_world.has_terrain and enclosed validation). Today prototypeHeight != nullptr; S23a
    // reimplements it as Generator::hasTerrain()
    bool hasTerrain() const { return prototypeHeight != nullptr; }
    // The size of the space in metres (sizeChunks * 256)
    double sizeMeters() const { return double(sizeChunks) * chunkSize; }
};

// The checks of WorldConfig::spaceTemplates (D1; S21): SAIGA_EXIT_ERROR (every build) with a message naming the
// template and the field. At most 65535 templates with unique non-empty names (at most 256 bytes); an enclosed
// template has no terrain and 1..512 chunks (its seaLevel and baseSpacing are ignored); an open-world template has
// spacing 1 or 2, a multiple of 16 chunks in 16..512 and a seaLevel within kTerrainHeightLimit (a hook is optional)
inline void validateSpaceTemplates(const std::vector<SpaceTemplate>& templates)
{
    if (templates.size() > 65535)
        SAIGA_EXIT_ERROR("WorldConfig::spaceTemplates: more than 65535 space templates (" +
                         std::to_string(templates.size()) + ")");
    std::set<std::string> names;
    for (const SpaceTemplate& t : templates)
    {
        const std::string what = "space template '" + t.name + "': ";
        if (t.name.empty()) SAIGA_EXIT_ERROR("WorldConfig::spaceTemplates: a space template has an empty name");
        if (t.name.size() > 256) SAIGA_EXIT_ERROR(what + "the name is longer than 256 bytes");
        if (!names.insert(t.name).second) SAIGA_EXIT_ERROR(what + "the name is used by another space template");
        if (t.kind == SpaceKind::Enclosed)
        {
            if (t.hasTerrain())
                SAIGA_EXIT_ERROR(what +
                                 "an enclosed space has no terrain (no prototypeHeight; use "
                                 "SpaceTemplate::enclosed(name, sizeChunks))");
            if (t.sizeChunks < 1 || t.sizeChunks > 512)
                SAIGA_EXIT_ERROR(what + "sizeChunks of an enclosed space must be in 1..512 (is " +
                                 std::to_string(t.sizeChunks) + ")");
            continue;
        }
        if (t.kind != SpaceKind::OpenWorld) SAIGA_EXIT_ERROR(what + "unknown kind " + std::to_string(int(t.kind)));
        if (t.baseSpacing != 1 && t.baseSpacing != 2)
            SAIGA_EXIT_ERROR(what + "baseSpacing must be 1 or 2 (is " + std::to_string(t.baseSpacing) + ")");
        if (t.sizeChunks % 16 != 0 || t.sizeChunks < 16 || t.sizeChunks > 512)
            SAIGA_EXIT_ERROR(what + "sizeChunks of an open-world space must be a multiple of 16 in 16..512 (is " +
                             std::to_string(t.sizeChunks) + ")");
        if (t.seaLevel && (*t.seaLevel > kTerrainHeightLimit || *t.seaLevel < -kTerrainHeightLimit))
            SAIGA_EXIT_ERROR(what + "seaLevel beyond kTerrainHeightLimit (" + std::to_string(*t.seaLevel) + ")");
    }
}

}  // namespace Engine
}  // namespace Saiga
