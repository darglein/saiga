/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// SpaceTemplate (docs/engine_design.md §7.9, §12.2, §12.4; S20, S21): the template of a space of the world
// (WorldConfig::spaceTemplates; the space instances of the world plan use them, S23a). S20 implements name, kind,
// baseSpacing, seaLevel and sizeChunks; S21 enclosed spaces (SpaceTemplate::enclosed, the site origin) and the
// background; S23a the generator and fullRadius; the other §7.9 fields come with their stages. validateSpaceTemplates
// checks a list (the World constructor; fatal in every build).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/math/TerrainSurface.h"
#include "saiga/core/math/Types.h"
#include "saiga/core/util/assert.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/engine/world/SpaceKind.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
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
    // The content generator of the space (§12.1; S23a): a GeneratorId of WorldSetup::generators; 0 is the engine's
    // FlatGenerator (flat, y = 0, no terrain drawn, no content; D1). S20's height hook is a HookGenerator
    GeneratorId generator = 0;
    // Prefab sites within this distance of a Player (metres) are stamped (§7.9, §12.3; S23a); scenario worlds lower
    // it to at most 128 m
    double fullRadius = 1000;
    // Streaming (§12.3; S24, worlds with WorldConfig::streaming): chunks within residencyRadius (0: fullRadius; else in
    // [simulationRadius, fullRadius]) of a streaming centre are required, within residencyRadius + prefetchMargin
    // requested, and they are unloaded beyond residencyRadius + prefetchMargin + unloadMargin of every centre; the
    // entities within simulationRadius of a player (by 32 m grid cells) are simulated. Scenario worlds lower
    // simulationRadius to at most 128 m and prefetchMargin to 0
    double residencyRadius  = 0;
    double simulationRadius = 384;
    double prefetchMargin   = 256;
    double unloadMargin     = 128;
    // The clear color of an enclosed space (linear RGB; open-world spaces draw the sky). S36's ambient / fog replace it
    Vec3 background = Vec3(0.02, 0.02, 0.025);
    // The lighting of an enclosed space (S36, the environment extract; open-world spaces have the sky): its ambient
    // (lux-equivalent linear colour, no sun or sky) and its EV100 (a room of about 50 lux)
    Vec3 ambient = Vec3(50, 50, 50);
    double ev100 = 6;

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

    // The size of the space in metres (sizeChunks * 256)
    double sizeMeters() const { return double(sizeChunks) * chunkSize; }
    // The streaming distances (S24, G3): required (residencyRadius, 0: fullRadius), requested (+ prefetchMargin) and
    // kept (+ unloadMargin), measured from a centre to the nearest point of a chunk's square in x/z
    double requiredDistance() const { return residencyRadius > 0 ? residencyRadius : fullRadius; }
    double requestedDistance() const { return requiredDistance() + prefetchMargin; }
    double keptDistance() const { return requestedDistance() + unloadMargin; }
};

// The checks of WorldConfig::spaceTemplates (D1; S21): SAIGA_EXIT_ERROR (every build) with a message naming the
// template and the field. At most 65535 templates with unique non-empty names (at most 256 bytes); an enclosed
// template has 1..512 chunks (its seaLevel and baseSpacing are ignored; a generator with terrain is refused by the
// generation pipeline); an open-world template has spacing 1 or 2, a multiple of 16 chunks in 16..512 and a seaLevel
// within kTerrainHeightLimit. S24: every streaming radius and margin finite, fullRadius and simulationRadius positive,
// the margins not negative; with streaming also simulationRadius <= fullRadius and residencyRadius 0 or in
// [simulationRadius, fullRadius]
inline void validateSpaceTemplates(const std::vector<SpaceTemplate>& templates, bool streaming = false)
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
        if (!std::isfinite(t.ev100) || !std::isfinite(t.ambient(0)) || !std::isfinite(t.ambient(1)) ||
            !std::isfinite(t.ambient(2)) || t.ambient.minCoeff() < 0)
            SAIGA_EXIT_ERROR(what + "ambient must be finite and not negative, ev100 finite");
        // S24: the streaming distances
        const std::pair<const char*, double> radii[] = {{"fullRadius", t.fullRadius},
                                                        {"residencyRadius", t.residencyRadius},
                                                        {"simulationRadius", t.simulationRadius},
                                                        {"prefetchMargin", t.prefetchMargin},
                                                        {"unloadMargin", t.unloadMargin}};
        for (const auto& r : radii)
            if (!std::isfinite(r.second)) SAIGA_EXIT_ERROR(what + r.first + " must be finite");
        if (!(t.fullRadius > 0)) SAIGA_EXIT_ERROR(what + "fullRadius must be positive");
        if (!(t.simulationRadius > 0)) SAIGA_EXIT_ERROR(what + "simulationRadius must be positive");
        // The order of the radii only matters where they are used (a world without streaming keeps S23b's templates
        // valid, e.g. a fullRadius below the default simulationRadius)
        if (streaming && t.simulationRadius > t.fullRadius)
            SAIGA_EXIT_ERROR(what + "simulationRadius must be in (0, fullRadius] (is " +
                             std::to_string(t.simulationRadius) + ")");
        if (streaming && t.residencyRadius != 0 &&
            (t.residencyRadius < t.simulationRadius || t.residencyRadius > t.fullRadius))
            SAIGA_EXIT_ERROR(what + "residencyRadius must be 0 or in [simulationRadius, fullRadius] (is " +
                             std::to_string(t.residencyRadius) + ")");
        if (t.prefetchMargin < 0) SAIGA_EXIT_ERROR(what + "prefetchMargin must not be negative");
        if (t.unloadMargin < 0) SAIGA_EXIT_ERROR(what + "unloadMargin must not be negative");
        if (t.kind == SpaceKind::Enclosed)
        {
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
