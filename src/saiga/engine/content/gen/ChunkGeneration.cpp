/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/gen/ChunkGeneration.h"

#include "saiga/core/math/TerrainSurface.h"
#include "saiga/core/util/assert.h"

#include <string>

namespace Saiga
{
namespace Engine
{
namespace
{
void checkChunk(const ChunkInputs& in, const ChunkId& chunk)
{
    SAIGA_ASSERT(in.generator && in.plan && in.sites, "chunk generation: incomplete inputs");
    SAIGA_ASSERT(chunk.space == in.space, "chunk generation: the chunk is not in this space");
    SAIGA_ASSERT(
        chunk.x >= 0 && chunk.z >= 0 && chunk.x < in.plan->space.sizeChunks && chunk.z < in.plan->space.sizeChunks,
        "chunk generation: the chunk (" + std::to_string(chunk.x) + ", " + std::to_string(chunk.z) +
            ") lies outside the space [0, " + std::to_string(in.plan->space.sizeChunks) + ")^2");
}

int32_t spacingOf(const ChunkInputs& in)
{
    return in.plan->space.baseSpacing == 1 ? 1 : 2;
}
}  // namespace

ChunkBase generateChunkBase(const ChunkInputs& in, const ChunkId& chunk)
{
    checkChunk(in, chunk);
    ChunkBase b;
    b.id = chunk;
    if (!in.openWorld)
    {
        b.heightsHash = hashHeights(nullptr, 0, 0);
        return b;
    }
    const int32_t spacing = spacingOf(in);
    const int32_t n       = 256 / spacing + 1;
    auto heights          = std::make_shared<std::vector<int32_t>>(uint32_t(n) * uint32_t(n));
    const int64_t x0 = int64_t(chunk.x) * 256, z0 = int64_t(chunk.z) * 256;
    for (int32_t j = 0; j < n; ++j)
        for (int32_t i = 0; i < n; ++i)
            (*heights)[uint32_t(j * n + i)] =
                composedHeight(*in.generator, in.pinnedVersion, *in.plan, in.space, *in.sites,
                               x0 + int64_t(i) * spacing, z0 + int64_t(j) * spacing);
    b.samplesPerSide = n;
    b.baseSpacing    = spacing;
    b.heightsHash    = hashHeights(heights.get(), n, spacing);
    b.heights        = std::move(heights);
    return b;
}

ChunkPlacement generateChunkPlacement(const ChunkInputs& in, const ChunkBase& base, uint16_t placementVersion,
                                      uint64_t stampsHash, uint64_t staticsHash)
{
    checkChunk(in, base.id);
    SAIGA_ASSERT(placementVersion >= in.pinnedVersion, "chunk generation: the placement version is below the pin");
    const ChunkId& chunk = base.id;
    ChunkPlacement p;
    p.id               = chunk;
    p.pinnedVersion    = in.pinnedVersion;
    p.placementVersion = placementVersion;

    const Generator& g    = *in.generator;
    const int32_t spacing = spacingOf(in);
    const int64_t cell    = int64_t(spacing) * kPosUnitsPerMetre;  // a base cell in 1/256 m
    const int fracBits    = spacing == 1 ? 8 : 9;
    const int64_t ox = int64_t(chunk.x) * 256 * kPosUnitsPerMetre, oz = int64_t(chunk.z) * 256 * kPosUnitsPerMetre;
    const uint32_t slots = uint32_t(g.slots().size());
    for (uint32_t slot = 0; slot < slots; ++slot)
    {
        const uint32_t count = g.candidatesPerChunk(slot);
        for (uint32_t counter = 0; counter < count; ++counter)
        {
            Candidate c;
            if (!g.candidate(placementVersion, *in.plan, chunk, slot, counter, c)) continue;
            if (c.x < 0 || c.z < 0 || c.x >= 65536 || c.z >= 65536)
                SAIGA_EXIT_ERROR(std::string("generator '") + g.name() + "': candidate " + std::to_string(counter) +
                                 " of slot " + std::to_string(slot) + " lies outside its chunk");
            if (excluded(*in.plan, in.space, *in.sites, ox + c.x, oz + c.z)) continue;
            TierOneInstance t;
            t.key.generatorSlot = slot;
            t.key.counter       = counter;
            t.meshSlot          = c.meshSlot;
            t.x                 = c.x;
            t.z                 = c.z;
            t.yaw               = c.yaw;
            t.scale             = c.scale;
            t.flags             = c.flags;
            if (base.heights)
            {
                const int32_t n               = base.samplesPerSide;
                const int64_t i               = c.x / cell;
                const int64_t j               = c.z / cell;
                const int64_t fx              = c.x - i * cell;
                const int64_t fz              = c.z - j * cell;
                const std::vector<int32_t>& h = *base.heights;
                const uint32_t i00            = uint32_t(j * n + i);
                t.y = terrainSurfaceUnits(h[i00], h[i00 + 1], h[i00 + uint32_t(n)], h[i00 + uint32_t(n) + 1], fx, fz,
                                          fracBits);
            }
            p.instances.push_back(t);
        }
    }

    g.grassMask(placementVersion, *in.plan, chunk, p.grass);
    if (p.grass.size != 256 / spacing ||
        uint32_t(p.grass.density.size()) != uint32_t(p.grass.size) * uint32_t(p.grass.size))
        SAIGA_EXIT_ERROR(std::string("generator '") + g.name() + "': the grass mask must have " +
                         std::to_string(256 / spacing) + "^2 cells");
    // Cells whose centre lies in an exclusion rectangle
    for (int32_t j = 0; j < p.grass.size; ++j)
        for (int32_t i = 0; i < p.grass.size; ++i)
            if (excluded(*in.plan, in.space, *in.sites, ox + int64_t(i) * cell + cell / 2,
                         oz + int64_t(j) * cell + cell / 2))
                p.grass.density[uint32_t(j * p.grass.size + i)] = 0;

    p.hashes.heights = base.heightsHash;
    p.hashes.tier1   = hashTierOne(p.instances);
    p.hashes.grass   = hashGrass(p.grass);
    p.hashes.stamps  = stampsHash;
    p.hashes.statics = staticsHash;
    p.hashes.total   = hashTotal(chunk, in.pinnedVersion, placementVersion, p.hashes);
    return p;
}

}  // namespace Engine
}  // namespace Saiga
