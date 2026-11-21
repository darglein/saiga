/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Generator (docs/engine_design.md §12.1, §15.3, R-CONTENT-1; S23a): the interface of a content generator. A game
// registers its generators in WorldSetup::generators; a space template names one by GeneratorId (0: the engine's
// FlatGenerator). All functions are const, pure and called concurrently (streaming pool, JobPool): a generator keeps
// no mutable state. Integer only (content code: the ContentLint test scans this header).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/content/WorldPlan.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
// Per slot (§12.2, §17.9): exactly one of ReSnapOnEdit / TombstoneOnEdit
enum SlotFlags : uint32_t
{
    ReSnapOnEdit    = 1,
    TombstoneOnEdit = 2,
    Cosmetic        = 4
};

// A placement slot (tier-1 instances): name and flags are content (saved in GENR, checked on clients); the meshes map
// a candidate's meshSlot to mesh names for rendering only (presentation: not hashed, not saved)
struct GeneratorSlot
{
    std::string name;
    uint32_t flags = ReSnapOnEdit;
    std::vector<std::string> meshes;
};

// A tier-1 candidate of a chunk
struct Candidate
{
    uint16_t meshSlot = 0;
    int32_t x = 0, z = 0;   // 1/256 m, chunk relative, [0, 65536)
    uint16_t yaw   = 0;     // 65536 = one turn
    uint16_t scale = 1024;  // 1/1024
    uint32_t flags = 0;     // game bits: hashed, not interpreted
};

// A grass mask (S20's GrassMaskData layout): one texel per base cell (256 / baseSpacing per side), row-major z then x
struct GrassMask
{
    int32_t size = 0;
    std::vector<uint8_t> density;
};

class Generator
{
   public:
    virtual ~Generator() = default;
    // Saves, messages and the hash file name generators by name
    virtual const char* name() const = 0;
    // The newest version; every older one stays callable (§15.3)
    virtual uint16_t version() const = 0;
    // Append-only across versions
    virtual std::vector<GeneratorSlot> slots() const = 0;
    // False only for the float prototype fixture (HookGenerator "prototype")
    virtual bool integerOnly() const { return true; }
    // Heights equal the pre-S23 hook of its template: no slots, no sites, no generation state (GENR)
    virtual bool legacyTerrain() const { return false; }
    // False: flat at 0, no terrain drawn or streamed (E6: World::spaceHasTerrain)
    virtual bool hasTerrain() const { return true; }
    // In/out: the pipeline passes a plan with seed and space set (everything else empty); plan() fills sites, spaces
    // and grids and must not change seed or space (asserted). It may call height(version, plan, x, z) for natural
    // heights (site slopes, water tests, baseHeight) once the grids height() reads are filled.
    virtual void plan(uint16_t version, WorldPlan& plan) const = 0;
    // The natural terrain at integer metres in height units, without site patches (the pipeline composes them); reads
    // plan.seed, plan.space and the plan's grids. Within +-kTerrainHeightLimit (asserted by the pipeline)
    virtual int32_t height(uint16_t version, const WorldPlan& plan, int64_t x, int64_t z) const = 0;
    // Candidate `counter` of a slot in a chunk (false: none). Accepts a plan of any version <= version (§15.3)
    virtual bool candidate(uint16_t version, const WorldPlan& plan, ChunkId chunk, uint32_t slot, uint32_t counter,
                           Candidate& out) const             = 0;
    virtual uint32_t candidatesPerChunk(uint32_t slot) const = 0;
    // The candidates kept at a far level (S27); keptCount(slot, 0) == candidatesPerChunk(slot), non-increasing
    virtual uint32_t keptCount(uint32_t slot, int level) const = 0;
    // The chunk's grass mask (size 256 / baseSpacing)
    virtual void grassMask(uint16_t version, const WorldPlan& plan, ChunkId chunk, GrassMask& out) const = 0;
};

}  // namespace Engine
}  // namespace Saiga
