/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// OverworldGenerator (docs/engine_design.md §12.1, §18.4, R-CONTENT-1; S23c): the flight demo's integer content
// generator "flight_overworld", version 1. Content code: integer only, linted (the ContentLint test scans the games'
// content/ directories). Everything here enters the committed hashes of data/flight/content_hashes.json.
//
// - Layout: a land disc around the demo's circuit centre (4096, 4096) with a noisy coast radius in [4500, 6500] m, the
//   rest of the 32 km space is ocean. The natural height in this order: land base (20 m + hills of +-60 m + ridged
//   mountains of 0..400 m weighted by the biome value), clamped to [sea level + 1 m, 520 m], the coastal falloff to
//   the coast floor (-40 m) over the last 500 m and the sea floor beyond the coast (-40 m to -100 m over 1 km), the
//   river channels carved below the sea level; the result lies in [-100 m, 520 m] (asserted).
// - Plan grids: `biomes` (256 m cells, the biome value fbm Q16, bilinear between the cell centres: meadow, forest,
//   mountain) and `rivers` (64 m cells, 0 / 1: two channels from the mountains to the coast, bilinear banks).
// - Slots: spruce and broadleaf (TombstoneOnEdit) share an 8 m jittered grid (one species per cell by the biome's
//   density), rock (ReSnapOnEdit) on a 32 m grid; keptCount halves per level.
// - Grass: from the natural heights on a 4 m lattice (height above the sea, below 400 m, slope), the density by biome.
// - Sites (prefab "ruin", an 80 m footprint): the fixed ruin (key 1) in front of the demo's screenshot point (its
//   ground is checked in every build), grid ruins (keys 100 + cell) on a 2 km grid over [0, 10240)^2 where the ground
//   is dry and flat, the footprint fits and the origin is at least 300 m away from the circuit line.

#pragma once

#include "saiga/engine/content/Generator.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Saiga
{
namespace Game
{
// The generator's constants (content; height units of 1/1024 m)
constexpr int32_t kFlightSeaLevel       = 6 * 1024;  // equal to S20's kPrototypeSeaLevel (static_assert in FlightWorld)
constexpr int32_t kFlightMountainHeight = 520 * 1024;  // the highest land (kPrototypeMountainHeight)
constexpr int32_t kFlightLowestHeight   = -100 * 1024;
// The demo's circuit (S20: centre (4096, 4096), radius 0.3 * 8192 m), integer metres
constexpr int64_t kFlightCircuitCentre = 4096;
constexpr int64_t kFlightCircuitRadius = 2458;
// The fixed ruin (site key 1): the centre of chunk (24, 22), ahead of the plane at the --demo screenshot
constexpr int64_t kFlightFixedRuinX = 6304, kFlightFixedRuinZ = 5760;
constexpr uint32_t kFlightFixedRuinKey = 1, kFlightGridRuinKey = 100;
// The prefab of the ruin sites and its half size (footprint [-40, -40, 40, 40])
constexpr const char* kFlightRuinPrefab = "ruin";
constexpr int32_t kFlightRuinHalf       = 40;

class OverworldGenerator final : public Engine::Generator
{
   public:
    // The slots (the order is the save's; append-only)
    enum Slot : uint32_t
    {
        Spruce    = 0,
        Broadleaf = 1,
        Rock      = 2
    };
    // The biomes of the biome value
    enum Biome : int32_t
    {
        Meadow   = 0,
        Forest   = 1,
        Mountain = 2
    };

    const char* name() const override { return "flight_overworld"; }
    uint16_t version() const override { return 1; }
    std::vector<Engine::GeneratorSlot> slots() const override;
    void plan(uint16_t version, Engine::WorldPlan& plan) const override;
    int32_t height(uint16_t version, const Engine::WorldPlan& plan, int64_t x, int64_t z) const override;
    bool candidate(uint16_t version, const Engine::WorldPlan& plan, ChunkId chunk, uint32_t slot, uint32_t counter,
                   Engine::Candidate& out) const override;
    uint32_t candidatesPerChunk(uint32_t slot) const override { return slot == Rock ? 64 : 1024; }
    uint32_t keptCount(uint32_t slot, int level) const override
    {
        return level >= 16 ? 0 : candidatesPerChunk(slot) >> level;
    }
    void grassMask(uint16_t version, const Engine::WorldPlan& plan, ChunkId chunk,
                   Engine::GrassMask& out) const override;

    // The biome value at integer metres (Q16, bilinear over the biomes grid's cell centres; the grid must be filled)
    static int32_t biomeValue(const Engine::WorldPlan& plan, int64_t x, int64_t z);
    static Biome biomeOf(int32_t biomeValue);
    // The coast radius at integer metres in 1/256 m: 5500 m + fbm * 1000 m, in [4500, 6500] m
    static int64_t coastRadius256(uint64_t seed, int64_t x, int64_t z);
    // The distance from the circuit centre in 1/256 m
    static int64_t centreDistance256(int64_t x, int64_t z);
    // The max - min of the natural heights at a site's centre and its footprint's four corners (half size)
    int32_t siteSlope(uint16_t version, const Engine::WorldPlan& plan, int64_t x, int64_t z, int32_t half) const;
    // The grids height() reads (biomes, then rivers); plan() calls it first, then places the sites
    void planGrids(Engine::WorldPlan& plan) const;

   private:
    void planRivers(Engine::WorldPlan& plan) const;
    void planSites(uint16_t version, Engine::WorldPlan& plan) const;
};

}  // namespace Game
}  // namespace Saiga
