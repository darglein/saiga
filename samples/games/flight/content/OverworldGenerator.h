/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// OverworldGenerator (docs/engine_design.md §12.1, §18.4, R-CONTENT-1; S23c): the flight demo's integer content
// generator "flight_overworld", version 1. Content code: integer only, linted (the ContentLint test scans the games'
// content/ directories). Everything here enters the committed hashes of data/flight/content_hashes.json.
//
// - Layout (after the live demo: land over the whole 32 km space): a land mask (Q16, land where > 0): a continent
//   fbm of 8 km cells (3 octaves) plus a bias towards land; around the circuit centre (the centre of the space) at
//   least 30000 within 2.9 km (full land under the circuit, the respawn point and the fixed ruin), fading over
//   2.9-4.5 km (lakes and mountains from about 3.4 km on stay in view of the circuit); a 1 km coast detail; the space
//   edge falls to the ocean over its last 1200 m. Coasts, islands, bays and inland seas come from the mask. Mountain
//   ranges (rangeWeight, per sample): the zero band of an 8 km fbm, faded along its length, inland only, none within
//   3.3 km of the circuit centre (full beyond 4.3 km). The natural height in this order: below the mask's zero the sea
//   floor (-4 m at the coast to -100 m); on land 12 m + an inland plateau of up to 43 m (the mask) + hills of +-60 m +
//   the range weight times (120 m + ridged peaks of up to 900 m), clamped to [sea level + 1 m, 1200 m], blended over
//   the shore band of the mask to the coast floor (-4 m), then the river valleys and lakes towards 3 m below the sea
//   level (faded out by the range weight). The result lies in [-100 m, 1200 m] (asserted).
// - Plan grids: `biomes` (256 m cells, the vegetation value Q16 at most 12000, bilinear between the cell
//   centres: meadow or forest; biomeAt raises it by the range weight towards mountain) and `rivers` (64 m cells, the
//   valley weight Q16: 65536 on a river or lake cell, falling linearly to 0 over 7 cells; rivers walk from the
//   foothills down the continent to the sea, join other rivers, stop before a range or end in a lake in a basin; more
//   lakes on a 4 km grid in the lowlands).
// - Slots: spruce and broadleaf (TombstoneOnEdit) share an 8 m jittered grid (one species per cell by the biome's
//   density), rock (ReSnapOnEdit) on a 32 m grid; keptCount halves per level.
// - Grass: from the natural heights on a 4 m lattice (height above the sea, below 700 m, slope), the density by biome.
// - Sites (prefab "ruin", an 80 m footprint): the fixed ruin (key 1) in front of the demo's screenshot point (its
//   ground is checked in every build), grid ruins (keys 100 + cell) on a 2 km grid over the whole space where the
//   ground is dry and flat, the footprint fits and the origin is at least 300 m away from the circuit line.

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
constexpr int32_t kFlightMountainHeight = 1200 * 1024;  // the highest land
constexpr int32_t kFlightLowestHeight   = -100 * 1024;
// The demo's circuit (S23c after the live demo: the centre of the 32 km space; S20's radius 0.3 * 8192 m), integer
// metres
constexpr int64_t kFlightCircuitCentre = 16384;
constexpr int64_t kFlightCircuitRadius = 2458;
// The fixed ruin (site key 1): in chunk (72, 70), about 830 m ahead of the plane at the --demo screenshot
constexpr int64_t kFlightFixedRuinX = 18544, kFlightFixedRuinZ = 18144;
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

    // The vegetation value at integer metres (Q16, bilinear over the biomes grid's cell centres; the grid must be
    // filled): meadow or forest, at most the mountain threshold's base
    static int32_t biomeValue(const Engine::WorldPlan& plan, int64_t x, int64_t z);
    // The biome value at integer metres: the vegetation value, raised by the mountain ranges (biomeOf's input)
    static int32_t biomeAt(const Engine::WorldPlan& plan, int64_t x, int64_t z);
    static Biome biomeOf(int32_t biomeValue);
    // The land mask at integer metres (Q16, land where > 0; the layout above)
    static int64_t landMask(const Engine::WorldPlan& plan, int64_t x, int64_t z);
    // The mountain range weight at integer metres (Q16) given the land mask there (0 in the sea)
    static int64_t rangeWeight(const Engine::WorldPlan& plan, int64_t x, int64_t z, int64_t mask);
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
