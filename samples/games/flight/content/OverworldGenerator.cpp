/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The flight demo's integer overworld generator (S23c): layout, heights, rivers, biomes, slots, grass and the ruin
// sites. Content code (R-CONTENT-1, linted): integer only, the toolkit of engine/content/gen/.

#include "OverworldGenerator.h"

#include "saiga/core/util/assert.h"
#include "saiga/engine/content/gen/FixedMath.h"
#include "saiga/engine/content/gen/IntNoise.h"
#include "saiga/engine/content/gen/SitePatch.h"
#include "saiga/engine/content/gen/TrigTable.h"

#include <algorithm>
#include <string>

namespace Saiga
{
namespace Game
{
using namespace Engine;

namespace
{
constexpr int64_t kMetre = kHeightUnitsPerMetre;  // height units
constexpr int64_t kPos   = kPosUnitsPerMetre;     // 1/256 m

// The noise seeds of the plan seed (one per purpose)
enum SeedTag : uint64_t
{
    TagCoast = 1,
    TagHills,
    TagMountains,
    TagBiomes,
    TagRivers,
    TagSites,
    TagTrees,
    TagRocks,
    TagGrass,
    TagClumps
};
uint64_t seedOf(uint64_t planSeed, SeedTag tag)
{
    return contentHash64(planSeed, 0xF11647, uint64_t(tag), 0, 0);
}

// The biome value thresholds (Q16): meadow below, mountain above; the mountain weight rises over [lo, hi]
constexpr int64_t kMeadowBelow = -9000, kMountainAbove = 22000;
constexpr int64_t kMountainLo = 12000, kMountainHi = 40000;

// The biomes grid: 256 m cells over the space; the rivers grid: 64 m cells over [0, 10752)^2 (the land disc)
constexpr int32_t kBiomeCell = 256, kRiverCell = 64, kRiverCells = 168;

int64_t absI(int64_t v)
{
    return v < 0 ? -v : v;
}

// The mountain weight of a biome value (Q16)
int64_t mountainWeight(int64_t b)
{
    if (b <= kMountainLo) return 0;
    if (b >= kMountainHi) return 65536;
    return smoothQ16((b - kMountainLo) * 65536 / (kMountainHi - kMountainLo));
}

// The river weight at integer metres (Q16): the 0 / 1 cells bilinear between their centres, doubled (full carving
// within about half a cell of a channel) and smoothed; 0 outside the grid
int64_t riverWeight(const WorldPlan& plan, int64_t x, int64_t z)
{
    const PlanGrid& g = plan.rivers;
    if (g.values.empty()) return 0;
    const int64_t u = x - kRiverCell / 2, v = z - kRiverCell / 2;
    const int64_t cx = floorDiv(u, kRiverCell), cz = floorDiv(v, kRiverCell);
    if (cx < -1 || cz < -1 || cx >= g.width || cz >= g.height) return 0;
    const int64_t fx = floorMod(u, kRiverCell), fz = floorMod(v, kRiverCell);
    auto cell = [&](int64_t i, int64_t k) -> int64_t
    {
        if (i < 0 || k < 0 || i >= g.width || k >= g.height) return 0;
        return g.values[uint32_t(k * g.width + i)] != 0 ? 1 : 0;
    };
    const int64_t sum = cell(cx, cz) * (kRiverCell - fx) * (kRiverCell - fz) +
                        cell(cx + 1, cz) * fx * (kRiverCell - fz) + cell(cx, cz + 1) * (kRiverCell - fx) * fz +
                        cell(cx + 1, cz + 1) * fx * fz;
    if (sum == 0) return 0;
    // sum in [0, 4096]: Q16 is sum * 16, doubled
    return smoothQ16(std::min<int64_t>(65536, sum * 32));
}

uint32_t gridIndex(const PlanGrid& g, int64_t x, int64_t z)
{
    return uint32_t(z * g.width + x);
}

}  // namespace

std::vector<GeneratorSlot> OverworldGenerator::slots() const
{
    return {{"spruce", TombstoneOnEdit, {"flight/spruce"}},
            {"broadleaf", TombstoneOnEdit, {"flight/broadleaf"}},
            {"rock", ReSnapOnEdit, {"flight/rock"}}};
}

int64_t OverworldGenerator::centreDistance256(int64_t x, int64_t z)
{
    const int64_t dx = x - kFlightCircuitCentre, dz = z - kFlightCircuitCentre;
    return int64_t(isqrt64(uint64_t(dx * dx + dz * dz) << 16));
}

int64_t OverworldGenerator::coastRadius256(uint64_t seed, int64_t x, int64_t z)
{
    return 5500 * kPos + mulShift(fbm(seed, x, z, 11, 3), 1000 * kPos, 16);
}

int32_t OverworldGenerator::biomeValue(const WorldPlan& plan, int64_t x, int64_t z)
{
    const PlanGrid& g = plan.biomes;
    const int64_t u = x - kBiomeCell / 2, v = z - kBiomeCell / 2;
    const int64_t cx = floorDiv(u, kBiomeCell), cz = floorDiv(v, kBiomeCell);
    const int64_t fx = floorMod(u, kBiomeCell), fz = floorMod(v, kBiomeCell);
    const int64_t top = int64_t(g.at(cx, cz)) * (kBiomeCell - fx) + int64_t(g.at(cx + 1, cz)) * fx;
    const int64_t bot = int64_t(g.at(cx, cz + 1)) * (kBiomeCell - fx) + int64_t(g.at(cx + 1, cz + 1)) * fx;
    return int32_t(floorShift(top * (kBiomeCell - fz) + bot * fz, 16));
}

OverworldGenerator::Biome OverworldGenerator::biomeOf(int32_t b)
{
    if (b < kMeadowBelow) return Meadow;
    if (b >= kMountainAbove) return Mountain;
    return Forest;
}

int32_t OverworldGenerator::height(uint16_t, const WorldPlan& plan, int64_t x, int64_t z) const
{
    const int64_t sea = kFlightSeaLevel;
    const int64_t d   = centreDistance256(x, z);
    int64_t h         = 0;
    if (d >= 7500 * kPos)
    {
        // Beyond the largest coast radius + 1 km: the open sea floor
        h = -100 * kMetre;
    }
    else
    {
        const int64_t coast = coastRadius256(seedOf(plan.seed, TagCoast), x, z);
        if (d >= coast)
        {
            // 3. The sea floor: -40 m at the coast to -100 m 1 km beyond it
            const int64_t t = std::min<int64_t>(d - coast, 1000 * kPos) * 65536 / (1000 * kPos);
            h               = lerpQ16(-40 * kMetre, -100 * kMetre, t);
        }
        else
        {
            // 1. The land base: 20 m, hills of +-60 m, ridged mountains of 0..400 m weighted by the biome value
            int64_t land    = 20 * kMetre + mulShift(fbm(seedOf(plan.seed, TagHills), x, z, 10, 6), 60 * kMetre, 16);
            const int64_t w = mountainWeight(biomeValue(plan, x, z));
            if (w > 0)
            {
                const int64_t r = ridged(seedOf(plan.seed, TagMountains), x, z, 12, 5);
                land += mulShift(mulShift(floorShift(r * r, 16), 400 * kMetre, 16), w, 16);
            }
            // 2. Clamped to [sea level + 1 m, 520 m]
            land = clampI<int64_t>(land, sea + kMetre, kFlightMountainHeight);
            // 3. The coastal falloff over the last 500 m to the coast floor
            if (d >= coast - 500 * kPos)
                h = lerpQ16(land, -40 * kMetre, (d - (coast - 500 * kPos)) * 65536 / (500 * kPos));
            else
                h = land;
        }
    }
    // 4. The river channels below the sea level (a no-op in the sea, which lies below)
    const int64_t river = riverWeight(plan, x, z);
    if (river > 0) h = lerpQ16(h, std::min<int64_t>(h, sea - 2 * kMetre), river);
    // 5. The generator's own range
    SAIGA_ASSERT(h >= kFlightLowestHeight && h <= kFlightMountainHeight, "flight_overworld: height out of range");
    return int32_t(h);
}

int32_t OverworldGenerator::siteSlope(uint16_t version, const WorldPlan& plan, int64_t x, int64_t z, int32_t half) const
{
    const int32_t c = height(version, plan, x, z);
    int32_t lo = c, hi = c;
    for (int k = 0; k < 4; ++k)
    {
        const int32_t h = height(version, plan, x + ((k & 1) ? half : -half), z + ((k & 2) ? half : -half));
        lo              = std::min(lo, h);
        hi              = std::max(hi, h);
    }
    return hi - lo;
}

void OverworldGenerator::planRivers(WorldPlan& plan) const
{
    PlanGrid& g = plan.rivers;
    g.cellSize  = kRiverCell;
    g.originX   = 0;
    g.originZ   = 0;
    g.width     = kRiverCells;
    g.height    = kRiverCells;
    g.values.assign(uint32_t(kRiverCells) * uint32_t(kRiverCells), 0);

    // The sources: the highest biome value of the biome cells within 3500 m of the circuit centre, and the highest at
    // least 2500 m away from the first
    const PlanGrid& b = plan.biomes;
    int64_t src[2][2] = {{-1, -1}, {-1, -1}};
    for (int s = 0; s < 2; ++s)
    {
        int64_t best = -(int64_t(1) << 40);
        for (int64_t cz = 0; cz < b.height; ++cz)
            for (int64_t cx = 0; cx < b.width; ++cx)
            {
                const int64_t x = cx * kBiomeCell + kBiomeCell / 2, z = cz * kBiomeCell + kBiomeCell / 2;
                if (centreDistance256(x, z) >= 3500 * kPos) continue;
                if (s == 1)
                {
                    const int64_t dx = x - src[0][0], dz = z - src[0][1];
                    if (dx * dx + dz * dz < 2500 * 2500) continue;
                }
                const int64_t v = b.values[gridIndex(b, cx, cz)];
                if (v > best)
                {
                    best      = v;
                    src[s][0] = x;
                    src[s][1] = z;
                }
            }
    }

    const uint64_t seed   = seedOf(plan.seed, TagRivers);
    const uint64_t coast  = seedOf(plan.seed, TagCoast);
    const int64_t size256 = int64_t(plan.space.sizeChunks) * 256 * kPos;
    for (int s = 0; s < 2; ++s)
    {
        if (src[s][0] < 0) continue;
        // Walk outward in 32 m steps, the outward direction turned by up to +-40 degrees by a noise, until 64 m beyond
        // the coast (1/256 m)
        int64_t px = src[s][0] * kPos, pz = src[s][1] * kPos;
        for (int step = 0; step < 512; ++step)
        {
            if (px < 0 || pz < 0 || px >= size256 || pz >= size256) break;
            const int64_t mx = floorDiv(px, kPos), mz = floorDiv(pz, kPos);
            const int64_t cx = floorDiv(mx, kRiverCell), cz = floorDiv(mz, kRiverCell);
            if (cx < kRiverCells && cz < kRiverCells) g.values[gridIndex(g, cx, cz)] = 1;
            const int64_t d = centreDistance256(mx, mz);
            if (d >= coastRadius256(coast, mx, mz) + 64 * kPos) break;
            // The outward unit vector (Q16)
            int64_t ox = 65536, oz = 0;
            if (d > 0)
            {
                ox = (px - kFlightCircuitCentre * kPos) * 65536 / d;
                oz = (pz - kFlightCircuitCentre * kPos) * 65536 / d;
            }
            // +-40 degrees: 40 / 360 * 2^17 = 14564 trig steps at |noise| = 65536
            const int64_t n      = gradientNoise(contentHash64(seed, uint64_t(s), 0, 0, 0), mx, mz, 9);
            const uint32_t angle = uint32_t(floorMod(floorDiv(n * 14564, 65536), int64_t(kTrigSteps)));
            const int64_t c = cosQ30(angle), sn = sinQ30(angle);
            const int64_t dx = floorShift(ox * c - oz * sn, 30), dz = floorShift(ox * sn + oz * c, 30);
            px += mulShift(dx, 32 * kPos, 16);
            pz += mulShift(dz, 32 * kPos, 16);
        }
    }
}

void OverworldGenerator::planSites(uint16_t version, WorldPlan& plan) const
{
    const int64_t size = int64_t(plan.space.sizeChunks) * 256;
    auto addRuin       = [&](uint32_t key, int64_t x, int64_t z, uint8_t q, int32_t baseHeight)
    {
        Site s;
        s.key           = key;
        s.prefab        = kFlightRuinPrefab;
        s.prefabVersion = 1;
        s.x             = x;
        s.z             = z;
        s.quarterTurns  = q;
        s.baseHeight    = baseHeight;
        s.footprint[0]  = -kFlightRuinHalf;
        s.footprint[1]  = -kFlightRuinHalf;
        s.footprint[2]  = kFlightRuinHalf;
        s.footprint[3]  = kFlightRuinHalf;
        plan.sites.push_back(s);
    };
    // Dry (2 m above the sea) and flat (8 m over the footprint) ground
    auto suitable = [&](int64_t x, int64_t z, int32_t& base)
    {
        base = height(version, plan, x, z);
        return base >= kFlightSeaLevel + 2 * kMetre && siteSlope(version, plan, x, z, kFlightRuinHalf) <= 8 * kMetre;
    };

    // The fixed ruin (exempt from the circuit rule): its ground is checked in every build
    int32_t base = 0;
    if (!suitable(kFlightFixedRuinX, kFlightFixedRuinZ, base))
        SAIGA_EXIT_ERROR(
            "flight_overworld: fixed ruin site on unsuitable ground: move it (natural height " + std::to_string(base) +
            ", slope " +
            std::to_string(siteSlope(version, plan, kFlightFixedRuinX, kFlightFixedRuinZ, kFlightRuinHalf)) + ")");
    addRuin(kFlightFixedRuinKey, kFlightFixedRuinX, kFlightFixedRuinZ, 0, base);

    // The grid ruins: one jittered candidate per 2 km cell of [0, 10240)^2
    const uint64_t seed = seedOf(plan.seed, TagSites);
    for (int64_t gz = 0; gz < 5; ++gz)
        for (int64_t gx = 0; gx < 5; ++gx)
        {
            const uint64_t h = contentHash64(seed, uint64_t(gx), uint64_t(gz), 0, 0);
            const int64_t x  = gx * 2048 + 1024 - 600 + int64_t(h % 1201);
            const int64_t z  = gz * 2048 + 1024 - 600 + int64_t((h >> 20) % 1201);
            const uint8_t q  = uint8_t((h >> 40) & 3);
            // Inside the space (the square footprint is the same in every rotation)
            if (x - kFlightRuinHalf < 0 || z - kFlightRuinHalf < 0 || x + kFlightRuinHalf > size ||
                z + kFlightRuinHalf > size)
                continue;
            // At least 300 m away from the circuit line
            if (absI(centreDistance256(x, z) - kFlightCircuitRadius * kPos) <= 300 * kPos) continue;
            // Apart from every site (closed bounds, as the pipeline tests them)
            bool overlaps = false;
            for (const Site& s : plan.sites)
            {
                int64_t r[4];
                rotateBounds(s.footprint, s.quarterTurns, r);
                if (x - kFlightRuinHalf <= s.x + r[2] && s.x + r[0] <= x + kFlightRuinHalf &&
                    z - kFlightRuinHalf <= s.z + r[3] && s.z + r[1] <= z + kFlightRuinHalf)
                    overlaps = true;
            }
            if (overlaps || !suitable(x, z, base)) continue;
            addRuin(kFlightGridRuinKey + uint32_t(gz * 5 + gx), x, z, q, base);
        }
}

void OverworldGenerator::planGrids(WorldPlan& plan) const
{
    // The biome values at the 256 m cell centres, then the rivers (they start at the highest biome values)
    PlanGrid& b = plan.biomes;
    b.cellSize  = kBiomeCell;
    b.originX   = 0;
    b.originZ   = 0;
    b.width     = plan.space.sizeChunks;
    b.height    = plan.space.sizeChunks;
    b.values.assign(uint32_t(b.width) * uint32_t(b.height), 0);
    const uint64_t biomeSeed = seedOf(plan.seed, TagBiomes);
    for (int64_t cz = 0; cz < b.height; ++cz)
        for (int64_t cx = 0; cx < b.width; ++cx)
            b.values[gridIndex(b, cx, cz)] =
                fbm(biomeSeed, cx * kBiomeCell + kBiomeCell / 2, cz * kBiomeCell + kBiomeCell / 2, 11, 2);
    planRivers(plan);
}

void OverworldGenerator::plan(uint16_t version, WorldPlan& plan) const
{
    SAIGA_ASSERT(plan.space.seaLevel && *plan.space.seaLevel == kFlightSeaLevel,
                 "flight_overworld: the template's sea level must be kFlightSeaLevel");
    // The grids height() reads first (S23a in/out plan), then the sites with natural heights
    planGrids(plan);
    planSites(version, plan);
}

bool OverworldGenerator::candidate(uint16_t version, const WorldPlan& plan, ChunkId chunk, uint32_t slot,
                                   uint32_t counter, Candidate& out) const
{
    const int64_t sea = kFlightSeaLevel;
    const int64_t cx = int64_t(chunk.x), cz = int64_t(chunk.z);
    if (slot == Spruce || slot == Broadleaf)
    {
        // One cell of the 8 m grid: one species (or none) by the biome's density, shared by both slots
        const uint64_t h = contentHash64(seedOf(plan.seed, TagTrees), uint64_t(cx), uint64_t(cz), counter, 0);
        const int64_t i = counter & 31, j = counter >> 5;
        const int64_t x  = i * 2048 + 256 + int64_t((h >> 16) % 1536);
        const int64_t z  = j * 2048 + 256 + int64_t((h >> 32) % 1536);
        const int64_t mx = cx * 256 + x / kPos, mz = cz * 256 + z / kPos;
        // The biome dithered by +-64 m (no straight borders along the grid cells)
        const int64_t jx = int64_t((h >> 48) & 127) - 64, jz = int64_t((h >> 56) & 127) - 64;
        const Biome biome = biomeOf(biomeValue(plan, mx + jx, mz + jz));
        // Q16 probabilities of spruce and broadleaf per biome, scaled by a 128 m clump noise (glades and groves: 0 to
        // twice the mean)
        static constexpr int64_t pSpruce[3]    = {900, 12000, 12000};
        static constexpr int64_t pBroadleaf[3] = {2600, 16000, 0};
        const int64_t clump = 65536 + int64_t(gradientNoise(seedOf(plan.seed, TagClumps), mx, mz, 7));
        const int64_t r     = int64_t(h & 0xFFFF);
        const int64_t ps = mulShift(pSpruce[biome], clump, 16), pb = mulShift(pBroadleaf[biome], clump, 16);
        if (slot == Spruce ? r >= ps : (r < ps || r >= ps + pb)) return false;
        // At least 1 m above the sea, below the tree line on mountains, not too steep (2 m steps)
        const int64_t g = height(version, plan, mx, mz);
        if (g < sea + kMetre || (biome == Mountain && g > 380 * kMetre)) return false;
        const int64_t gx = height(version, plan, mx + 2, mz), gz = height(version, plan, mx, mz + 2);
        if (absI(gx - g) + absI(gz - g) > 2560) return false;
        const uint64_t h2 = contentHash64(seedOf(plan.seed, TagTrees), uint64_t(cx), uint64_t(cz), counter, 1 + slot);
        out.meshSlot      = 0;
        out.x             = int32_t(x);
        out.z             = int32_t(z);
        out.yaw           = uint16_t(h2 & 0xFFFF);
        // The meshes are 1 unit high: the scale is the tree height (spruce 10-20 m, broadleaf 8-16 m)
        out.scale = slot == Spruce ? uint16_t(10240 + (h2 >> 16) % 10241) : uint16_t(8192 + (h2 >> 16) % 8193);
        out.flags = 0;
        return true;
    }
    if (slot == Rock)
    {
        // One candidate per 32 m cell
        const uint64_t h = contentHash64(seedOf(plan.seed, TagRocks), uint64_t(cx), uint64_t(cz), counter, 0);
        const int64_t i = counter & 7, j = counter >> 3;
        const int64_t x  = i * 8192 + 1024 + int64_t((h >> 16) % 6144);
        const int64_t z  = j * 8192 + 1024 + int64_t((h >> 32) % 6144);
        const int64_t mx = cx * 256 + x / kPos, mz = cz * 256 + z / kPos;
        static constexpr int64_t pRock[3] = {6000, 10000, 30000};
        if (int64_t(h & 0xFFFF) >= pRock[biomeOf(biomeValue(plan, mx, mz))]) return false;
        if (height(version, plan, mx, mz) < sea + kMetre / 2) return false;
        out.meshSlot = 0;
        out.x        = int32_t(x);
        out.z        = int32_t(z);
        out.yaw      = uint16_t((h >> 48) & 0xFFFF);
        // A 1 m mesh: 1-3 m
        out.scale = uint16_t(1024 + contentHash64(h, 1, 0, 0, 0) % 2049);
        out.flags = 0;
        return true;
    }
    return false;
}

void OverworldGenerator::grassMask(uint16_t version, const WorldPlan& plan, ChunkId chunk, GrassMask& out) const
{
    const int32_t spacing = plan.space.baseSpacing == 1 ? 1 : 2;
    const int32_t n       = 256 / spacing;
    out.size              = n;
    out.density.assign(uint32_t(n) * uint32_t(n), 0);
    // The natural heights on a 4 m lattice (65 x 65)
    constexpr int32_t step = 4, lattice = 256 / step + 1;
    const int64_t x0 = int64_t(chunk.x) * 256, z0 = int64_t(chunk.z) * 256;
    std::vector<int32_t> h(uint32_t(lattice) * uint32_t(lattice));
    for (int32_t j = 0; j < lattice; ++j)
        for (int32_t i = 0; i < lattice; ++i)
            h[uint32_t(j * lattice + i)] = height(version, plan, x0 + int64_t(i) * step, z0 + int64_t(j) * step);
    const int64_t sea                = kFlightSeaLevel;
    const uint64_t seed              = seedOf(plan.seed, TagGrass);
    static constexpr int64_t base[3] = {230, 150, 80};
    const int32_t perBlock           = step / spacing;
    for (int32_t bj = 0; bj < lattice - 1; ++bj)
        for (int32_t bi = 0; bi < lattice - 1; ++bi)
        {
            const int64_t h00 = h[uint32_t(bj * lattice + bi)], h10 = h[uint32_t(bj * lattice + bi + 1)];
            const int64_t h01 = h[uint32_t((bj + 1) * lattice + bi)], h11 = h[uint32_t((bj + 1) * lattice + bi + 1)];
            // Not steeper than 2.5 m over 4 m on an edge of the block
            const int64_t slope =
                std::max(std::max(absI(h10 - h00), absI(h01 - h00)), std::max(absI(h11 - h10), absI(h11 - h01)));
            if (slope > 2560) continue;
            const int64_t bx = x0 + int64_t(bi) * step + step / 2, bz = z0 + int64_t(bj) * step + step / 2;
            const int64_t d = base[biomeOf(biomeValue(plan, bx, bz))];
            for (int32_t k = 0; k < perBlock; ++k)
                for (int32_t l = 0; l < perBlock; ++l)
                {
                    // The height at the cell centre, bilinear in the block (Q8 weights)
                    const int64_t fx  = int64_t(l) * spacing * 64 + spacing * 32;
                    const int64_t fz  = int64_t(k) * spacing * 64 + spacing * 32;
                    const int64_t top = h00 * (256 - fx) + h10 * fx, bot = h01 * (256 - fx) + h11 * fx;
                    const int64_t hc = floorShift(top * (256 - fz) + bot * fz, 16);
                    if (hc < sea + kMetre / 2 || hc > 400 * kMetre) continue;
                    const int32_t ci = bi * perBlock + l, cj = bj * perBlock + k;
                    const int64_t v = valueNoise(seed, x0 + int64_t(ci) * spacing, z0 + int64_t(cj) * spacing, 4);
                    out.density[uint32_t(cj * n + ci)] = uint8_t(clampI<int64_t>(d + floorShift(d * v, 17), 1, 255));
                }
        }
}

}  // namespace Game
}  // namespace Saiga
