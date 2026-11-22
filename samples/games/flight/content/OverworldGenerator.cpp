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
    TagClumps,
    TagContinent,
    TagRanges,
    TagRangeStrength,
    TagLakes
};
uint64_t seedOf(uint64_t planSeed, SeedTag tag)
{
    return contentHash64(planSeed, 0xF11647, uint64_t(tag), 0, 0);
}

// The biome value thresholds (Q16): meadow below, mountain above; the vegetation value of the biomes grid is at most
// kMountainLo (never mountain), a range raises it towards 65536
constexpr int64_t kMeadowBelow = -9000, kMountainAbove = 22000;
constexpr int64_t kMountainLo = 12000;

// The biomes grid: 256 m cells over the space; the rivers grid: 64 m cells over the space
constexpr int32_t kBiomeCell = 256, kRiverCell = 64;

// The land mask (Q16): the bias towards land, the coast detail's amplitude, the shore band (the land height blends to
// the coast floor below it) and the mask below 0 over which the sea floor falls to -100 m
constexpr int64_t kLandBias = 12000, kCoastDetail = 9000, kShoreBand = 5000, kSeaSlope = 20000;
// The space edge falls to the ocean over its last kEdgeWidth metres (the mask lowered by up to kEdgePenalty)
constexpr int64_t kEdgeWidth = 1200, kEdgePenalty = 80000;
// The land under the circuit: the mask at least kCircuitFloor within kCircuitLand metres of the circuit centre, the
// floor fading to -65536 at kCircuitLandEnd
constexpr int64_t kCircuitLand = 3000, kCircuitLandEnd = 7000, kCircuitFloor = 30000;
// No mountain ranges, rivers or lakes near the circuit: none within kClearInner metres of its centre, full beyond
// kClearOuter
constexpr int64_t kClearInner = 4000, kClearOuter = 6000;
// The mountain range belts: |fbm| below kRangeWidth (Q16)
constexpr int64_t kRangeWidth = 20000;
// The river valleys: the chamfer distance (2 per cell orthogonal, 3 diagonal) over which the valley weight falls to 0
constexpr int32_t kValleyDistance = 14;
// The river walk: at most kMaxRivers sources, kRiverSpacing metres apart; a walk ends in a lake when the land mask did
// not fall for kRiverStall steps
constexpr int kMaxRivers = 24, kRiverStall = 40, kRiverSteps = 1500;
constexpr int64_t kRiverSpacing = 2500;

int64_t absI(int64_t v)
{
    return v < 0 ? -v : v;
}

// The river weight at integer metres (Q16): the valley weights of the cells bilinear between their centres; 0
// outside the grid
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
        return g.values[uint32_t(k * g.width + i)];
    };
    const int64_t sum = cell(cx, cz) * (kRiverCell - fx) * (kRiverCell - fz) +
                        cell(cx + 1, cz) * fx * (kRiverCell - fz) + cell(cx, cz + 1) * (kRiverCell - fx) * fz +
                        cell(cx + 1, cz + 1) * fx * fz;
    // The weights sum to 64 * 64
    return floorShift(sum, 12);
}

// The clearance around the circuit centre (Q16): 0 within kClearInner, 65536 beyond kClearOuter, smooth between
int64_t clearance(int64_t d256)
{
    if (d256 <= kClearInner * kPos) return 0;
    if (d256 >= kClearOuter * kPos) return 65536;
    return smoothQ16((d256 - kClearInner * kPos) * 65536 / ((kClearOuter - kClearInner) * kPos));
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

int64_t OverworldGenerator::landMask(const WorldPlan& plan, int64_t x, int64_t z)
{
    // The continent (8 km cells down to 2 km) with the bias towards land
    int64_t m = kLandBias + fbm(seedOf(plan.seed, TagContinent), x, z, 13, 3);
    // The land under the circuit (before the coast detail, so a coast near the circuit is not round)
    const int64_t dx = x - kFlightCircuitCentre, dz = z - kFlightCircuitCentre;
    if (absI(dx) < kCircuitLandEnd && absI(dz) < kCircuitLandEnd)
    {
        const int64_t d = centreDistance256(x, z);
        if (d < kCircuitLandEnd * kPos)
        {
            int64_t floor = kCircuitFloor;
            if (d > kCircuitLand * kPos)
                floor = lerpQ16(kCircuitFloor, -65536,
                                (d - kCircuitLand * kPos) * 65536 / ((kCircuitLandEnd - kCircuitLand) * kPos));
            m = std::max(m, floor);
        }
    }
    // The coast detail (1 km down to 256 m)
    m += mulShift(fbm(seedOf(plan.seed, TagCoast), x, z, 10, 3), kCoastDetail, 16);
    // The space edge falls to the ocean (the circuit lies more than kEdgeWidth inside the space)
    const int64_t size = int64_t(plan.space.sizeChunks) * 256;
    const int64_t edge = clampI<int64_t>(std::min(std::min(x, z), std::min(size - x, size - z)), 0, kEdgeWidth);
    if (edge < kEdgeWidth) m -= mulShift(smoothQ16((kEdgeWidth - edge) * 65536 / kEdgeWidth), kEdgePenalty, 16);
    return m;
}

int64_t OverworldGenerator::rangeWeight(const WorldPlan& plan, int64_t x, int64_t z, int64_t mask)
{
    // Inland only (rising from the shore over the mask's [0, 24000])
    if (mask <= 0) return 0;
    // The belt: the zero band of an 8 km fbm (long, winding lines)
    const int64_t n = absI(fbm(seedOf(plan.seed, TagRanges), x, z, 13, 2));
    if (n >= kRangeWidth) return 0;
    int64_t w = smoothQ16((kRangeWidth - n) * 65536 / kRangeWidth);
    // Faded along its length by a 4 km noise (ranges begin and end)
    const int64_t g = gradientNoise(seedOf(plan.seed, TagRangeStrength), x, z, 12);
    const int64_t s = clampI<int64_t>((g + 24000) * 65536 / 30000, 0, 65536);
    w               = mulShift(w, smoothQ16(s), 16);
    w               = mulShift(w, smoothQ16(std::min<int64_t>(mask, 24000) * 65536 / 24000), 16);
    // Not near the circuit
    const int64_t dx = x - kFlightCircuitCentre, dz = z - kFlightCircuitCentre;
    if (absI(dx) < kClearOuter && absI(dz) < kClearOuter) w = mulShift(w, clearance(centreDistance256(x, z)), 16);
    return w;
}

int32_t OverworldGenerator::biomeAt(const WorldPlan& plan, int64_t x, int64_t z)
{
    const int64_t veg   = biomeValue(plan, x, z);
    const int64_t range = rangeWeight(plan, x, z, landMask(plan, x, z));
    return int32_t(veg + mulShift(65536 - veg, range, 16));
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
    const int64_t m   = landMask(plan, x, z);
    int64_t h         = 0;
    if (m <= 0)
    {
        // 1. The sea floor: -4 m at the coast to -100 m at the mask -kSeaSlope
        h = lerpQ16(-4 * kMetre, -100 * kMetre, smoothQ16(std::min<int64_t>(-m, kSeaSlope) * 65536 / kSeaSlope));
    }
    else
    {
        // 2. The land: 12 m, the inland plateau (up to 43 m), hills of +-60 m, the mountain ranges (120 m + ridged
        // peaks of 0..900 m) weighted by the biome value
        int64_t land = 12 * kMetre + mulShift(std::min<int64_t>(m, 40000), 70 * kMetre, 16) +
                       mulShift(fbm(seedOf(plan.seed, TagHills), x, z, 10, 6), 60 * kMetre, 16);
        const int64_t w = rangeWeight(plan, x, z, m);
        if (w > 0)
        {
            const int64_t r = ridged(seedOf(plan.seed, TagMountains), x, z, 11, 5);
            land += mulShift(120 * kMetre + mulShift(floorShift(r * r, 16), 900 * kMetre, 16), w, 16);
        }
        // 3. Clamped to [sea level + 1 m, 1200 m]
        land = clampI<int64_t>(land, sea + kMetre, kFlightMountainHeight);
        // 4. The shore band: blended to the coast floor
        h = m < kShoreBand ? lerpQ16(-4 * kMetre, land, smoothQ16(m * 65536 / kShoreBand)) : land;
        // 5. The river valleys and lakes towards 3 m below the sea level, faded out in the mountains
        const int64_t river = mulShift(riverWeight(plan, x, z), 65536 - w, 16);
        if (river > 0) h = lerpQ16(h, std::min<int64_t>(h, sea - 3 * kMetre), river);
    }
    // 6. The generator's own range
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
    PlanGrid& g        = plan.rivers;
    const int32_t size = plan.space.sizeChunks * 256;
    const int32_t n    = size / kRiverCell;
    g.cellSize         = kRiverCell;
    g.originX          = 0;
    g.originZ          = 0;
    g.width            = n;
    g.height           = n;
    g.values.assign(uint32_t(n) * uint32_t(n), 0);
    // The water cells (rivers and lakes) and the river that marked a cell (0: none)
    std::vector<uint8_t> owner(g.values.size(), 0);
    auto mark = [&](int64_t mx, int64_t mz, uint8_t river)
    {
        const int64_t cx = floorDiv(mx, kRiverCell), cz = floorDiv(mz, kRiverCell);
        if (cx < 0 || cz < 0 || cx >= n || cz >= n) return;
        owner[gridIndex(g, cx, cz)] = river;
    };
    const uint64_t lakeSeed = seedOf(plan.seed, TagLakes);
    // A lake: the cells within a noisy radius (+-35 % by a 128 m value noise) of a centre (integer metres)
    auto lake = [&](int64_t x, int64_t z, int64_t radius, uint8_t river)
    {
        const int64_t reach = radius * 3 / 2;
        for (int64_t mz = floorDiv(z - reach, kRiverCell); mz <= floorDiv(z + reach, kRiverCell); ++mz)
            for (int64_t mx = floorDiv(x - reach, kRiverCell); mx <= floorDiv(x + reach, kRiverCell); ++mx)
            {
                const int64_t px = mx * kRiverCell + kRiverCell / 2, pz = mz * kRiverCell + kRiverCell / 2;
                const int64_t dx = px - x, dz = pz - z;
                const int64_t r = radius + mulShift(valueNoise(lakeSeed, px, pz, 7), radius * 35 / 100, 16);
                if (dx * dx + dz * dz <= r * r) mark(px, pz, river);
            }
    };

    // The sources: points on the 256 m grid in the foothills (range weight in [0.1, 0.27]), inland and away from the
    // circuit, in the order of a hash, at least kRiverSpacing apart
    const PlanGrid& b   = plan.biomes;
    const uint64_t seed = seedOf(plan.seed, TagRivers);
    std::vector<std::pair<uint64_t, int64_t>> order;
    for (int64_t cz = 0; cz < b.height; ++cz)
        for (int64_t cx = 0; cx < b.width; ++cx)
        {
            const int64_t x = cx * kBiomeCell + kBiomeCell / 2, z = cz * kBiomeCell + kBiomeCell / 2;
            const int64_t m = landMask(plan, x, z);
            const int64_t w = rangeWeight(plan, x, z, m);
            if (w < 6554 || w > 18000 || m < 20000 || centreDistance256(x, z) < kClearOuter * kPos) continue;
            order.push_back({contentHash64(seed, uint64_t(cx), uint64_t(cz), 1, 0), cz * b.width + cx});
        }
    std::sort(order.begin(), order.end());
    std::vector<std::pair<int64_t, int64_t>> sources;
    for (const auto& o : order)
    {
        if (int(sources.size()) >= kMaxRivers) break;
        const int64_t x = (o.second % b.width) * kBiomeCell + kBiomeCell / 2;
        const int64_t z = (o.second / b.width) * kBiomeCell + kBiomeCell / 2;
        bool far        = true;
        for (const auto& s : sources)
            if ((x - s.first) * (x - s.first) + (z - s.second) * (z - s.second) < kRiverSpacing * kRiverSpacing)
                far = false;
        if (far) sources.push_back({x, z});
    }

    // The walks: down the land mask in 32 m steps (its gradient over +-96 m), turned by up to +-35 degrees by a
    // noise, until the sea, the space edge or another river; a walk whose mask stops falling ends in a lake
    const int64_t size256 = int64_t(size) * kPos;
    // The rivers follow the continent's two coarsest octaves (8 and 4 km: few local minima)
    const uint64_t continent = seedOf(plan.seed, TagContinent);
    auto guide               = [&](int64_t x, int64_t z) { return int64_t(fbm(continent, x, z, 13, 2)); };
    for (uint32_t s = 0; s < uint32_t(sources.size()); ++s)
    {
        const uint8_t river = uint8_t(s + 1);
        int64_t px = sources[s].first * kPos, pz = sources[s].second * kPos;
        int64_t best = guide(sources[s].first, sources[s].second), stall = 0;
        for (int step = 0; step < kRiverSteps; ++step)
        {
            if (px < 0 || pz < 0 || px >= size256 || pz >= size256) break;
            const int64_t mx = floorDiv(px, kPos), mz = floorDiv(pz, kPos);
            const uint8_t o = owner[gridIndex(g, floorDiv(mx, kRiverCell), floorDiv(mz, kRiverCell))];
            if (o != 0 && o != river) break;
            if (centreDistance256(mx, mz) < kClearInner * kPos) break;
            // A river does not cross a mountain range (it starts on the other side)
            const int64_t mask = landMask(plan, mx, mz);
            if (rangeWeight(plan, mx, mz, mask) > 19661) break;
            mark(mx, mz, river);
            if (mask < 0) break;
            const int64_t m = guide(mx, mz);
            if (m < best)
            {
                best  = m;
                stall = 0;
            }
            else if (++stall > kRiverStall)
            {
                lake(mx, mz, 320, river);
                break;
            }
            const int64_t gx  = guide(mx + 128, mz) - guide(mx - 128, mz);
            const int64_t gz  = guide(mx, mz + 128) - guide(mx, mz - 128);
            const int64_t len = int64_t(isqrt64(uint64_t(gx * gx + gz * gz)));
            // Downhill (Q16)
            int64_t ox = 65536, oz = 0;
            if (len > 0)
            {
                ox = floorDiv(-gx * 65536, len);
                oz = floorDiv(-gz * 65536, len);
            }
            // +-35 degrees: 35 / 360 * 2^17 = 12743 trig steps at |noise| = 65536
            const int64_t nz     = gradientNoise(contentHash64(seed, uint64_t(s), 0, 0, 0), mx, mz, 9);
            const uint32_t angle = uint32_t(floorMod(floorDiv(nz * 12743, 65536), int64_t(kTrigSteps)));
            const int64_t c = cosQ30(angle), sn = sinQ30(angle);
            const int64_t dx = floorShift(ox * c - oz * sn, 30), dz = floorShift(ox * sn + oz * c, 30);
            px += mulShift(dx, 32 * kPos, 16);
            pz += mulShift(dz, 32 * kPos, 16);
        }
    }

    // The lowland lakes: one candidate per 4 km cell (45 %), jittered by +-1500 m, inland, below the mountains and
    // away from the circuit; radius 160-640 m
    for (int64_t lz = 0; lz < size / 4096; ++lz)
        for (int64_t lx = 0; lx < size / 4096; ++lx)
        {
            const uint64_t h = contentHash64(lakeSeed, uint64_t(lx), uint64_t(lz), 0, 0);
            if (h % 100 >= 45) continue;
            const int64_t x = lx * 4096 + 2048 - 1500 + int64_t((h >> 8) % 3001);
            const int64_t z = lz * 4096 + 2048 - 1500 + int64_t((h >> 24) % 3001);
            const int64_t m = landMask(plan, x, z);
            if (m < 16000 || rangeWeight(plan, x, z, m) > 16384 || centreDistance256(x, z) < (kClearOuter - 500) * kPos)
                continue;
            lake(x, z, 160 + int64_t((h >> 40) % 481), 255);
        }

    // The valley weights: a chamfer distance transform from the water cells (2 orthogonal, 3 diagonal), the weight
    // 65536 on a water cell falling linearly to 0 at kValleyDistance
    std::vector<int32_t> dist(g.values.size(), kValleyDistance);
    for (uint32_t i = 0; i < uint32_t(owner.size()); ++i)
        if (owner[i] != 0) dist[i] = 0;
    auto relax = [&](int64_t cx, int64_t cz, int64_t nx, int64_t nz, int32_t cost)
    {
        if (nx < 0 || nz < 0 || nx >= n || nz >= n) return;
        int32_t& d = dist[gridIndex(g, cx, cz)];
        d          = std::min(d, dist[gridIndex(g, nx, nz)] + cost);
    };
    for (int64_t cz = 0; cz < n; ++cz)
        for (int64_t cx = 0; cx < n; ++cx)
        {
            relax(cx, cz, cx - 1, cz, 2);
            relax(cx, cz, cx, cz - 1, 2);
            relax(cx, cz, cx - 1, cz - 1, 3);
            relax(cx, cz, cx + 1, cz - 1, 3);
        }
    for (int64_t cz = n - 1; cz >= 0; --cz)
        for (int64_t cx = n - 1; cx >= 0; --cx)
        {
            relax(cx, cz, cx + 1, cz, 2);
            relax(cx, cz, cx, cz + 1, 2);
            relax(cx, cz, cx + 1, cz + 1, 3);
            relax(cx, cz, cx - 1, cz + 1, 3);
        }
    for (uint32_t i = 0; i < uint32_t(dist.size()); ++i)
        g.values[i] = int32_t(int64_t(kValleyDistance - std::min(dist[i], kValleyDistance)) * 65536 / kValleyDistance);
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

    // The grid ruins: one jittered candidate per 2 km cell of the space
    const uint64_t seed = seedOf(plan.seed, TagSites);
    const int64_t cells = size / 2048;
    for (int64_t gz = 0; gz < cells; ++gz)
        for (int64_t gx = 0; gx < cells; ++gx)
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
            addRuin(kFlightGridRuinKey + uint32_t(gz * cells + gx), x, z, q, base);
        }
}

void OverworldGenerator::planGrids(WorldPlan& plan) const
{
    // The vegetation values at the 256 m cell centres, then the rivers (they start in the foothills)
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
        {
            // Meadow and forest from a 2 km fbm (at most kMountainLo: the ranges are added by biomeAt)
            const int64_t x = cx * kBiomeCell + kBiomeCell / 2, z = cz * kBiomeCell + kBiomeCell / 2;
            b.values[gridIndex(b, cx, cz)] = int32_t(std::min<int64_t>(fbm(biomeSeed, x, z, 11, 2), kMountainLo));
        }
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
        const Biome biome = biomeOf(biomeAt(plan, mx + jx, mz + jz));
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
        if (g < sea + kMetre || (biome == Mountain && g > 700 * kMetre)) return false;
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
        if (int64_t(h & 0xFFFF) >= pRock[biomeOf(biomeAt(plan, mx, mz))]) return false;
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
            const int64_t d = base[biomeOf(biomeAt(plan, bx, bz))];
            for (int32_t k = 0; k < perBlock; ++k)
                for (int32_t l = 0; l < perBlock; ++l)
                {
                    // The height at the cell centre, bilinear in the block (Q8 weights)
                    const int64_t fx  = int64_t(l) * spacing * 64 + spacing * 32;
                    const int64_t fz  = int64_t(k) * spacing * 64 + spacing * 32;
                    const int64_t top = h00 * (256 - fx) + h10 * fx, bot = h01 * (256 - fx) + h11 * fx;
                    const int64_t hc = floorShift(top * (256 - fz) + bot * fz, 16);
                    if (hc < sea + kMetre / 2 || hc > 700 * kMetre) continue;
                    const int32_t ci = bi * perBlock + l, cj = bj * perBlock + k;
                    const int64_t v = valueNoise(seed, x0 + int64_t(ci) * spacing, z0 + int64_t(cj) * spacing, 4);
                    out.density[uint32_t(cj * n + ci)] = uint8_t(clampI<int64_t>(d + floorShift(d * v, 17), 1, 255));
                }
        }
}

}  // namespace Game
}  // namespace Saiga
