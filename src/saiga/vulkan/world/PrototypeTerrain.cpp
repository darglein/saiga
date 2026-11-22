/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/PrototypeTerrain.h"

#include "saiga/vulkan/world/WorldNoise.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Saiga
{
namespace Vulkan
{
namespace
{
float smoothstepf(float e0, float e1, float x)
{
    float t = std::min(std::max((x - e0) / (e1 - e0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
}  // namespace

int32_t prototypeHeight(uint64_t seed, int64_t ix, int64_t iz)
{
    const uint32_t s     = uint32_t(seed) * 7717U;
    const float x        = float(ix);
    const float z        = float(iz);
    const float W        = float(kPrototypeSpaceChunks) * 256.0f;
    const float water    = float(double(kPrototypeSeaLevel) * kHeightUnit);
    const float mountain = float(kPrototypeMountainHeight);
    // Noise coordinates in km
    const float px = x * 0.001f, pz = z * 0.001f;

    // Domain warp for less regular shapes
    const float wx = WorldNoise::fbm(px * 0.6f, pz * 0.6f, 4, s + 1) * 0.35f;
    const float wz = WorldNoise::fbm(px * 0.6f + 5.2f, pz * 0.6f + 1.3f, 4, s + 2) * 0.35f;
    const float qx = px + wx, qz = pz + wz;

    // Mountain ranges in some areas, rolling hills elsewhere
    const float mask      = smoothstepf(-0.15f, 0.45f, WorldNoise::fbm(qx * 0.35f, qz * 0.35f, 3, s + 3));
    const float hills     = WorldNoise::fbm(qx * 1.4f, qz * 1.4f, 6, s + 4) * 38.0f + 34.0f;
    const float r         = WorldNoise::ridged(qx * 0.55f, qz * 0.55f, 7, s + 5);
    const float mountains = r * r * mountain * mask;
    float h               = hills + mountains + mask * 30.0f;

    // Small detail
    h += WorldNoise::fbm(px * 12.0f, pz * 12.0f, 3, s + 6) * 1.5f;

    // The space ends in the water: 819.2 m of falloff at its edges
    const float border = std::min(std::min(x, W - x), std::min(z, W - z));
    const float e      = smoothstepf(0.0f, 819.2f, border);
    h                  = (water - 25.0f) * (1.0f - e) + h * e;
    const double units = std::round(double(h) * 1024.0);
    return clampTerrainHeight(int32_t(std::max(std::min(units, 2.0e9), -2.0e9)));
}

double prototypeSurfaceHeight(uint64_t seed, int baseSpacing, double x, double z)
{
    // As TerrainSystem::height for a space of kPrototypeSpaceChunks with the prototype hook (NaN: 0, as there; the
    // clamps below would keep it for the integer conversion)
    if (std::isnan(x) || std::isnan(z)) return 0;
    const double size    = double(kPrototypeSpaceChunks) * 256.0;
    const double spacing = baseSpacing;
    const double gx      = std::min(std::max(x, 0.0), size) / spacing;
    const double gz      = std::min(std::max(z, 0.0), size) / spacing;
    const int64_t n      = int64_t(kPrototypeSpaceChunks) * 256 / baseSpacing;
    const int64_t ci     = std::min(int64_t(std::floor(gx)), n - 1);
    const int64_t cj     = std::min(int64_t(std::floor(gz)), n - 1);
    const double fx      = gx - double(ci);
    const double fz      = gz - double(cj);
    auto sample       = [&](int64_t i, int64_t j) { return prototypeHeight(seed, i * baseSpacing, j * baseSpacing); };
    const int32_t h00 = sample(ci, cj), h11 = sample(ci + 1, cj + 1);
    int32_t h10 = 0, h01 = 0;
    if (fx >= fz)
        h10 = sample(ci + 1, cj);
    else
        h01 = sample(ci, cj + 1);
    return terrainSurfaceDouble(h00, h10, h01, h11, fx, fz) * kHeightUnit;
}

PrototypeAreaStats prototypeAreaStats(uint64_t seed, double x0, double z0, double size, double step)
{
    PrototypeAreaStats s;
    s.minHeight      = std::numeric_limits<double>::max();
    s.maxHeight      = -std::numeric_limits<double>::max();
    const int64_t nx = int64_t(std::floor(size / step));
    for (int64_t j = 0; j <= nx; ++j)
        for (int64_t i = 0; i <= nx; ++i)
        {
            const int64_t x = int64_t(std::llround(x0 + double(i) * step)),
                          z = int64_t(std::llround(z0 + double(j) * step));
            const double h  = double(prototypeHeight(seed, x, z)) * kHeightUnit;
            s.minHeight     = std::min(s.minHeight, h);
            if (h > s.maxHeight)
            {
                s.maxHeight = h;
                s.highest   = Vec3(double(x), h, double(z));
            }
        }
    return s;
}

}  // namespace Vulkan
}  // namespace Saiga
