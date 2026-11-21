/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/HeightField.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Vulkan
{
static inline float smoothstepf(float e0, float e1, float x)
{
    float t = std::min(std::max((x - e0) / (e1 - e0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

void HeightField::generate(const Parameters& _params)
{
    params = _params;
    n      = params.size;
    data.resize(size_t(n) * n);

    const float W    = worldSize();
    const uint32_t s = params.seed * 7717U;

#pragma omp parallel for schedule(dynamic, 16)
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            float x = i * params.spacing;
            float z = j * params.spacing;
            // Noise coordinates in km
            float px = x * 0.001f, pz = z * 0.001f;

            // Domain warp for less regular shapes
            float wx = WorldNoise::fbm(px * 0.6f, pz * 0.6f, 4, s + 1) * 0.35f;
            float wz = WorldNoise::fbm(px * 0.6f + 5.2f, pz * 0.6f + 1.3f, 4, s + 2) * 0.35f;
            float qx = px + wx, qz = pz + wz;

            // Mountain ranges in some areas, rolling hills elsewhere
            float mask      = smoothstepf(-0.15f, 0.45f, WorldNoise::fbm(qx * 0.35f, qz * 0.35f, 3, s + 3));
            float hills     = WorldNoise::fbm(qx * 1.4f, qz * 1.4f, 6, s + 4) * 38.0f + 34.0f;
            float r         = WorldNoise::ridged(qx * 0.55f, qz * 0.55f, 7, s + 5);
            float mountains = r * r * params.mountainHeight * mask;
            float h         = hills + mountains + mask * 30.0f;

            // Small detail
            h += WorldNoise::fbm(px * 12.0f, pz * 12.0f, 3, s + 6) * 1.5f;

            // The world ends in the water
            float border = std::min(std::min(x, W - x), std::min(z, W - z)) / W;
            float e      = smoothstepf(0.0f, 0.1f, border);
            h            = (params.waterLevel - 25.0f) * (1.0f - e) + h * e;

            data[size_t(j) * n + i] = h;
        }
    }

    minH                = data[0];
    maxH                = data[0];
    size_t highestIndex = 0;
    for (size_t k = 0; k < data.size(); ++k)
    {
        minH = std::min(minH, data[k]);
        if (data[k] > maxH)
        {
            maxH         = data[k];
            highestIndex = k;
        }
    }
    highest = vec3((highestIndex % n) * params.spacing, maxH, (highestIndex / n) * params.spacing);
}

float HeightField::height(float x, float z) const
{
    float gx = std::min(std::max(x / params.spacing, 0.0f), float(n - 1));
    float gz = std::min(std::max(z / params.spacing, 0.0f), float(n - 1));
    int ix   = std::min(int(gx), n - 2);
    int iz   = std::min(int(gz), n - 2);
    float fx = gx - ix, fz = gz - iz;

    float ha = sample(ix, iz), hb = sample(ix + 1, iz), hc = sample(ix, iz + 1), hd = sample(ix + 1, iz + 1);
    if (fx + fz <= 1.0f) return ha + fx * (hb - ha) + fz * (hc - ha);
    return hd + (1.0f - fx) * (hc - hd) + (1.0f - fz) * (hb - hd);
}

vec3 HeightField::normal(float x, float z, float step) const
{
    float hl = height(x - step, z), hr = height(x + step, z);
    float hd = height(x, z - step), hu = height(x, z + step);
    vec3 nrm(hl - hr, 2.0f * step, hd - hu);
    return nrm.normalized();
}

}  // namespace Vulkan
}  // namespace Saiga
