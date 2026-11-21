/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/ForestRenderer.h"

#include "saiga/vulkan/world/WorldNoise.h"

#include <algorithm>
#include <cmath>
#include <random>

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

std::vector<ForestRenderer::TreeInstance> ForestRenderer::placeTrees(const PrototypeArea& area, int count,
                                                                     uint32_t seed,
                                                                     const std::vector<TreeSpecies>& species)
{
    const float cell     = 4.0f;
    const int cells      = int(area.size / double(cell));
    const int n          = cells + 1;
    const float water    = float(double(kPrototypeSeaLevel) * kHeightUnit);
    const float treeline = float(kPrototypeMountainHeight) * 0.72f;
    const float x0 = float(area.x0), z0 = float(area.z0);

    // The hook's samples on the 4 m grid (heights in metres)
    std::vector<float> grid(size_t(n) * n);
#pragma omp parallel for schedule(dynamic, 16)
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            grid[size_t(j) * n + i] = float(double(prototypeHeight(area.seed, int64_t(std::llround(area.x0 + 4.0 * i)),
                                                                   int64_t(std::llround(area.z0 + 4.0 * j)))) *
                                            kHeightUnit);

    // Probability of a tree per cell: forest patches, flat enough (the cell's gradient), above the water and below
    // the tree line
    std::vector<float> p(size_t(cells) * cells);
#pragma omp parallel for schedule(dynamic, 16)
    for (int j = 0; j < cells; ++j)
    {
        for (int i = 0; i < cells; ++i)
        {
            const float h00 = grid[size_t(j) * n + i], h10 = grid[size_t(j) * n + i + 1];
            const float h01 = grid[size_t(j + 1) * n + i], h11 = grid[size_t(j + 1) * n + i + 1];
            const float h  = 0.25f * (h00 + h10 + h01 + h11);
            const float dx = ((h10 + h11) - (h00 + h01)) / (2 * cell), dz = ((h01 + h11) - (h00 + h10)) / (2 * cell);
            const float ny = 1.0f / std::sqrt(1.0f + dx * dx + dz * dz);
            const float x = x0 + (i + 0.5f) * cell, z = z0 + (j + 0.5f) * cell;
            const float forest =
                smoothstepf(-0.05f, 0.3f, WorldNoise::fbm(x / 700.0f, z / 700.0f, 4, seed * 31 + 1)) * 0.95f + 0.02f;
            const float slope = smoothstepf(0.8f, 0.9f, ny);
            const float height =
                smoothstepf(water + 1.0f, water + 4.0f, h) * smoothstepf(treeline, treeline - 80.0f, h);
            p[size_t(j) * cells + i] = forest * slope * height;
        }
    }
    // Serial sum: a parallel reduction would make the result (and the placement) depend on the scheduling
    double sum = 0;
    for (float v : p) sum += v;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0, 1);
    const float scale = sum > 0 ? float(count / sum) : 0;

    std::vector<TreeInstance> trees;
    trees.reserve(size_t(count * 1.1f));
    for (int j = 0; j < cells; ++j)
    {
        for (int i = 0; i < cells; ++i)
        {
            const float prob = p[size_t(j) * cells + i] * scale;
            if (u01(rng) >= prob) continue;
            const float x = x0 + (i + u01(rng)) * cell, z = z0 + (j + u01(rng)) * cell;
            const float h = float(prototypeSurfaceHeight(area.seed, area.baseSpacing, double(x), double(z)));

            // Spruce in the mountains, broadleaf and birch in the valleys (in patches)
            const float spruce = smoothstepf(70, 200, h) * 0.85f +
                                 0.15f * smoothstepf(-0.2f, 0.4f, WorldNoise::fbm(x / 400.0f, z / 400.0f, 2, seed + 7));
            int s;
            if (u01(rng) < spruce)
                s = 0;
            else
                s = WorldNoise::fbm(x / 300.0f, z / 300.0f, 2, seed + 9) + u01(rng) * 0.6f - 0.3f > 0 ? 1 : 2;
            s = std::min(s, int(species.size()) - 1);

            const vec2 hr = species[size_t(s)].heightRange;
            TreeInstance t;
            // Sink the stem base a little into slopes
            t.posYaw       = vec4(x, h - 0.25f, z, u01(rng) * 6.2831853f);
            t.scaleSpecies = vec4(hr(0) + (hr(1) - hr(0)) * u01(rng), float(s), 0, 0);
            trees.push_back(t);
        }
    }
    if ((int)trees.size() > count)
    {
        std::shuffle(trees.begin(), trees.end(), rng);
        trees.resize(size_t(count));
    }
    return trees;
}

}  // namespace Vulkan
}  // namespace Saiga
