/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/ForestRenderer.h"

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

std::vector<ForestRenderer::TreeInstance> ForestRenderer::placeTrees(const HeightField& hf, int count, uint32_t seed,
                                                                     const std::vector<TreeSpecies>& species)
{
    const float cell     = 4.0f;
    const float W        = hf.worldSize();
    const int cells      = int(W / cell);
    const float water    = hf.waterLevel();
    const float treeline = hf.parameters().mountainHeight * 0.72f;

    // Probability of a tree per cell
    std::vector<float> p(size_t(cells) * cells);
#pragma omp parallel for schedule(dynamic, 16)
    for (int j = 0; j < cells; ++j)
    {
        for (int i = 0; i < cells; ++i)
        {
            float x = (i + 0.5f) * cell, z = (j + 0.5f) * cell;
            float h = hf.height(x, z);
            vec3 n  = hf.normal(x, z, 3.0f);
            float forest =
                smoothstepf(-0.05f, 0.3f, WorldNoise::fbm(x / 700.0f, z / 700.0f, 4, seed * 31 + 1)) * 0.95f + 0.02f;
            float slope  = smoothstepf(0.8f, 0.9f, n(1));
            float height = smoothstepf(water + 1.0f, water + 4.0f, h) * smoothstepf(treeline, treeline - 80.0f, h);
            float v      = forest * slope * height;
            p[size_t(j) * cells + i] = v;
        }
    }
    // Serial sum: a parallel reduction would make the result (and the placement) depend on the scheduling
    double sum = 0;
    for (float v : p) sum += v;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0, 1);
    float scale = sum > 0 ? float(count / sum) : 0;

    std::vector<TreeInstance> trees;
    trees.reserve(size_t(count * 1.1f));
    for (int j = 0; j < cells; ++j)
    {
        for (int i = 0; i < cells; ++i)
        {
            float prob = p[size_t(j) * cells + i] * scale;
            if (u01(rng) >= prob) continue;
            float x = (i + u01(rng)) * cell, z = (j + u01(rng)) * cell;
            float h = hf.height(x, z);

            // Spruce in the mountains, broadleaf and birch in the valleys (in patches)
            float spruce = smoothstepf(70, 200, h) * 0.85f +
                           0.15f * smoothstepf(-0.2f, 0.4f, WorldNoise::fbm(x / 400.0f, z / 400.0f, 2, seed + 7));
            int s;
            if (u01(rng) < spruce)
                s = 0;
            else
                s = WorldNoise::fbm(x / 300.0f, z / 300.0f, 2, seed + 9) + u01(rng) * 0.6f - 0.3f > 0 ? 1 : 2;
            s = std::min(s, int(species.size()) - 1);

            vec2 hr = species[s].heightRange;
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
        trees.resize(count);
    }
    return trees;
}

}  // namespace Vulkan
}  // namespace Saiga
