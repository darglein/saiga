/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/WorldNoise.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Vulkan
{
static inline uint32_t hashU32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static inline uint32_t hash2(int x, int y, uint32_t seed)
{
    return hashU32(uint32_t(x) * 0x8da6b343U ^ hashU32(uint32_t(y) * 0xd8163841U ^ seed));
}

static inline float smootherstep(float t)
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float WorldNoise::hash01(int x, int y, uint32_t seed)
{
    return float(hash2(x, y, seed) >> 8) * (1.0f / 16777216.0f);
}

float WorldNoise::gradient(float x, float y, uint32_t seed)
{
    // 16 gradient directions
    static const float gx[16] = {1.0f,  0.924f,  0.707f,  0.383f,  0.0f, -0.383f, -0.707f, -0.924f,
                                 -1.0f, -0.924f, -0.707f, -0.383f, 0.0f, 0.383f,  0.707f,  0.924f};
    static const float gy[16] = {0.0f, 0.383f,  0.707f,  0.924f,  1.0f,  0.924f,  0.707f,  0.383f,
                                 0.0f, -0.383f, -0.707f, -0.924f, -1.0f, -0.924f, -0.707f, -0.383f};

    float fx0 = std::floor(x), fy0 = std::floor(y);
    int ix = int(fx0), iy = int(fy0);
    float fx = x - fx0, fy = y - fy0;

    auto g = [&](int cx, int cy, float dx, float dy)
    {
        uint32_t h = hash2(cx, cy, seed) & 15;
        return gx[h] * dx + gy[h] * dy;
    };

    float n00 = g(ix, iy, fx, fy);
    float n10 = g(ix + 1, iy, fx - 1, fy);
    float n01 = g(ix, iy + 1, fx, fy - 1);
    float n11 = g(ix + 1, iy + 1, fx - 1, fy - 1);

    float u = smootherstep(fx), v = smootherstep(fy);
    float nx0 = n00 + u * (n10 - n00);
    float nx1 = n01 + u * (n11 - n01);
    return (nx0 + v * (nx1 - nx0)) * 1.4f;
}

float WorldNoise::fbm(float x, float y, int octaves, uint32_t seed, float lacunarity, float gain)
{
    float sum = 0, amp = 1, norm = 0;
    for (int o = 0; o < octaves; ++o)
    {
        sum += amp * gradient(x, y, seed + uint32_t(o) * 1013U);
        norm += amp;
        amp *= gain;
        // Rotate a little per octave to hide the grid
        float nx = (x * 0.8f - y * 0.6f) * lacunarity;
        float ny = (x * 0.6f + y * 0.8f) * lacunarity;
        x        = nx + 17.3f;
        y        = ny - 9.1f;
    }
    return sum / norm;
}

float WorldNoise::ridged(float x, float y, int octaves, uint32_t seed)
{
    float sum = 0, amp = 1, norm = 0, weight = 1;
    for (int o = 0; o < octaves; ++o)
    {
        float n = 1.0f - std::abs(gradient(x, y, seed + uint32_t(o) * 7919U));
        n       = n * n * weight;
        weight  = std::min(std::max(n * 2.0f, 0.0f), 1.0f);
        sum += n * amp;
        norm += amp;
        amp *= 0.5f;
        float nx = (x * 0.8f - y * 0.6f) * 2.0f;
        float ny = (x * 0.6f + y * 0.8f) * 2.0f;
        x        = nx + 3.7f;
        y        = ny + 11.9f;
    }
    return sum / norm;
}

}  // namespace Vulkan
}  // namespace Saiga
