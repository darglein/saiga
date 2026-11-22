/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/export.h"

#include <cstdint>

namespace Saiga
{
namespace Vulkan
{
/**
 * Fast hash based 2D noise for procedural content (deterministic for a seed).
 */
struct SAIGA_VULKAN_API WorldNoise
{
    // Gradient noise in about [-1, 1] with a period of 1 unit.
    static float gradient(float x, float y, uint32_t seed);
    // Fractal sum of gradient noise, about [-1, 1].
    static float fbm(float x, float y, int octaves, uint32_t seed, float lacunarity = 2.0f, float gain = 0.5f);
    // Ridged multifractal noise in [0, 1] (sharp ridges at the zero crossings of the gradient noise).
    static float ridged(float x, float y, int octaves, uint32_t seed);
    // Uniform random number in [0, 1) for the integer coordinates.
    static float hash01(int x, int y, uint32_t seed);
};

}  // namespace Vulkan
}  // namespace Saiga
