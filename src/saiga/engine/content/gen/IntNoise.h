/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// IntNoise (docs/engine_design.md §12.1, R-CONTENT-1; S23a): the counter-based randomness and the integer noise
// functions of the content generators. Unsigned or int64 arithmetic only; the same values on every compiler.
// contentHash64 is the only randomness in content code (a candidate is a pure function of its inputs, §12.1 (b)); the
// sequential Rng (PCG32) is not used in generators.

#pragma once

#include "saiga/export.h"

#include <cstdint>

namespace Saiga
{
namespace Engine
{
// splitmix64's finalizer
constexpr uint64_t splitMix64(uint64_t z)
{
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// splitmix64 finalizers chained over the seed and the four inputs
constexpr uint64_t contentHash64(uint64_t seed, uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
    uint64_t h = splitMix64(seed);
    h          = splitMix64(h ^ a);
    h          = splitMix64(h ^ b);
    h          = splitMix64(h ^ c);
    return splitMix64(h ^ d);
}

// 2D gradient noise in [-65536, 65536] (Q16) over lattice cells of 2^cellShift input units, 1 <= cellShift <= 40: 16
// gradient directions at multiples of 22.5 degrees (Q16 components from the trig table), chosen by
// contentHash64(seed, cellX, cellZ, 0, 0) & 15; the four corner dot products blended with smoothQ16 and lerpQ16; scaled
// by sqrt(2) (the maximum magnitude of the raw value is sqrt(2) / 2, at a cell centre) and clamped.
SAIGA_ENGINE_API int32_t gradientNoise(uint64_t seed, int64_t x, int64_t z, int cellShift);

// Value noise in [-65536, 65536]: a hashed value per lattice corner, blended with smoothQ16
SAIGA_ENGINE_API int32_t valueNoise(uint64_t seed, int64_t x, int64_t z, int cellShift);

// Fractal sums of gradientNoise: octave i uses cellShift - i, its amplitude shifted right by i * gainShift and the seed
// contentHash64(seed, i, 0, 0, 0); normalized by the integer amplitude sum to [-65536, 65536]. Checked in every build:
// 1 <= octaves <= cellShift, octaves <= 16, (octaves - 1) * gainShift <= 16
SAIGA_ENGINE_API int32_t fbm(uint64_t seed, int64_t x, int64_t z, int cellShift, int octaves, int gainShift = 1);
// 65536 - |n| per octave with fbm's octave rules: in [0, 65536]
SAIGA_ENGINE_API int32_t ridged(uint64_t seed, int64_t x, int64_t z, int cellShift, int octaves, int gainShift = 1);

}  // namespace Engine
}  // namespace Saiga
