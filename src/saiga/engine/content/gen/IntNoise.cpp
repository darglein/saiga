/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/gen/IntNoise.h"

#include "saiga/core/util/assert.h"
#include "saiga/engine/content/gen/FixedMath.h"
#include "saiga/engine/content/gen/TrigTable.h"

#include <array>
#include <string>

namespace Saiga
{
namespace Engine
{
namespace
{
// The 16 gradient directions (Q16 components): direction k at k * 22.5 degrees, k * 8192 steps of the trig table
struct Gradients
{
    std::array<int64_t, 16> gx{}, gz{};
};

const Gradients& gradients()
{
    static const Gradients g = []
    {
        Gradients r;
        for (uint32_t k = 0; k < 16; ++k)
        {
            r.gx[k] = floorShift(cosQ30(k * 8192), 14);
            r.gz[k] = floorShift(sinQ30(k * 8192), 14);
        }
        return r;
    }();
    return g;
}

struct Cell
{
    int64_t cx = 0, cz = 0;  // the lattice cell
    int64_t fx = 0, fz = 0;  // the fraction in Q16, [0, 65536)
};

Cell cellOf(int64_t x, int64_t z, int cellShift)
{
    SAIGA_ASSERT(cellShift >= 1 && cellShift <= 40, "noise: 1 <= cellShift <= 40");
    Cell c;
    c.cx                = floorShift(x, cellShift);
    c.cz                = floorShift(z, cellShift);
    const uint64_t mask = (uint64_t(1) << cellShift) - 1;
    // The two's complement bits: x - cx * 2^cellShift in [0, 2^cellShift)
    const int64_t fx = int64_t(uint64_t(x) & mask);
    const int64_t fz = int64_t(uint64_t(z) & mask);
    if (cellShift <= 16)
    {
        c.fx = fx << (16 - cellShift);
        c.fz = fz << (16 - cellShift);
    }
    else
    {
        c.fx = fx >> (cellShift - 16);
        c.fz = fz >> (cellShift - 16);
    }
    return c;
}

// The dot product of the corner's gradient with the offset (dx, dz) (Q16), in Q16
int64_t cornerDot(uint64_t seed, int64_t cx, int64_t cz, int64_t dx, int64_t dz)
{
    const uint32_t k   = uint32_t(contentHash64(seed, uint64_t(cx), uint64_t(cz), 0, 0) & 15);
    const Gradients& g = gradients();
    return floorShift(g.gx[k] * dx + g.gz[k] * dz, 16);
}

int64_t cornerValue(uint64_t seed, int64_t cx, int64_t cz)
{
    return int64_t(contentHash64(seed, uint64_t(cx), uint64_t(cz), 1, 0) % 131073) - 65536;
}

void checkOctaves(int cellShift, int octaves, int gainShift)
{
    if (octaves < 1 || octaves > cellShift || octaves > 16 || gainShift < 0 || (octaves - 1) * gainShift > 16)
        SAIGA_EXIT_ERROR(
            "fbm / ridged: need 1 <= octaves <= cellShift, octaves <= 16, (octaves - 1) * gainShift <= 16 "
            "(cellShift " +
            std::to_string(cellShift) + ", octaves " + std::to_string(octaves) + ", gainShift " +
            std::to_string(gainShift) + ")");
}
}  // namespace

int32_t gradientNoise(uint64_t seed, int64_t x, int64_t z, int cellShift)
{
    const Cell c      = cellOf(x, z, cellShift);
    const int64_t one = 65536;
    const int64_t n00 = cornerDot(seed, c.cx, c.cz, c.fx, c.fz);
    const int64_t n10 = cornerDot(seed, c.cx + 1, c.cz, c.fx - one, c.fz);
    const int64_t n01 = cornerDot(seed, c.cx, c.cz + 1, c.fx, c.fz - one);
    const int64_t n11 = cornerDot(seed, c.cx + 1, c.cz + 1, c.fx - one, c.fz - one);
    const int64_t u   = smoothQ16(c.fx);
    const int64_t v   = smoothQ16(c.fz);
    const int64_t a   = lerpQ16(n00, n10, u);
    const int64_t b   = lerpQ16(n01, n11, u);
    const int64_t n   = lerpQ16(a, b, v);
    // sqrt(2) in Q16; the clamp only absorbs rounding
    return int32_t(clampI<int64_t>(floorDiv(n * 92682, 65536), -65536, 65536));
}

int32_t valueNoise(uint64_t seed, int64_t x, int64_t z, int cellShift)
{
    const Cell c      = cellOf(x, z, cellShift);
    const int64_t v00 = cornerValue(seed, c.cx, c.cz);
    const int64_t v10 = cornerValue(seed, c.cx + 1, c.cz);
    const int64_t v01 = cornerValue(seed, c.cx, c.cz + 1);
    const int64_t v11 = cornerValue(seed, c.cx + 1, c.cz + 1);
    const int64_t u   = smoothQ16(c.fx);
    const int64_t v   = smoothQ16(c.fz);
    return int32_t(lerpQ16(lerpQ16(v00, v10, u), lerpQ16(v01, v11, u), v));
}

int32_t fbm(uint64_t seed, int64_t x, int64_t z, int cellShift, int octaves, int gainShift)
{
    checkOctaves(cellShift, octaves, gainShift);
    int64_t sum = 0, total = 0;
    for (int i = 0; i < octaves; ++i)
    {
        const int64_t amplitude = int64_t(65536) >> (i * gainShift);
        sum += int64_t(gradientNoise(contentHash64(seed, uint64_t(i), 0, 0, 0), x, z, cellShift - i)) * amplitude;
        total += amplitude;
    }
    return int32_t(floorDiv(sum, total));
}

int32_t ridged(uint64_t seed, int64_t x, int64_t z, int cellShift, int octaves, int gainShift)
{
    checkOctaves(cellShift, octaves, gainShift);
    int64_t sum = 0, total = 0;
    for (int i = 0; i < octaves; ++i)
    {
        const int64_t amplitude = int64_t(65536) >> (i * gainShift);
        const int64_t n         = gradientNoise(contentHash64(seed, uint64_t(i), 0, 0, 0), x, z, cellShift - i);
        sum += (65536 - (n < 0 ? -n : n)) * amplitude;
        total += amplitude;
    }
    return int32_t(floorDiv(sum, total));
}

}  // namespace Engine
}  // namespace Saiga
