/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The terrain surface rule of docs/engine_design.md §12.2 (S20): the integer and double forms, the clamp and the grid
// triangulation along the (0,0)-(1,1) diagonal.

#include "saiga/core/math/TerrainSurface.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

using namespace Saiga;

namespace
{
// The surface by the triangle's plane equation in exact rational arithmetic (numerator in units of 2^-f), independent
// of the implementation's slope form
int64_t referenceNumerator(int32_t h00, int32_t h10, int32_t h01, int32_t h11, int64_t fx, int64_t fz, int f)
{
    const int64_t one = int64_t(1) << f;
    if (fx >= fz)
    {
        // A = (00, 10, 11): barycentric weights (1 - fx), (fx - fz), fz
        return h00 * (one - fx) + h10 * (fx - fz) + h11 * fz;
    }
    // B = (00, 11, 01): (1 - fz), fx, (fz - fx)
    return h00 * (one - fz) + h11 * fx + h01 * (fz - fx);
}

int64_t floorDiv(int64_t a, int64_t b)
{
    int64_t q = a / b;
    if (a % b != 0 && ((a < 0) != (b < 0))) q--;
    return q;
}
}  // namespace

TEST(TerrainSurface, IntegerRule)
{
    // Vertices
    EXPECT_EQ(terrainSurfaceUnits(10, 20, 30, 40, 0, 0, 0), 10);
    EXPECT_EQ(terrainSurfaceUnits(10, 20, 30, 40, 1, 0, 0), 20);
    EXPECT_EQ(terrainSurfaceUnits(10, 20, 30, 40, 0, 1, 0), 30);
    EXPECT_EQ(terrainSurfaceUnits(10, 20, 30, 40, 1, 1, 0), 40);
    // The diagonal (0,0)-(1,1): the mean of h00 and h11, independent of h10 and h01
    EXPECT_EQ(terrainSurfaceUnits(0, 1000, -1000, 100, 4, 4, 3), 50);
    EXPECT_EQ(terrainSurfaceUnits(0, -7000, 9000, 100, 4, 4, 3), 50);
    // A saddle: the other diagonal would give the mean of h10 and h01 at the centre
    EXPECT_EQ(terrainSurfaceUnits(0, 100, 100, 0, 1, 1, 1), 0);
    // Triangle A (fx >= fz) uses h10, triangle B (fz > fx) uses h01
    EXPECT_EQ(terrainSurfaceUnits(0, 64, 0, 0, 2, 1, 2), 32 - 16);
    EXPECT_EQ(terrainSurfaceUnits(0, 0, 64, 0, 1, 2, 2), 32 - 16);
    // Edges: fz = 0 interpolates h00-h10, fx = 0 h00-h01, fx = 1 h10-h11, fz = 1 h01-h11
    EXPECT_EQ(terrainSurfaceUnits(0, 64, 128, 256, 1, 0, 2), 16);
    EXPECT_EQ(terrainSurfaceUnits(0, 64, 128, 256, 0, 1, 2), 32);
    EXPECT_EQ(terrainSurfaceUnits(0, 64, 128, 256, 4, 2, 2), 160);
    EXPECT_EQ(terrainSurfaceUnits(0, 64, 128, 256, 2, 4, 2), 192);
    // Negative heights round toward -infinity
    EXPECT_EQ(terrainSurfaceUnits(-1, 0, 0, 0, 1, 0, 1), -1);
    EXPECT_EQ(terrainSurfaceUnits(0, -1, 0, 0, 1, 0, 1), -1);
    EXPECT_EQ(terrainSurfaceUnits(-3, -3, -3, -4, 1, 1, 1), -4);
    EXPECT_EQ(terrainSurfaceUnits(0, 1, 0, 0, 1, 0, 1), 0);
    // The limits with fracBits 20: no overflow
    const int32_t L = kTerrainHeightLimit;
    const int64_t o = int64_t(1) << 20;
    EXPECT_EQ(terrainSurfaceUnits(L, -L, L, -L, o, 0, 20), -L);
    EXPECT_EQ(terrainSurfaceUnits(-L, L, -L, L, o / 2, o / 2 - 1, 20),
              int32_t(floorDiv(referenceNumerator(-L, L, -L, L, o / 2, o / 2 - 1, 20), o)));
    EXPECT_EQ(terrainSurfaceUnits(-L, L, L, -L, o - 1, 1, 20),
              int32_t(floorDiv(referenceNumerator(-L, L, L, -L, o - 1, 1, 20), o)));
    EXPECT_EQ(terrainSurfaceNumerator(-L, L, L, -L, o - 1, 1, 20), referenceNumerator(-L, L, L, -L, o - 1, 1, 20));
}

TEST(TerrainSurface, DoubleEqualsIntegerForDyadic)
{
    std::mt19937_64 rng(20);
    std::uniform_int_distribution<int32_t> height(-kTerrainHeightLimit, kTerrainHeightLimit);
    std::uniform_int_distribution<int32_t> small(-5000, 5000);
    for (int k = 0; k < 100000; ++k)
    {
        const int f       = int(rng() % 21);
        const int64_t one = int64_t(1) << f;
        // Half of the points with heights at the full range, half with small heights (the common case)
        auto h          = [&]() { return k % 2 ? height(rng) : small(rng); };
        const int32_t a = h(), b = h(), c = h(), d = h();
        int64_t fx = int64_t(rng() % uint64_t(one + 1)), fz = int64_t(rng() % uint64_t(one + 1));
        // Every 8th point on the diagonal
        if (k % 8 == 0) fz = fx;
        const int64_t num = terrainSurfaceNumerator(a, b, c, d, fx, fz, f);
        ASSERT_EQ(num, referenceNumerator(a, b, c, d, fx, fz, f));
        const double x = double(fx) / double(one), z = double(fz) / double(one);
        const double s = terrainSurfaceDouble(a, b, c, d, x, z);
        ASSERT_EQ(s * double(one), double(num)) << "f " << f << " fx " << fx << " fz " << fz;
        ASSERT_EQ(int64_t(std::floor(s)), int64_t(terrainSurfaceUnits(a, b, c, d, fx, fz, f)));
    }
}

TEST(TerrainSurface, ClampToLimit)
{
    EXPECT_EQ(clampTerrainHeight(0), 0);
    EXPECT_EQ(clampTerrainHeight(-17), -17);
    EXPECT_EQ(clampTerrainHeight(kTerrainHeightLimit), kTerrainHeightLimit);
    EXPECT_EQ(clampTerrainHeight(kTerrainHeightLimit + 1), kTerrainHeightLimit);
    EXPECT_EQ(clampTerrainHeight(-kTerrainHeightLimit - 1), -kTerrainHeightLimit);
    EXPECT_EQ(clampTerrainHeight(std::numeric_limits<int32_t>::max()), kTerrainHeightLimit);
    EXPECT_EQ(clampTerrainHeight(std::numeric_limits<int32_t>::min()), -kTerrainHeightLimit);

    // fromHook clamps every result; a null hook is no terrain
    auto source =
        TerrainHeightSource::fromHook([](uint64_t seed, int64_t x, int64_t)
                                      { return x > 0 ? std::numeric_limits<int32_t>::max() : int32_t(seed); }, 7);
    ASSERT_TRUE(bool(source));
    EXPECT_EQ(source(1, 0), kTerrainHeightLimit);
    EXPECT_EQ(source(-1, 0), 7);
    EXPECT_FALSE(bool(TerrainHeightSource::fromHook(nullptr, 1)));
}

TEST(TerrainSurface, GridTrianglesUseTheDiagonal)
{
    for (int n : {1, 2, 5, 32, 34})
    {
        std::vector<uint16_t> idx;
        terrainGridTriangles(n, idx);
        ASSERT_EQ(idx.size(), size_t(n) * n * 6);
        auto pos = [n](uint16_t v) { return std::pair<int, int>(v % (n + 1), v / (n + 1)); };
        for (int cell = 0; cell < n * n; ++cell)
        {
            const int i = cell % n, j = cell / n;
            for (int t = 0; t < 2; ++t)
            {
                const uint16_t* tri = &idx[size_t(cell) * 6 + t * 3];
                bool has00 = false, has11 = false;
                for (int v = 0; v < 3; ++v)
                {
                    auto [x, z] = pos(tri[v]);
                    ASSERT_TRUE(x == i || x == i + 1);
                    ASSERT_TRUE(z == j || z == j + 1);
                    has00 |= x == i && z == j;
                    has11 |= x == i + 1 && z == j + 1;
                }
                // Both triangles share the diagonal (i, j)-(i+1, j+1)
                EXPECT_TRUE(has00 && has11);
                // Counter-clockwise from above: the cross product of the edges points to +y
                auto [x0, z0] = pos(tri[0]);
                auto [x1, z1] = pos(tri[1]);
                auto [x2, z2] = pos(tri[2]);
                const int ax = x1 - x0, az = z1 - z0, bx = x2 - x0, bz = z2 - z0;
                // (a x b).y = az * bx - ax * bz
                EXPECT_GT(az * bx - ax * bz, 0);
            }
        }
    }
}
