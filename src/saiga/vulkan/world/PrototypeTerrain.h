/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The world prototype's terrain as a height hook (S20, docs/engine_design.md §12.2, A5, A6; the prototype's game side,
// no engine types, D2): the S01 noise evaluated per point, the surface rule over it for code without a World (the
// flight and racing contexts, the tree placement, tests), and the statistics of an area.

#pragma once

#include "saiga/core/math/TerrainSurface.h"
#include "saiga/core/math/math.h"
#include "saiga/export.h"

#include <cstdint>

namespace Saiga
{
namespace Vulkan
{
// The sea level of the prototype (height units: 6 m)
constexpr int32_t kPrototypeSeaLevel = 6144;
// The rough height of its mountains (m)
constexpr double kPrototypeMountainHeight = 520;
// The prototype's space: 128 chunks (32 km); its content (trees, benchmarks) lies in the 8 km area [origin, origin +
// kPrototypeArea]^2 at the space's centre (S36: it was in the space's corner [0, 8192]^2, the start 3 km from the edge)
constexpr int32_t kPrototypeSpaceChunks = 128;
constexpr double kPrototypeArea         = 8192;
constexpr double kPrototypeAreaOrigin   = 0.5 * (double(kPrototypeSpaceChunks) * 256.0 - kPrototypeArea);
constexpr double kPrototypeAreaCentre   = kPrototypeAreaOrigin + 0.5 * kPrototypeArea;

// The prototype's height hook (a TerrainHeightHook): the S01 noise of the old HeightField::generate, per point, with
// the noise seed uint32_t(seed) * 7717 and the border falloff of 819.2 m at the edges of the 32 km space (A5: the old
// falloff was 10 % of an 8 km world). clampTerrainHeight(lround(h * 1024)). Pure and thread safe; about 0.7 us
SAIGA_VULKAN_API int32_t prototypeHeight(uint64_t seed, int64_t x, int64_t z);

// The surface height (m) of a space of kPrototypeSpaceChunks with this hook and base spacing (1 or 2), equal to
// World::terrainHeight of such a space bit for bit (the same clamp to the space, cell rule and diagonal). Three hook
// calls (about 2 us)
SAIGA_VULKAN_API double prototypeSurfaceHeight(uint64_t seed, int baseSpacing, double x, double z);

// The statistics of an area on a grid of hook samples (step metres): the highest sample and the min / max height (m)
struct PrototypeAreaStats
{
    Vec3 highest     = Vec3::Zero();
    double minHeight = 0, maxHeight = 0;
};
SAIGA_VULKAN_API PrototypeAreaStats prototypeAreaStats(uint64_t seed, double x0, double z0, double size, double step);

// The area of the prototype's content (trees) in the space: [x0, x0 + size] x [z0, z0 + size]; seed: the hook's
struct PrototypeArea
{
    double x0 = kPrototypeAreaOrigin, z0 = kPrototypeAreaOrigin, size = kPrototypeArea;
    uint64_t seed   = 1;
    int baseSpacing = 1;
};

}  // namespace Vulkan
}  // namespace Saiga
