/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/export.h"
#include "saiga/vulkan/world/WorldNoise.h"

#include <cstdint>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
/**
 * A square height map of the terrain. Sample (i, j) is at the world position (i * spacing, height, j * spacing), so
 * the world covers [0, worldSize()] in x and z.
 *
 * height() interpolates like the terrain mesh: each cell is split into the triangles (i,j) (i+1,j) (i,j+1) and
 * (i+1,j) (i,j+1) (i+1,j+1), so objects placed with it sit exactly on the rendered surface (also in the shaders, see
 * shader/vulkan/world/heightmap.glsl).
 */
class SAIGA_VULKAN_API HeightField
{
   public:
    struct Parameters
    {
        // Number of samples per side. 2^n + 1 for the terrain quadtree.
        int size = 4097;
        // Distance between two samples in meters
        float spacing = 2.0f;
        uint32_t seed = 1;
        // Rough height of the mountains (m)
        float mountainHeight = 520.0f;
        // Height of the water plane (m). The terrain falls below it at the border of the world.
        float waterLevel = 6.0f;
    };

    // Generates the terrain with noise (multi threaded).
    void generate(const Parameters& params);

    int size() const { return n; }
    float spacing() const { return params.spacing; }
    float worldSize() const { return (n - 1) * params.spacing; }
    float waterLevel() const { return params.waterLevel; }
    const Parameters& parameters() const { return params; }

    float sample(int i, int j) const { return data[size_t(clampIndex(j)) * n + clampIndex(i)]; }
    const std::vector<float>& heights() const { return data; }

    // The height of the terrain surface at the world position (clamped to the border).
    float height(float x, float z) const;
    // Smooth normal (central differences with the step size in meters)
    vec3 normal(float x, float z, float step = 2.0f) const;

    float minHeight() const { return minH; }
    float maxHeight() const { return maxH; }
    // The world position of the highest sample
    vec3 highestPoint() const { return highest; }

   private:
    Parameters params;
    int n = 0;
    std::vector<float> data;
    float minH = 0, maxH = 0;
    vec3 highest = vec3(0, 0, 0);

    int clampIndex(int i) const { return std::min(std::max(i, 0), n - 1); }
};

}  // namespace Vulkan
}  // namespace Saiga
