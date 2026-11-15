/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/templatedImage.h"
#include "saiga/core/math/math.h"
#include "saiga/export.h"

#include <array>
#include <cstdint>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
/**
 * Vertex of the trees and the grass (48 bytes):
 *   posAo:      position (tree: unit height, scaled per instance), ambient occlusion
 *   normalWind: shading normal, wind weight (0: rigid, 1: moves the most)
 *   uvLayer:    texture coordinate, layer of the foliage texture (-1: opaque, no texture), unused
 */
struct TreeVertex
{
    vec4 posAo;
    vec4 normalWind;
    vec4 uvLayer;
};

struct FoliageMesh
{
    std::vector<TreeVertex> vertices;
    std::vector<uint32_t> indices;
    int triangles() const { return int(indices.size() / 3); }
};

// The layers of the foliage alpha texture
enum FoliageLayer
{
    FOLIAGE_BROADLEAF = 0,
    FOLIAGE_NEEDLES   = 1,
    FOLIAGE_GRASS     = 2,
    FOLIAGE_LAYERS    = 3,
};

/**
 * One tree species: a mesh per level of detail (0: solid trunk and canopy core + many alpha tested leaf cards,
 * 1: fewer cards, 2: a few solid triangles). The meshes are 1 unit high, the instances scale them to their height in
 * meters. bounds: (center y, radius) of a sphere around all levels, in units of the height.
 */
struct TreeSpecies
{
    const char* name;
    std::array<FoliageMesh, 3> lods;
    vec2 bounds;
    // Height range of the instances in meters
    vec2 heightRange;
};

/**
 * Procedural low-poly content of the world (deterministic for a seed).
 */
struct SAIGA_VULKAN_API FoliageGenerator
{
    // Spruce, broadleaf and birch
    static std::vector<TreeSpecies> treeSpecies(uint32_t seed);

    // A clump of grass: three crossing cards with the grass layer (unit size).
    static FoliageMesh grassClump();

    // Alpha masks (UC1, white = opaque) of the foliage layers, row 0 is the top (v = 0).
    static std::vector<TemplatedImage<unsigned char>> foliageTextures(int size, uint32_t seed);
};

}  // namespace Vulkan
}  // namespace Saiga
