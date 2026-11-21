/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <string>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
struct MeshImportOptions
{
    // Palette colors (sRGB8 RGB; alpha not compared): the color words become the index of the exact RGB match, a miss
    // is an error; nullptr: the color words are sRGB8 RGB + linear alpha
    const std::vector<ucvec4>* palette = nullptr;
};

/**
 * Imports a static mesh with the flat-color model (S18, docs/engine_design.md §10.8, §14) from glTF 2.0 (.gltf, .glb)
 * through assimp:
 * - LODs: the nodes whose names end in LOD0, LOD1, ... (e.g. Tree_LOD0; contiguous from LOD0, at most 4; once a LOD
 *   node exists, mesh nodes outside the LOD nodes are an error), or the whole scene as one LOD. All primitives under a
 *   LOD node are merged with the node transforms baked (a negative determinant flips the winding).
 * - Triangles only (strips and fans are converted by assimp; points and lines are errors). Every face gets its own
 *   three vertices with the face normal of the baked positions; zero-area faces are dropped.
 * - The color word: COLOR_0 * baseColorFactor (both linear) as sRGB8 RGB (rounded to nearest) + linear alpha; alpha
 *   is 255 unless the material's alphaMode is MASK (a COLOR_0 whose alphas are all 0 is a VEC3 color: assimp 5.2.5
 *   delivers alpha 0 for them). With a palette the index of the exact RGB entry. TEXCOORD_0 into uv (zero without).
 * - One material (material may be nullptr): usePalette when a palette is given, baseColor 1 (the factors are baked
 *   into the color words), alphaMask from alphaMode MASK, twoSided from doubleSided, emissive, roughness and metalness
 *   from the glTF material; merged primitives whose materials differ in any of these are an error naming both.
 * - The bounding sphere over all LODs. End distances are not in glTF: the LODs get placeholders (LOD i: (i + 1) *
 *   1000), the game sets them (R-MECH-1).
 * Errors (no mesh, too many vertices, non-triangles, a palette miss, ...) are messages, not asserts: false and error.
 * Without assimp (SAIGA_USE_ASSIMP undefined) it returns false with an error.
 */
SAIGA_VULKAN_API bool importMeshLods(const std::string& file, MeshLods& out, MaterialData* material, std::string& error,
                                     const MeshImportOptions& options = {});

}  // namespace Vulkan
}  // namespace Saiga
