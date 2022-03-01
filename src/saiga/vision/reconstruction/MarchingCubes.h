/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"

#include <array>


namespace Saiga
{
//  Given a grid cell and an isolevel, calculate the triangular facets required to represent the isosurface through the
//  cell. The maximum number of triangles is 16.
// Return: <Triangles, num_triangles>
SAIGA_VISION_API std::pair<std::array<std::array<vec3, 3>, 16>, int> MarchingCubes(
    const std::array<std::pair<vec3, float>, 8>& cell, float isolevel);

SAIGA_VISION_API std::vector<std::array<vec3, 3>> MarchingCubes(float* data, int depth, int height, int width,
                                                                float isolevel);
}  // namespace Saiga
