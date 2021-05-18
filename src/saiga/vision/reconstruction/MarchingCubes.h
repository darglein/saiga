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
std::pair<std::array<std::array<vec3, 3>, 16>, int> SAIGA_VISION_API
MarchingCubes(const std::array<std::pair<vec3, float>, 8>& cell, float isolevel);

}  // namespace Saiga
