/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "triangle_mesh_generator2.h"

#include "internal/noGraphicsAPI.h"
namespace Saiga
{
UnifiedModel FullScreenQuad()
{
    UnifiedModel model;

    model.position            = {vec3(-1, -1, 0), vec3(1, -1, 0), vec3(1, 1, 0), vec3(-1, 1, 0)};
    model.normal              = {vec3(0, 0, 1), vec3(0, 0, 1), vec3(0, 0, 1), vec3(0, 0, 1)};
    model.texture_coordinates = {vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)};

    model.triangles = {ivec3(0, 1, 2), ivec3(0, 2, 3)};

    return model;
}

}  // namespace Saiga
