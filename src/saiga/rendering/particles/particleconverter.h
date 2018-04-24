/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <algorithm>
#include <vector>
#include "saiga/geometry/aabb.h"
#include "saiga/geometry/triangle.h"

namespace Saiga {

class SAIGA_GLOBAL Particleconverter
{
public:
    Particleconverter(){}

    void convert(std::vector<Triangle> &triangles, std::vector<vec3> &points);

    AABB getBoundingBox(std::vector<Triangle> &triangles);

private:
    void voxelizeRange(std::vector<vec3> &points, vec3 start, vec3 end);
};

}
