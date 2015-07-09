#pragma once

#include <algorithm>
#include <vector>
//#include "libhello/rendering/particles/particle.h"
#include "libhello/geometry/aabb.h"
#include "libhello/geometry/triangle.h"

class SAIGA_GLOBAL Particleconverter
{
public:
    Particleconverter(){}

    void convert(std::vector<Triangle> &triangles, std::vector<vec3> &points);

    aabb getBoundingBox(std::vector<Triangle> &triangles);

private:
    void voxelizeRange(std::vector<vec3> &points, vec3 start, vec3 end);
};


