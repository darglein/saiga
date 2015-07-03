#pragma once

#include "libhello/geometry/triangle_mesh.h"
#include "libhello/rendering/material.h"


template<typename vertex_t, typename index_t>
class SAIGA_GLOBAL MaterialMesh : public TriangleMesh<vertex_t,index_t>
{
public:
    std::string name;
    std::vector<TriangleGroup> triangleGroups;
};
