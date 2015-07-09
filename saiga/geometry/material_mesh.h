#pragma once

#include "saiga/geometry/triangle_mesh.h"
#include "saiga/rendering/material.h"


template<typename vertex_t, typename index_t>
class SAIGA_GLOBAL MaterialMesh : public TriangleMesh<vertex_t,index_t>
{
public:
    std::string name;
    std::vector<TriangleGroup> triangleGroups;
};
