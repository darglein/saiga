/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/opengl/vertex.h"
#include "saiga/opengl/vertexBuffer.h"
#include "saiga/util/glm.h"
#include "saiga/geometry/plane.h"

#include <vector>

namespace Saiga {

class SAIGA_GLOBAL Grid : public Plane
{
public:
    vec3 d1,d2,mid;
    Grid(const vec3 &mid,const vec3 &d1, const vec3 &d2);
     void addToBuffer(std::vector<VertexN> &vertices,std::vector<GLuint> &indices, int linesX, int linesY);
      void createBuffers(VertexBuffer<VertexN> &buffer, int linesX, int linesY);
};

}
