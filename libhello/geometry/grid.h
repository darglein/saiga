#pragma once

#include "libhello/opengl/vertex.h"
#include "libhello/opengl/vertexBuffer.h"
#include "libhello/util/glm.h"
#include "libhello/geometry/plane.h"

#include <vector>

class Grid : public Plane
{
public:
    vec3 d1,d2,mid;
    Grid(const vec3 &mid,const vec3 &d1, const vec3 &d2);
     void addToBuffer(std::vector<VertexN> &vertices,std::vector<GLuint> &indices, int linesX, int linesY);
      void createBuffers(VertexBuffer<VertexN> &buffer, int linesX, int linesY);
};


