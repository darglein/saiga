/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/geometry/grid.h"

namespace Saiga {

Grid::Grid(const vec3 &mid, const vec3 &d1, const vec3 &d2) : d1(d1),d2(d2),mid(mid)
{
    set(mid,glm::cross(d1,d2));
}

void Grid::addToBuffer(std::vector<VertexN> &vertices,std::vector<GLuint> &indices, int linesX, int linesY){
    vec3 xOffset = d2*(float)(linesY-1)*2.0f;
    for(float i=-linesX+1;i<linesX;i++){
        vec3 pos = mid+d1*i- xOffset*0.5f;
        indices.push_back(vertices.size());
        vertices.push_back(VertexN(pos,normal));
        indices.push_back(vertices.size());
        vertices.push_back(VertexN(pos+xOffset,normal));
    }

    vec3 yOffset = d1*(float)(linesX-1)*2.0f;
    for(float i=-linesY+1;i<linesY;i++){
        vec3 pos = mid+d2*i- yOffset*0.5f;
        indices.push_back(vertices.size());
        vertices.push_back(VertexN(pos,normal));
        indices.push_back(vertices.size());
        vertices.push_back(VertexN(pos+yOffset,normal));
    }
}

void Grid::createBuffers(VertexBuffer<VertexN> &buffer, int linesX, int linesY){

    std::vector<VertexN> vertices;

    vec3 xOffset = d2*(float)(linesY-1)*2.0f;
    for(float i=-linesX+1;i<linesX;i++){
        vec3 pos = mid+d1*i- xOffset*0.5f;
        vertices.push_back(VertexN(pos,normal));
        vertices.push_back(VertexN(pos+xOffset,normal));
    }

    vec3 yOffset = d1*(float)(linesX-1)*2.0f;
    for(float i=-linesY+1;i<linesY;i++){
        vec3 pos = mid+d2*i- yOffset*0.5f;
        vertices.push_back(VertexN(pos,normal));
        vertices.push_back(VertexN(pos+yOffset,normal));
    }



    buffer.set(vertices,GL_STATIC_DRAW);
    buffer.setDrawMode(GL_LINES);
}

}
