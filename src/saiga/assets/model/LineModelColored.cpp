/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "LineModelColored.h"

#if defined(SAIGA_VULKAN_INCLUDED) || defined(SAIGA_OPENGL_INCLUDED)
//#error This module must be independent of any graphics API.
#endif

namespace Saiga {

void LineModelColored::createGrid(int numX, int numY, float quadSize, vec4 color)
{
    vec2 size = vec2(numX,numY) * quadSize;

    std::vector<vec3> vertices;

    for(float i=-numX;i<=numX;i++)
    {
        vec3 p1 = vec3(quadSize*i,0,-size.y);
        vec3 p2 = vec3(quadSize*i,0,size.y);
        mesh.indices.push_back(vertices.size());
        vertices.push_back(p1);
        mesh.indices.push_back(vertices.size());
        vertices.push_back(p2);
    }

    for(float i=-numY;i<=numY;i++)
    {
        vec3 p1 = vec3(-size.x,0,quadSize*i);
        vec3 p2 = vec3(+size.x,0,quadSize*i);
        mesh.indices.push_back(vertices.size());
        vertices.push_back(p1);
        mesh.indices.push_back(vertices.size());
        vertices.push_back(p2);
    }


    for(auto v : vertices)
    {
        VertexNC ver(v);
        mesh.vertices.push_back(ver);

    }
}


}






