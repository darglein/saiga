/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "LineModelColored.h"
#include "saiga/camera/camera.h"

#include "internal/noGraphicsAPI.h"

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
        ver.color = color;
        mesh.vertices.push_back(ver);

    }
}

void LineModelColored::createFrustum(const mat4 &proj, float farPlaneLimit, const vec4 &color, bool vulkanTransform)
{
    float d = 1.0f;
    vec4 bl(-1,-1,d,1);
    vec4 br(1,-1,d,1);
    vec4 tl(-1,1,d,1);
    vec4 tr(1,1,d,1);

    mat4 projInv = vulkanTransform ? inverse(inverse(Camera::getVulkanTransform())*proj) : inverse(proj);



    tl = projInv * tl;
    tr = projInv * tr;
    bl = projInv * bl;
    br = projInv * br;

    tl /= tl.w;
    tr /= tr.w;
    bl /= bl.w;
    br /= br.w;

    if(farPlaneLimit > 0)
    {
        tl.w = -tl.z / farPlaneLimit;
        tr.w = -tr.z / farPlaneLimit;
        bl.w = -bl.z / farPlaneLimit;
        br.w = -br.z / farPlaneLimit;

        tl /= tl.w;
        tr /= tr.w;
        bl /= bl.w;
        br /= br.w;
    }


    //    std::vector<VertexNC> vertices;

    vec4 positions[] = {
        vec4(0,0,0,1),
        tl,tr,br,bl,
        0.4f * tl + 0.6f * tr,
        0.6f * tl + 0.4f * tr,
        0.5f * tl + 0.5f * tr + vec4(0,(tl.y-bl.y)*0.1f,0,0)
    };

    for(int i = 0 ; i < 8 ; ++i){
                VertexNC v;
                v.position = positions[i];
                v.color = color;
        mesh.vertices.push_back(v);
    }


    mesh.indices =
     {
        0,1,
        0,2,
        0,3,
        0,4,

        1,2,
        3,4,
        1,4,
        2,3,

        5,7,
        6,7
    };
}


void LineModelColored::createFrustumCV(const mat3 &K, float farPlaneLimit, const vec4 &color, int w, int h)
{
    vec3 bl(0,h,1);
    vec3 br(w,h,1);
    vec3 tl(0,0,1);
    vec3 tr(w,0,1);

    mat3 projInv = inverse(K);

    tl = projInv * tl;
    tr = projInv * tr;
    bl = projInv * bl;
    br = projInv * br;


    if(farPlaneLimit > 0)
    {
        tl *= farPlaneLimit;
        tr *= farPlaneLimit;
        bl *= farPlaneLimit;
        br *= farPlaneLimit;
    }

    vec3 positions[] = {
        vec3(0,0,0),
        tl,tr,br,bl,
        0.4f * tl + 0.6f * tr,
        0.6f * tl + 0.4f * tr,
        0.5f * tl + 0.5f * tr + vec3(0,(tl.y-bl.y)*0.1f,0)
    };

    for(int i = 0 ; i < 8 ; ++i){
                VertexNC v;
                v.position = vec4(positions[i],1);
                v.color = color;
        mesh.vertices.push_back(v);
    }

    mesh.indices =
     {
        0,1,
        0,2,
        0,3,
        0,4,

        1,2,
        3,4,
        1,4,
        2,3,

        5,7,
        6,7
    };
}


}






