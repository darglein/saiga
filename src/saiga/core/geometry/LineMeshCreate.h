/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/CoordinateSystems.h"

#include "LineMesh.h"
namespace Saiga
{
template <typename VertexType, typename IndexType>
void LineMesh<VertexType, IndexType>::createGrid(const ivec2& dimension, const vec2& spacing)
{
    vec2 size = dimension.cast<float>().array() * spacing.array();


    std::vector<vec3> vertices;

    for (float i = -dimension.x(); i <= dimension.x(); i++)
    {
        vec3 p1 = vec3(spacing.x() * i, 0, -size[1]);
        vec3 p2 = vec3(spacing.x() * i, 0, size[1]);
        indices.push_back(vertices.size());
        vertices.push_back(p1);
        indices.push_back(vertices.size());
        vertices.push_back(p2);
    }

    for (float i = -dimension.y(); i <= dimension.y(); i++)
    {
        vec3 p1 = vec3(-size[0], 0, spacing.y() * i);
        vec3 p2 = vec3(+size[0], 0, spacing.y() * i);
        indices.push_back(vertices.size());
        vertices.push_back(p1);
        indices.push_back(vertices.size());
        vertices.push_back(p2);
    }


    for (auto v : vertices)
    {
        VertexType ver;
        ver.position = make_vec4(v, 1);
        this->vertices.push_back(ver);
    }
}

template <typename VertexType, typename IndexType>
void LineMesh<VertexType, IndexType>::createFrustum(const mat4& proj, float farPlaneDistance, bool vulkanTransform)
{
    float d = 1.0f;
    vec4 bl(-1, -1, d, 1);
    vec4 br(1, -1, d, 1);
    vec4 tl(-1, 1, d, 1);
    vec4 tr(1, 1, d, 1);

    mat4 tmp     = (inverse(GL2VulkanNormalizedImage()) * proj);
    mat4 projInv = vulkanTransform ? inverse(tmp) : inverse(proj);



    tl = projInv * tl;
    tr = projInv * tr;
    bl = projInv * bl;
    br = projInv * br;

    tl /= tl[3];
    tr /= tr[3];
    bl /= bl[3];
    br /= br[3];

    if (farPlaneDistance > 0)
    {
        tl[3] = -tl[2] / farPlaneDistance;
        tr[3] = -tr[2] / farPlaneDistance;
        bl[3] = -bl[2] / farPlaneDistance;
        br[3] = -br[2] / farPlaneDistance;

        tl /= tl[3];
        tr /= tr[3];
        bl /= bl[3];
        br /= br[3];
    }


    //    std::vector<VertexNC> vertices;

    vec4 positions[] = {vec4(0, 0, 0, 1),
                        tl,
                        tr,
                        br,
                        bl,
                        0.4f * tl + 0.6f * tr,
                        0.6f * tl + 0.4f * tr,
                        0.5f * tl + 0.5f * tr + vec4(0, (tl[1] - bl[1]) * 0.1f, 0, 0)};

    for (int i = 0; i < 8; ++i)
    {
        VertexType v;
        v.position = positions[i];

        vertices.push_back(v);
    }


    indices = {0, 1, 0, 2, 0, 3, 0, 4,

               1, 2, 3, 4, 1, 4, 2, 3,

               5, 7, 6, 7};
}

template <typename VertexType, typename IndexType>
void LineMesh<VertexType, IndexType>::createFrustumCV(float farPlaneLimit, int w, int h)
{
    mat3 K  = mat3::Identity();
    K(0, 2) = w / 2.0;
    K(1, 2) = h / 2.0;
    K(0, 0) = w;
    K(1, 1) = w;
    createFrustumCV(K, farPlaneLimit, w, h);
}

template <typename VertexType, typename IndexType>
void LineMesh<VertexType, IndexType>::createFrustumCV(const mat3& K, float farPlaneDistance, int w, int h)
{
    vec3 bl(0, h, 1);
    vec3 br(w, h, 1);
    vec3 tl(0, 0, 1);
    vec3 tr(w, 0, 1);

    mat3 projInv = inverse(K);

    tl = projInv * tl;
    tr = projInv * tr;
    bl = projInv * bl;
    br = projInv * br;


    if (farPlaneDistance > 0)
    {
        tl *= farPlaneDistance;
        tr *= farPlaneDistance;
        bl *= farPlaneDistance;
        br *= farPlaneDistance;
    }

    vec3 positions[] = {vec3(0, 0, 0),
                        tl,
                        tr,
                        br,
                        bl,
                        0.4f * tl + 0.6f * tr,
                        0.6f * tl + 0.4f * tr,
                        0.5f * tl + 0.5f * tr + vec3(0, (tl[1] - bl[1]) * 0.1f, 0)};

    for (int i = 0; i < 8; ++i)
    {
        VertexType v;
        v.position = make_vec4(positions[i], 1);
        vertices.push_back(v);
    }

    indices = {0, 1, 0, 2, 0, 3, 0, 4,

               1, 2, 3, 4, 1, 4, 2, 3,

               5, 7, 6, 7};
}
}  // namespace Saiga
