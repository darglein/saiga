/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/Align.h"
namespace Saiga
{
/**
 * Using inheritance here to enable an easy conversion between vertex types by object slicing.
 * Example:
 * VertexN v;
 * Vertex v2 = v;
 */

struct SAIGA_CORE_API Vertex
{
    vec4 position = make_vec4(0);

    Vertex() {}
    Vertex(const vec3& position) : position(make_vec4(position, 1)) {}
    Vertex(const vec4& position) : position(position) {}

    bool operator==(const Vertex& other) const;
    SAIGA_CORE_API friend std::ostream& operator<<(std::ostream& os, const Vertex& vert);
};

struct SAIGA_CORE_API VertexN : public Vertex
{
    vec4 normal = make_vec4(0);

    VertexN() {}
    VertexN(const vec3& position) : Vertex(position) {}
    VertexN(const vec4& position) : Vertex(position) {}
    VertexN(const vec3& position, const vec3& normal) : Vertex(position), normal(make_vec4(normal, 0)) {}
    VertexN(const vec4& position, const vec4& normal) : Vertex(position), normal(normal) {}

    bool operator==(const VertexN& other) const;
    SAIGA_CORE_API friend std::ostream& operator<<(std::ostream& os, const VertexN& vert);
};

struct SAIGA_CORE_API VertexC : public Vertex
{
    vec4 color = make_vec4(1);
    VertexC() {}
    VertexC(const vec3& position) : Vertex(position) {}
    VertexC(const vec4& position) : Vertex(position) {}
    VertexC(const vec3& position, const vec3& color) : Vertex(position), color(make_vec4(color, 1)) {}
    VertexC(const vec4& position, const vec4& color) : Vertex(position), color(color) {}
};


struct SAIGA_CORE_API VertexNT : public VertexN
{
    vec2 texture = make_vec2(0);
    vec2 padding = make_vec2(0);

    VertexNT() {}
    VertexNT(const vec3& position) : VertexN(position) {}
    VertexNT(const vec4& position) : VertexN(position) {}
    VertexNT(const vec3& position, const vec3& normal) : VertexN(position, normal) {}
    VertexNT(const vec3& position, const vec3& normal, const vec2& texture)
        : VertexN(position, normal), texture(texture)
    {
    }
    VertexNT(const vec4& position, const vec4& normal, const vec2& texture)
        : VertexN(position, normal), texture(texture)
    {
    }

    bool operator==(const VertexNT& other) const;
    SAIGA_CORE_API friend std::ostream& operator<<(std::ostream& os, const VertexNT& vert);
};


struct SAIGA_CORE_API VertexNC : public VertexN
{
    vec4 color = make_vec4(0);
    vec4 data  = make_vec4(0);

    VertexNC() {}
    VertexNC(const vec3& position) : VertexN(position) {}
    VertexNC(const vec4& position) : VertexN(position) {}
    VertexNC(const vec3& position, const vec3& normal) : VertexN(position, normal) {}
    VertexNC(const vec4& position, const vec4& normal) : VertexN(position, normal) {}
    VertexNC(const vec3& position, const vec3& normal, const vec3& color)
        : VertexN(position, normal), color(make_vec4(color, 0))
    {
    }
    VertexNC(const vec4& position, const vec4& normal, const vec4& color) : VertexN(position, normal), color(color) {}
    VertexNC(const VertexNT& v) : VertexNC(v.position, v.normal) {}

    bool operator==(const VertexNC& other) const;
    SAIGA_CORE_API friend std::ostream& operator<<(std::ostream& os, const VertexNC& vert);
};



/**
 * We are using a maximum number of 4 bones per vertex here, because it fits nicely in a vec4 on the gpu
 * and was sufficient in all cases I have encountered so far.
 */


constexpr int MAX_BONES_PER_VERTEX = 4;

using BoneIndices = std::array<int, MAX_BONES_PER_VERTEX>;
using BoneWeights = std::array<float, MAX_BONES_PER_VERTEX>;

struct SAIGA_CORE_API BoneInfo
{
    BoneIndices bone_indices;
    BoneWeights bone_weights;

    BoneInfo()
    {
        for (int i = 0; i < MAX_BONES_PER_VERTEX; ++i)
        {
            bone_indices[i] = 0;
            bone_weights[i] = 0;
        }
    }

    // add a bone with given index and weight to this vertex
    void addBone(int32_t index, float weight);

    // normalizes the weights so that the sum is 1.
    void normalizeWeights();

    // number of bones with weight > 0
    int activeBones();
};




}  // namespace Saiga
