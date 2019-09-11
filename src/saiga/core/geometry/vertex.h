/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"

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
    friend std::ostream& operator<<(std::ostream& os, const Vertex& vert);
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
    friend std::ostream& operator<<(std::ostream& os, const VertexN& vert);
};

struct SAIGA_CORE_API VertexC : public Vertex
{
    vec4 color = make_vec4(1);
};


struct SAIGA_CORE_API VertexNT : public VertexN
{
    vec2 texture = make_vec2(0);
    vec2 padding;

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
    friend std::ostream& operator<<(std::ostream& os, const VertexNT& vert);
};

struct SAIGA_CORE_API VertexNTD : public VertexNT
{
    vec4 data = make_vec4(0);

    VertexNTD() {}
    VertexNTD(const VertexNT& v) : VertexNT(v.position, v.normal, v.texture) {}
    VertexNTD(const vec3& position) : VertexNT(position) {}
    VertexNTD(const vec4& position) : VertexNT(position) {}
    VertexNTD(const vec3& position, const vec3& normal) : VertexNT(position, normal) {}
    VertexNTD(const vec3& position, const vec3& normal, const vec2& texture) : VertexNT(position, normal, texture) {}
    VertexNTD(const vec4& position, const vec4& normal, const vec2& texture) : VertexNT(position, normal, texture) {}

    bool operator==(const VertexNTD& other) const;
    friend std::ostream& operator<<(std::ostream& os, const VertexNTD& vert);
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
    friend std::ostream& operator<<(std::ostream& os, const VertexNC& vert);
};



}  // namespace Saiga
