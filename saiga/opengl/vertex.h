#pragma once

#include <string>
#include "saiga/config.h"
#include "saiga/util/glm.h"
#include "saiga/opengl/vertexBuffer.h"

/**
 * Using inheritance here to enable an easy conversion between vertex types by object slicing.
 * Example:
 * VertexN v;
 * Vertex v2 = v;
 */

struct SAIGA_GLOBAL Vertex {
    vec4 position = vec4(0);

    Vertex() {}
    Vertex(const vec3 &position) : position(position,1){}
    Vertex(const vec4 &position) : position(position){}

    bool operator==(const Vertex &other) const;
    friend std::ostream& operator<<(std::ostream& os, const Vertex& vert);
} GLM_ALIGN(16);

struct SAIGA_GLOBAL VertexN : public Vertex{
    vec4 normal =  vec4(0);

    VertexN() {}
    VertexN(const vec3 &position) : Vertex(position){}
    VertexN(const vec4 &position) : Vertex(position){}
    VertexN(const vec3 &position,const vec3 &normal) : Vertex(position),normal(normal,0){}
    VertexN(const vec4 &position,const vec4 &normal) : Vertex(position),normal(normal){}

    bool operator==(const VertexN &other) const;
    friend std::ostream& operator<<(std::ostream& os, const VertexN& vert);
} GLM_ALIGN(16);


struct SAIGA_GLOBAL VertexNT : public VertexN{
    vec2 texture = vec2(0);

    VertexNT() {}
    VertexNT(const vec3 &position) : VertexN(position){}
    VertexNT(const vec4 &position) : VertexN(position){}
    VertexNT(const vec3 &position,const vec3 &normal) : VertexN(position,normal){}
    VertexNT(const vec3 &position,const vec3 &normal,const vec2 &texture) : VertexN(position,normal),texture(texture){}
    VertexNT(const vec4 &position,const vec4 &normal,const vec2 &texture) : VertexN(position,normal),texture(texture){}

    bool operator==(const VertexNT &other) const;
    friend std::ostream& operator<<(std::ostream& os, const VertexNT& vert);
} GLM_ALIGN(16);


struct SAIGA_GLOBAL VertexNC : public VertexN{
    vec4 color = vec4(0);
    vec4 data = vec4(0);

    VertexNC() {}
    VertexNC(const vec3 &position) : VertexN(position){}
    VertexNC(const vec4 &position) : VertexN(position){}
    VertexNC(const vec3 &position,const vec3 &normal) : VertexN(position,normal){}
    VertexNC(const vec4 &position,const vec4 &normal) : VertexN(position,normal){}
    VertexNC(const vec3 &position,const vec3 &normal,const vec3 &color) : VertexN(position,normal),color(color,0){}
    VertexNC(const vec4 &position,const vec4 &normal,const vec4 &color) : VertexN(position,normal),color(color){}

    bool operator==(const VertexNC &other) const;
    friend std::ostream& operator<<(std::ostream& os, const VertexNC& vert);
} GLM_ALIGN(16);



template<>
SAIGA_GLOBAL void VertexBuffer<Vertex>::setVertexAttributes();
template<>
SAIGA_GLOBAL void VertexBuffer<VertexN>::setVertexAttributes();
template<>
SAIGA_GLOBAL void VertexBuffer<VertexNT>::setVertexAttributes();
template<>
SAIGA_GLOBAL void VertexBuffer<VertexNC>::setVertexAttributes();

