#pragma once

#include <string>
#include "libhello/util/glm.h"

struct Vertex{
    vec3 position;
    Vertex() : position(0){}
    Vertex(const vec3 &position):position(position){}

    bool operator==(const Vertex &other) const {
        return position==other.position;
    }


    friend std::ostream& operator<<(std::ostream& os, const Vertex& vert){
          os<<vert.position;
          return os;
    }
};

struct VertexN{
    vec3 position;
    vec3 normal;
    VertexN() : position(0),normal(0){}
    VertexN(const vec3 &position):position(position){}
    VertexN(const vec3 &position,const vec3 &normal):position(position),normal(normal){}
    bool operator==(const VertexN &other) const {
        return position==other.position && normal==other.normal;
    }
    friend std::ostream& operator<<(std::ostream& os, const VertexN& vert){
        os<<vert.position<<",";
          os<<vert.normal;
          return os;
    }
};

struct VertexNT{
    vec3 position;
    vec3 normal;
    vec2 texture;
    VertexNT() : position(0),normal(0),texture(0){}
    VertexNT(const vec3 &position):position(position){}
    VertexNT(const vec3 &position,const vec3 &normal,const vec2 &texture):position(position),normal(normal),texture(texture){}
    bool operator==(const VertexNT &other) const {
        return position==other.position && normal==other.normal && texture==other.texture;
    }
    friend std::ostream& operator<<(std::ostream& os, const VertexNT& vert){
        os<<vert.position<<",";
          os<<vert.normal<<",";
          os<<vert.texture;
          return os;
    }
};




