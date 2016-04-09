#pragma once

#include <string>
#include "saiga/config.h"
#include "saiga/util/glm.h"


struct SAIGA_GLOBAL VertexNT{
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



struct SAIGA_GLOBAL VertexN{
    vec3 position;
    vec3 normal;
    VertexN() : position(0),normal(0){}
    VertexN(const vec3 &position):position(position){}
    VertexN(const vec3 &position,const vec3 &normal):position(position),normal(normal){}
    VertexN(const VertexNT& v):position(v.position),normal(v.normal){}

    bool operator==(const VertexN &other) const {
        return position==other.position && normal==other.normal;
    }
    friend std::ostream& operator<<(std::ostream& os, const VertexN& vert){
        os<<vert.position<<",";
        os<<vert.normal;
        return os;
    }
};


struct SAIGA_GLOBAL Vertex{
    vec3 position;
    Vertex() : position(0){}
    Vertex(const vec3 &position):position(position){}
    Vertex(const VertexN& v):position(v.position){}
    Vertex(const VertexNT& v):position(v.position){}

    bool operator==(const Vertex &other) const {
        return position==other.position;
    }


    friend std::ostream& operator<<(std::ostream& os, const Vertex& vert){
        os<<vert.position;
        return os;
    }
};

struct SAIGA_GLOBAL VertexNC : public VertexN{
    vec3 color;
    vec3 data;
    VertexNC() : color(0){}
    VertexNC(const vec3 &position):VertexN(position){}
    VertexNC(const vec3 &position,const vec3 &normal):VertexN(position,normal){}
    VertexNC(const vec3 &position,const vec3 &normal,const vec3 &color):VertexN(position,normal),color(color){}
    bool operator==(const VertexNC &other) const {
        return position==other.position && normal==other.normal && color==other.color && data==other.data;
    }
    friend std::ostream& operator<<(std::ostream& os, const VertexNC& vert){
        os<<vert.position<<",";
        os<<vert.normal<<",";
        os<<vert.color<<",";
        os<<vert.data;
        return os;
    }
};




