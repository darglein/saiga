/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/geometry/vertex.h"

#if defined(SAIGA_VULKAN_INCLUDED) || defined(SAIGA_OPENGL_INCLUDED)
#error This module must be independent of any graphics API.
#endif

namespace Saiga {


bool Vertex::operator==(const Vertex &other) const {
    return position==other.position;
}

std::ostream &operator<<(std::ostream &os, const Vertex &vert){
    os<<vert.position;
    return os;
}

bool VertexN::operator==(const VertexN &other) const {
    return Vertex::operator==(other) && normal == other.normal;
}

std::ostream &operator<<(std::ostream &os, const VertexN &vert){
    os<<vert.position<<",";
    os<<vert.normal;
    return os;
}

bool VertexNT::operator==(const VertexNT &other) const {
    return VertexN::operator==(other) && texture == other.texture;
}

std::ostream &operator<<(std::ostream &os, const VertexNT &vert){
    os<<vert.position<<",";
    os<<vert.normal<<",";
    os<<vert.texture;
    return os;
}

bool VertexNTD::operator==(const VertexNTD &other) const {
	return VertexNT::operator==(other) && data == other.data;
}

std::ostream &operator<<(std::ostream &os, const VertexNTD &vert){
    os<<vert.position<<",";
    os<<vert.normal<<",";
    os<<vert.texture<<",";
    os<<vert.data;
    return os;
}

bool VertexNC::operator==(const VertexNC &other) const {
    return VertexN::operator==(other) && color == other.color && data == other.data;
}

std::ostream &operator<<(std::ostream &os, const VertexNC &vert){
    os<<vert.position<<",";
    os<<vert.normal<<",";
    os<<vert.color<<",";
    os<<vert.data;
    return os;
}

}
