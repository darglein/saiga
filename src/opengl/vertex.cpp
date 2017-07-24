/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/vertex.h"

namespace Saiga {

template<>
void VertexBuffer<Vertex>::setVertexAttributes(){
    glEnableVertexAttribArray( 0 );

    glVertexAttribPointer(0,4, GL_FLOAT, GL_FALSE, sizeof(Vertex), NULL );
}


template<>
void VertexBuffer<VertexN>::setVertexAttributes(){
    glEnableVertexAttribArray( 0 );
    glEnableVertexAttribArray( 1 );

    glVertexAttribPointer(0,4, GL_FLOAT, GL_FALSE, sizeof(VertexN), NULL );
    glVertexAttribPointer(1,4, GL_FLOAT, GL_FALSE, sizeof(VertexN), (void*) (4 * sizeof(GLfloat)) );
}

template<>
void VertexBuffer<VertexNT>::setVertexAttributes(){
    glEnableVertexAttribArray( 0 );
    glEnableVertexAttribArray( 1 );
    glEnableVertexAttribArray( 2 );

    glVertexAttribPointer(0,4, GL_FLOAT, GL_FALSE, sizeof(VertexNT), NULL );
    glVertexAttribPointer(1,4, GL_FLOAT, GL_FALSE, sizeof(VertexNT), (void*) (4 * sizeof(GLfloat)) );
    glVertexAttribPointer(2,2, GL_FLOAT, GL_FALSE, sizeof(VertexNT), (void*) (8 * sizeof(GLfloat)) );
}

template<>
void VertexBuffer<VertexNTD>::setVertexAttributes(){
    glEnableVertexAttribArray( 0 );
    glEnableVertexAttribArray( 1 );
    glEnableVertexAttribArray( 2 );
    glEnableVertexAttribArray( 3 );


    glVertexAttribPointer(0,4, GL_FLOAT, GL_FALSE, sizeof(VertexNTD), NULL );
    glVertexAttribPointer(1,4, GL_FLOAT, GL_FALSE, sizeof(VertexNTD), (void*) (4 * sizeof(GLfloat)) );
    glVertexAttribPointer(2,2, GL_FLOAT, GL_FALSE, sizeof(VertexNTD), (void*) (8 * sizeof(GLfloat)) );
    glVertexAttribPointer(3,4, GL_FLOAT, GL_FALSE, sizeof(VertexNTD), (void*) (12 * sizeof(GLfloat)) );
}

template<>
void VertexBuffer<VertexNC>::setVertexAttributes(){
    glEnableVertexAttribArray( 0 );
    glEnableVertexAttribArray( 1 );
    glEnableVertexAttribArray( 2 );
    glEnableVertexAttribArray( 3 );

    glVertexAttribPointer(0,4, GL_FLOAT, GL_FALSE, sizeof(VertexNC), NULL );
    glVertexAttribPointer(1,4, GL_FLOAT, GL_FALSE, sizeof(VertexNC), (void*) (4 * sizeof(GLfloat)) );
    glVertexAttribPointer(2,4, GL_FLOAT, GL_FALSE, sizeof(VertexNC), (void*) (8 * sizeof(GLfloat)) );
    glVertexAttribPointer(3,4, GL_FLOAT, GL_FALSE, sizeof(VertexNC), (void*) (12 * sizeof(GLfloat)) );
}



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
    return VertexNTD::operator==(other) && data == other.data;
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
