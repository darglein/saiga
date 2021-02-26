/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/vertex.h"

namespace Saiga
{
template <>
void VertexBuffer<Vertex>::setVertexAttributes()
{
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), NULL);
}


template <>
void VertexBuffer<VertexN>::setVertexAttributes()
{
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(VertexN), NULL);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexN), (void*)(4 * sizeof(GLfloat)));
}


template <>
void VertexBuffer<VertexC>::setVertexAttributes()
{
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(VertexC), NULL);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexC), (void*)(4 * sizeof(GLfloat)));
}

template <>
void VertexBuffer<VertexNT>::setVertexAttributes()
{
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(VertexNT), NULL);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexNT), (void*)(4 * sizeof(GLfloat)));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexNT), (void*)(8 * sizeof(GLfloat)));
}

template <>
void VertexBuffer<VertexNTD>::setVertexAttributes()
{
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);


    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(VertexNTD), NULL);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexNTD), (void*)(4 * sizeof(GLfloat)));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexNTD), (void*)(8 * sizeof(GLfloat)));
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(VertexNTD), (void*)(12 * sizeof(GLfloat)));
}

template <>
void VertexBuffer<VertexNC>::setVertexAttributes()
{
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(VertexNC), NULL);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexNC), (void*)(4 * sizeof(GLfloat)));
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(VertexNC), (void*)(8 * sizeof(GLfloat)));
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(VertexNC), (void*)(12 * sizeof(GLfloat)));
}

template <>
void VertexBuffer<BoneVertexCD>::setVertexAttributes()
{
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);
    glEnableVertexAttribArray(5);


    // bone indices + weights
    glVertexAttribIPointer(4, 4, GL_INT, sizeof(BoneVertexCD), (void*)(0 * sizeof(GLfloat)));
    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(BoneVertexCD), (void*)(4 * sizeof(GLfloat)));

    // position normal
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(BoneVertexCD), (void*)(8 * sizeof(GLfloat)));
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(BoneVertexCD), (void*)(12 * sizeof(GLfloat)));

    // color data
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(BoneVertexCD), (void*)(16 * sizeof(GLfloat)));
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(BoneVertexCD), (void*)(20 * sizeof(GLfloat)));
}



}  // namespace Saiga
