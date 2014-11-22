#include "opengl/vertexBuffer.h"


template<>
void VertexBuffer<Vertex>::setVertexAttributes(){
    glEnableVertexAttribArray( 0 );

    glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE, sizeof(Vertex), NULL );
}


template<>
void VertexBuffer<VertexN>::setVertexAttributes(){
    glEnableVertexAttribArray( 0 );
    glEnableVertexAttribArray( 1 );

    glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE, sizeof(VertexN), NULL );
    glVertexAttribPointer(1,3, GL_FLOAT, GL_FALSE, sizeof(VertexN), (void*) (3 * sizeof(GLfloat)) );
}

template<>
void VertexBuffer<VertexNT>::setVertexAttributes(){
    glEnableVertexAttribArray( 0 );
    glEnableVertexAttribArray( 1 );
    glEnableVertexAttribArray( 2 );

    glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE, sizeof(VertexNT), NULL );
    glVertexAttribPointer(1,3, GL_FLOAT, GL_FALSE, sizeof(VertexNT), (void*) (3 * sizeof(GLfloat)) );
    glVertexAttribPointer(2,2, GL_FLOAT, GL_FALSE, sizeof(VertexNT), (void*) (6 * sizeof(GLfloat)) );
}

