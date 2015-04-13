#include "animation/boneVertex.h"

template<>
void VertexBuffer<BoneVertex>::setVertexAttributes(){
    glEnableVertexAttribArray( 0 );
    glEnableVertexAttribArray( 1 );
    glEnableVertexAttribArray( 2 );
    glEnableVertexAttribArray( 3 );
    glEnableVertexAttribArray( 4 );

    glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE, sizeof(BoneVertex), NULL );
    glVertexAttribPointer(1,3, GL_FLOAT, GL_FALSE, sizeof(BoneVertex), (void*) (3 * sizeof(GLfloat)) );
    glVertexAttribPointer(2,2, GL_FLOAT, GL_FALSE, sizeof(BoneVertex), (void*) (6 * sizeof(GLfloat)) );

    glVertexAttribPointer(3,4, GL_FLOAT, GL_FALSE, sizeof(BoneVertex), (void*) (8 * sizeof(GLfloat)) );
    glVertexAttribPointer(4,4, GL_FLOAT, GL_FALSE, sizeof(BoneVertex), (void*) (12 * sizeof(GLfloat)) );
}


template<>
void VertexBuffer<BoneVertexNC>::setVertexAttributes(){
    glEnableVertexAttribArray( 0 );
    glEnableVertexAttribArray( 1 );
    glEnableVertexAttribArray( 2 );
    glEnableVertexAttribArray( 3 );
    glEnableVertexAttribArray( 4 );
    glEnableVertexAttribArray( 5 );

    glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE, sizeof(BoneVertexNC), NULL );
    glVertexAttribPointer(1,3, GL_FLOAT, GL_FALSE, sizeof(BoneVertexNC), (void*) (3 * sizeof(GLfloat)) );
    glVertexAttribPointer(2,3, GL_FLOAT, GL_FALSE, sizeof(BoneVertexNC), (void*) (6 * sizeof(GLfloat)) );
    glVertexAttribPointer(3,3, GL_FLOAT, GL_FALSE, sizeof(BoneVertexNC), (void*) (9 * sizeof(GLfloat)) );

    glVertexAttribPointer(4,4, GL_FLOAT, GL_FALSE, sizeof(BoneVertexNC), (void*) (12 * sizeof(GLfloat)) );
    glVertexAttribPointer(5,4, GL_FLOAT, GL_FALSE, sizeof(BoneVertexNC), (void*) (16 * sizeof(GLfloat)) );
}


