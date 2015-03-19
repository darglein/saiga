#include "rendering/particles/particle.h"



Particle::Particle()
{
}

template<>
void VertexBuffer<Particle>::setVertexAttributes(){
    glEnableVertexAttribArray( 0 );
    glEnableVertexAttribArray( 1 );
    glEnableVertexAttribArray( 2 );
    glEnableVertexAttribArray( 3 );
        glEnableVertexAttribArray( 4 );
        glEnableVertexAttribArray( 5 );


    glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE,  sizeof(Particle), NULL );
    glVertexAttribPointer(1,4, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (3 * sizeof(GLfloat)) );
    glVertexAttribPointer(2,3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (7 * sizeof(GLfloat)) );
    glVertexAttribPointer(3,3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (10 * sizeof(GLfloat)) );
    glVertexAttribPointer(4,4, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (13 * sizeof(GLfloat)) );
    glVertexAttribIPointer(5,3, GL_INT, sizeof(Particle), (void*) (17 * sizeof(GLfloat)) );
}
