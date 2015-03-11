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

    //    glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE,  sizeof(Particle), NULL );
    //    glVertexAttribPointer(1,3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (3 * sizeof(GLfloat)) );
    //    glVertexAttribPointer(2,3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (6 * sizeof(GLfloat)) );
    //    glVertexAttribPointer(3,4, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (9 * sizeof(GLfloat)) );
    //    glVertexAttribPointer(4,3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (13 * sizeof(GLfloat)) );

    glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE,  sizeof(Particle), NULL );
    glVertexAttribPointer(1,3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (3 * sizeof(GLfloat)) );
    glVertexAttribPointer(2,4, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (6 * sizeof(GLfloat)) );
    glVertexAttribPointer(3,1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (10 * sizeof(GLfloat)) );
    glVertexAttribIPointer(4,2, GL_INT, sizeof(Particle), (void*) (11 * sizeof(GLfloat)) );
//    glVertexAttribPointer(4,2, GL_INT, GL_FALSE, sizeof(Particle), (void*) (11 * sizeof(GLfloat)) );
}
