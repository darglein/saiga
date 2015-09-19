#include "saiga/rendering/particles/particle.h"



Particle::Particle()
{
}

void Particle::createFixedBillboard(const vec3 &normal, float angle)
{
    velocity = vec4(glm::normalize(normal),0);
    right = glm::rotate(quat(),angle,vec3(velocity)) * glm::cross(vec3(0.236027,-0.0934642,0.967241),normal);

    //    cout<<glm::normalize(vec3(3.1314,-1.24,12.8325))<<endl;
    right = glm::normalize(right);

    orientation = FIXED;
}

void Particle::createBillboard(float angle)
{
    right = vec3(glm::sin(angle),glm::cos(angle),0);
    orientation = BILLBOARD;
}




template<>
void VertexBuffer<Particle>::setVertexAttributes(){
    glEnableVertexAttribArray( 0 );
    glEnableVertexAttribArray( 1 );
    glEnableVertexAttribArray( 2 );
    glEnableVertexAttribArray( 3 );
    glEnableVertexAttribArray( 4 );
    glEnableVertexAttribArray( 5 );
    glEnableVertexAttribArray( 6 );
    glEnableVertexAttribArray( 7 );


    glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE,  sizeof(Particle), NULL );
    glVertexAttribPointer(1,4, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (3 * sizeof(GLfloat)) );
    glVertexAttribPointer(2,3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (7 * sizeof(GLfloat)) );
    glVertexAttribPointer(3,3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (10 * sizeof(GLfloat)) );
    glVertexAttribPointer(4,4, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (13 * sizeof(GLfloat)) );
    glVertexAttribPointer(5,4, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (17 * sizeof(GLfloat)) );
    glVertexAttribPointer(6,1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*) (21 * sizeof(GLfloat)) );
    glVertexAttribIPointer(7,3, GL_INT, sizeof(Particle), (void*) (22 * sizeof(GLfloat)) );
}
