#pragma once

#include "libhello/util/glm.h"
#include "libhello/opengl/vertexBuffer.h"


class Particle
{
public:
    vec3 position = vec3(0);
    vec3 velocity = vec3(0,1,0);
    vec3 force = vec3(0,1,0);
    vec4 color = vec4(1);
    vec2 age_mass = vec2(0,1);
    float radius=1;
    Particle();
}/*__attribute__((packed))*/;



template<>
void VertexBuffer<Particle>::setVertexAttributes();
