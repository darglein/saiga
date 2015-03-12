#pragma once

#include "libhello/util/glm.h"
#include "libhello/opengl/vertexBuffer.h"


class Particle
{
public:
    vec3 position = vec3(0);
    vec3 velocity = vec3(0,0,0);

    vec4 color = vec4(1);
    float radius=1;
    float lifetime = 0;
    int start = 0;
    int image = 0;

    Particle();
}/*__attribute__((packed))*/;



template<>
void VertexBuffer<Particle>::setVertexAttributes();
