#pragma once

#include "libhello/util/glm.h"
#include "libhello/opengl/vertexBuffer.h"


class Particle
{
public:
    enum Orientation{
        BILLBOARD = 0,
        VELOCITY,
        FIXED
    };

    vec3 position = vec3(0);
    vec4 velocity = vec4(0); //normalized velocity x,y,z in worldspace. w is the scale factor
    vec3 force = vec3(0); //force on the particle. for example gravity
    vec3 right = vec3(1,0,0); //still unused
    float radius=1;
    float lifetime = 0; //lifetime in seconds
    float scale = 0; //upscaling over time. increases the radius by 'scale' per second
    float fadetime = 0; //time when fading begins. if(lifetime==fadetime) -> no fading at all

    int start = 0; //start tick
    int image = 0; //texture from texture array
    int orientation = BILLBOARD;
    Particle();

    void createFixedBillboard(const vec3& normal, float angle);
    void createBillboard(float angle);
}/*__attribute__((packed))*/;



template<>
void VertexBuffer<Particle>::setVertexAttributes();
