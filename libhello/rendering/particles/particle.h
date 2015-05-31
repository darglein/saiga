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

private:
    vec3 right = vec3(1,0,0); //right vector when orientation is BILLBARD or FIXED

    vec4 scale = vec4(1,1,0,0); //x,y: initial size. z,w upscale over time
public:
    float drag = 0;
    float lifetime = 0; //lifetime in seconds
    float fadetime = 0; //time when fading begins. if(lifetime==fadetime) -> no fading at all
    float specular = 1.0f;

    int start = 0; //start tick
    int image = 0; //texture from texture array
    int orientation = BILLBOARD;
public:
    Particle();

    void createFixedBillboard(const vec3& normal, float angle);
    void createBillboard(float angle);


    //uniform scaled
    void setScale(float radius, float upscale=0);
    //non uniform scaled
    void setScale(const vec2& scale, const vec2& upscale=vec2(0));

    void setVelocity(const vec3& v);

};/*__attribute__((packed))*/

inline void Particle::setScale(float radius, float upscale){
    scale = vec4(radius,radius,upscale,upscale);
}

inline void Particle::setScale(const vec2 &scale, const vec2 &upscale){
    this->scale = vec4(scale,upscale);
}

inline void Particle::setVelocity(const vec3 &v){
    float l = glm::length(v);
    velocity = vec4(v/l,l);
}



template<>
void VertexBuffer<Particle>::setVertexAttributes();
