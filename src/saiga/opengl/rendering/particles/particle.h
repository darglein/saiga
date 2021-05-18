/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/vertexBuffer.h"
#include "saiga/core/math/math.h"

namespace Saiga
{
class SAIGA_OPENGL_API Particle
{
   public:
    enum Orientation
    {
        BILLBOARD = 0,
        VELOCITY,
        FIXED
    };

    vec4 position = vec4(0, 0, 0, 1);


    vec4 velocity = make_vec4(0);  // normalized velocity x,y,z in worldspace. w is the scale factor
    vec4 force    = make_vec4(0);  // force on the particle. for example gravity

   private:
    vec4 right = vec4(1, 0, 0, 0);  // right vector when orientation is BILLBARD or FIXED

    vec4 scale = vec4(1, 1, 0, 0);  // x,y: initial size. z,w upscale over time
   public:
    float drag     = 0;
    float lifetime = 0;  // lifetime in seconds
    float fadetime = 0;  // time when fading begins. if(lifetime==fadetime) -> no fading at all
    float specular = 1.0f;

    float startFade = 1.0f;

    int start       = 0;  // start tick
    int image       = 0;  // texture from texture array
    int orientation = BILLBOARD;

   public:
    Particle();

    void createFixedBillboard(const vec3& normal, float angle);
    void createBillboard(float angle);


    // uniform scaled
    void setScale(float radius, float upscale = 0);
    // non uniform scaled
    void setScale(const vec2& scale, const vec2& upscale = vec2(0));

    void setVelocity(const vec3& v);

}; /*__attribute__((packed))*/

inline void Particle::setScale(float radius, float upscale)
{
    scale = vec4(radius, radius, upscale, upscale);
}

inline void Particle::setScale(const vec2& _scale, const vec2& upscale)
{
    this->scale = make_vec4(_scale, upscale);
}

inline void Particle::setVelocity(const vec3& v)
{
    float l  = length(v);
    velocity = make_vec4(v / l, l);
}



template <>
void VertexBuffer<Particle>::setVertexAttributes();

}  // namespace Saiga
