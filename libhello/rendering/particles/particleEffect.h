#pragma once

#include "libhello/rendering/object3d.h"


class ParticleSystem;


class SAIGA_GLOBAL ParticleEffect : public Object3D{
public:
    float velocity = 1.0f;
    float radius = 0.1f;
    float lifetime = 1.0f;
public:

    virtual void apply(ParticleSystem& ps) = 0;
};

